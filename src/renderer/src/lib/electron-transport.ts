import type { UseStreamTransport } from '@langchain/langgraph-sdk/react'
import type { StreamPayload, StreamEvent, IPCEvent, IPCStreamEvent } from '../../../types'

// Types for serialized LangGraph message chunks (LangChain serialization format)
interface SerializedMessageChunk {
  lc?: number
  type?: string
  // LangChain serialization: id array like ['langchain_core', 'messages', 'AIMessageChunk']
  id?: string[]
  // Actual message data is in kwargs
  kwargs?: {
    content?: string | Array<{ type: string; text?: string }>
    id?: string
    tool_call_chunks?: Array<{ id?: string; name?: string; args?: string }>
  }
}

interface MessageMetadata {
  langgraph_node?: string
}

/**
 * Custom transport for useStream that uses Electron IPC instead of HTTP.
 * This allows useStream to work seamlessly in an Electron app where the
 * LangGraph agent runs in the main process.
 */
export class ElectronIPCTransport implements UseStreamTransport {
  // Track current message ID for grouping tokens across chunks
  private currentMessageId: string | null = null

  async stream(payload: StreamPayload): Promise<AsyncGenerator<StreamEvent>> {
    // Reset state for new stream
    this.currentMessageId = null
    // Extract thread ID from config
    const threadId = payload.config?.configurable?.thread_id
    if (!threadId) {
      return this.createErrorGenerator('MISSING_THREAD_ID', 'Thread ID is required')
    }

    // Extract the message content from input
    const input = payload.input as
      | { messages?: Array<{ content: string; type: string }> }
      | null
      | undefined
    const messages = input?.messages ?? []
    const lastHumanMessage = messages.find((m) => m.type === 'human')
    const messageContent = lastHumanMessage?.content ?? ''

    if (!messageContent) {
      return this.createErrorGenerator('MISSING_MESSAGE', 'Message content is required')
    }

    // Create an async generator that bridges IPC events
    return this.createStreamGenerator(threadId, messageContent, payload.command, payload.signal)
  }

  private async *createErrorGenerator(code: string, message: string): AsyncGenerator<StreamEvent> {
    yield {
      event: 'error',
      data: { error: code, message }
    }
  }

  private async *createStreamGenerator(
    threadId: string,
    message: string,
    command: unknown,
    signal: AbortSignal
  ): AsyncGenerator<StreamEvent> {
    // Create a queue to buffer events from IPC
    const eventQueue: StreamEvent[] = []
    let resolveNext: ((value: StreamEvent | null) => void) | null = null
    let isDone = false
    let hasError = false

    // Generate a run ID for this stream
    const runId = crypto.randomUUID()

    // Emit metadata event first to establish run context
    yield {
      event: 'metadata',
      data: {
        run_id: runId,
        thread_id: threadId
      }
    }

    // Start the stream via IPC
    const cleanup = window.api.agent.streamAgent(threadId, message, command, (ipcEvent) => {
      // Convert IPC events to SDK format
      const sdkEvents = this.convertToSDKEvents(ipcEvent as IPCEvent, threadId)

      for (const sdkEvent of sdkEvents) {
        console.log('[Transport] Converted event:', sdkEvent)

        if (sdkEvent.event === 'done' || sdkEvent.event === 'error') {
          isDone = true
          hasError = sdkEvent.event === 'error'
        }

        // If someone is waiting for the next event, resolve immediately
        if (resolveNext) {
          const resolve = resolveNext
          resolveNext = null
          resolve(sdkEvent)
        } else {
          // Otherwise queue the event
          eventQueue.push(sdkEvent)
        }
      }
    })

    // Handle abort signal
    if (signal) {
      signal.addEventListener('abort', () => {
        cleanup()
        isDone = true
        if (resolveNext) {
          const resolve = resolveNext
          resolveNext = null
          resolve(null)
        }
      })
    }

    // Yield events as they come in
    while (!isDone || eventQueue.length > 0) {
      // Check for queued events first
      if (eventQueue.length > 0) {
        const event = eventQueue.shift()!
        if (event.event === 'done') {
          break
        }
        if (event.event !== 'error' || hasError) {
          yield event
        }
        if (hasError) {
          break
        }
        continue
      }

      // Wait for the next event
      const event = await new Promise<StreamEvent | null>((resolve) => {
        resolveNext = resolve
      })

      if (event === null) {
        break
      }

      if (event.event === 'done') {
        break
      }

      yield event

      if (event.event === 'error') {
        break
      }
    }
  }

  /**
   * Convert IPC events to LangGraph SDK format
   * Returns an array since a single IPC event may produce multiple SDK events
   */
  private convertToSDKEvents(event: IPCEvent, threadId: string): StreamEvent[] {
    const events: StreamEvent[] = []

    switch (event.type) {
      // Raw stream events from LangGraph - parse and convert
      case 'stream':
        events.push(...this.processStreamEvent(event))
        break

      // Legacy: Token streaming for real-time typing effect
      case 'token':
        events.push({
          event: 'messages',
          data: [
            { id: event.messageId, type: 'ai', content: event.token },
            { langgraph_node: 'agent' }
          ]
        })
        break

      // Legacy: Tool call chunks
      case 'tool_call':
        events.push({
          event: 'custom',
          data: {
            type: 'tool_call',
            messageId: event.messageId,
            tool_calls: event.tool_calls
          }
        })
        break

      // Legacy: Full state values
      case 'values': {
        const { todos, files, workspacePath, subagents, interrupt } = event.data

        // Only emit values event if todos is defined
        // Avoid emitting { todos: [] } when undefined, which would wipe out existing todos
        if (todos !== undefined) {
          events.push({
            event: 'values',
            data: { todos }
          })
        }

        // Emit files/workspace
        if (files) {
          const filesList = Array.isArray(files)
            ? files
            : Object.entries(files).map(([path, data]) => ({
                path,
                is_dir: false,
                size:
                  typeof (data as { content?: string })?.content === 'string'
                    ? (data as { content: string }).content.length
                    : undefined
              }))

          if (filesList.length) {
            events.push({
              event: 'custom',
              data: { type: 'workspace', files: filesList, path: workspacePath || '/' }
            })
          }
        }

        // Emit subagents
        if (subagents?.length) {
          events.push({
            event: 'custom',
            data: { type: 'subagents', subagents }
          })
        }

        // Emit interrupt
        if (interrupt) {
          events.push({
            event: 'custom',
            data: {
              type: 'interrupt',
              request: {
                id: interrupt.id || crypto.randomUUID(),
                tool_call: interrupt.tool_call,
                allowed_decisions: ['approve', 'reject', 'edit']
              }
            }
          })
        }
        break
      }

      case 'error':
        events.push({
          event: 'error',
          data: { error: 'STREAM_ERROR', message: event.error }
        })
        break

      case 'done':
        events.push({
          event: 'done',
          data: { thread_id: threadId }
        })
        break
    }

    return events
  }

  /**
   * Process raw LangGraph stream events (mode + data tuples)
   */
  private processStreamEvent(event: IPCStreamEvent): StreamEvent[] {
    const events: StreamEvent[] = []
    const { mode, data } = event

    if (mode === 'messages') {
      // Messages mode returns [message, metadata] tuples
      const [msgChunk, metadata] = data as [SerializedMessageChunk, MessageMetadata]

      // Detect AI message chunks via id array (LangChain serialization format)
      // id is an array like ['langchain_core', 'messages', 'AIMessageChunk']
      const isAIMessageChunk = msgChunk?.id?.some(
        (id) => id === 'AIMessageChunk' || id === 'AIMessage'
      )

      if (isAIMessageChunk && msgChunk.kwargs) {
        const content = this.extractContent(msgChunk.kwargs.content)

        if (content) {
          // Track message ID for grouping tokens (from kwargs.id)
          const msgId = msgChunk.kwargs.id || this.currentMessageId || crypto.randomUUID()
          this.currentMessageId = msgId

          console.log('[Transport] Processing token:', content.substring(0, 50))
          events.push({
            event: 'messages',
            data: [
              { id: msgId, type: 'ai', content },
              { langgraph_node: metadata?.langgraph_node || 'agent' }
            ]
          })
        }

        // Handle tool calls in the chunk (from kwargs.tool_call_chunks)
        if (msgChunk.kwargs.tool_call_chunks?.length) {
          events.push({
            event: 'custom',
            data: {
              type: 'tool_call',
              messageId: this.currentMessageId,
              tool_calls: msgChunk.kwargs.tool_call_chunks
            }
          })
        }
      }
    } else if (mode === 'values') {
      // Values mode returns full state
      const state = data as {
        messages?: Array<{
          id?: string[]
          kwargs?: {
            id?: string
            content?: string | Array<{ type: string; text?: string }>
            type?: string
          }
        }>
        todos?: { id?: string; content?: string; status?: string }[]
        files?: Record<string, unknown> | Array<{ path: string; is_dir?: boolean; size?: number }>
        workspacePath?: string
        subagents?: Array<{
          id?: string
          name?: string
          description?: string
          status?: string
          startedAt?: Date | string
          completedAt?: Date | string
        }>
        __interrupt__?: { id?: string; tool_call?: unknown }
      }

      // Transform messages from LangChain serialization format to SDK format
      // LangChain format: { id: ['langchain_core', 'messages', 'AIMessageChunk'], kwargs: { id, content, ... } }
      // SDK format: { id: string, type: 'ai'|'human', content: string }
      // Filter out human messages - they're already in the UI from when the user sent them
      const transformedMessages = state.messages
        ?.map((msg) => {
          // Determine message type from the class name in id array
          const className = msg.id?.[msg.id.length - 1] || ''
          const type = className.toLowerCase().includes('human')
            ? 'human'
            : className.toLowerCase().includes('ai')
              ? 'ai'
              : className.toLowerCase().includes('tool')
                ? 'tool'
                : 'ai'

          // Extract content from kwargs
          const content = this.extractContent(msg.kwargs?.content)

          return {
            id: msg.kwargs?.id || crypto.randomUUID(),
            type,
            content
          }
        })
        .filter((msg) => msg.type !== 'human')

      events.push({
        event: 'values',
        data: {
          messages: transformedMessages,
          todos: state.todos,
          workspacePath: state.workspacePath
        }
      })

      // Emit files/workspace
      if (state.files) {
        const filesList = Array.isArray(state.files)
          ? state.files
          : Object.entries(state.files).map(([path, fileData]) => ({
              path,
              is_dir: false,
              size:
                typeof (fileData as { content?: string })?.content === 'string'
                  ? (fileData as { content: string }).content.length
                  : undefined
            }))

        if (filesList.length) {
          events.push({
            event: 'custom',
            data: { type: 'workspace', files: filesList, path: state.workspacePath || '/' }
          })
        }
      }

      // Emit subagents
      if (state.subagents?.length) {
        events.push({
          event: 'custom',
          data: { type: 'subagents', subagents: state.subagents }
        })
      }

      // Emit interrupt
      if (state.__interrupt__) {
        events.push({
          event: 'custom',
          data: {
            type: 'interrupt',
            request: {
              id: state.__interrupt__.id || crypto.randomUUID(),
              tool_call: state.__interrupt__.tool_call,
              allowed_decisions: ['approve', 'reject', 'edit']
            }
          }
        })
      }
    }

    return events
  }

  /**
   * Extract text content from message content (string or content blocks)
   */
  private extractContent(
    content: string | Array<{ type: string; text?: string }> | undefined
  ): string {
    if (typeof content === 'string') {
      return content
    }
    if (Array.isArray(content)) {
      return content
        .filter((block): block is { type: 'text'; text: string } => block.type === 'text')
        .map((block) => block.text)
        .join('')
    }
    return ''
  }
}
