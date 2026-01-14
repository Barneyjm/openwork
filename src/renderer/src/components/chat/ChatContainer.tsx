import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { Send, Square, Loader2 } from 'lucide-react'
import { useStream } from '@langchain/langgraph-sdk/react'
import type { InferAgentState } from 'langchain'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useAppStore } from '@/lib/store'
import { MessageBubble } from './MessageBubble'
import { ModelSwitcher } from './ModelSwitcher'
import { WorkspacePicker } from './WorkspacePicker'
import { ApprovalDialog } from '@/components/hitl/ApprovalDialog'
import { ElectronIPCTransport } from '@/lib/electron-transport'
import type { Message } from '@/types'
import type { DeepAgent } from '../../../../main/agent/types'

interface ChatContainerProps {
  threadId: string
}

// Define custom event data types
interface FileEventData {
  path: string
  is_dir?: boolean
  size?: number
}

interface SubagentEventData {
  id?: string
  name?: string
  description?: string
  status?: string
  startedAt?: Date
  completedAt?: Date
}

interface MessageEventData {
  id?: string
  type?: string
  role?: string
  content?: string
  tool_calls?: unknown[]
  created_at?: Date
}

interface CustomEventData {
  type?: string
  message?: MessageEventData
  files?: FileEventData[]
  path?: string
  subagents?: SubagentEventData[]
  request?: unknown
}

export function ChatContainer({ threadId }: ChatContainerProps): React.JSX.Element {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const isAtBottomRef = useRef(true)

  const {
    messages: storeMessages,
    pendingApproval,
    setTodos,
    setWorkspaceFiles,
    setWorkspacePath,
    setSubagents,
    setPendingApproval,
    appendMessage,
    loadThreads,
    generateTitleForFirstMessage
  } = useAppStore()

  // Create transport instance (memoized to avoid recreating)
  const transport = useMemo(() => new ElectronIPCTransport(), [])

  // Handle custom events from the stream
  const handleCustomEvent = useCallback(
    (data: CustomEventData): void => {
      console.log('[ChatContainer] Custom event:', data)
      switch (data.type) {
        case 'message':
          if (data.message) {
            const msg = data.message
            const storeMsg: Message = {
              id: msg.id || crypto.randomUUID(),
              role:
                msg.role === 'user' || msg.type === 'human'
                  ? 'user'
                  : msg.role === 'assistant' || msg.type === 'ai'
                    ? 'assistant'
                    : msg.role === 'tool' || msg.type === 'tool'
                      ? 'tool'
                      : 'system',
              content: msg.content || '',
              tool_calls: msg.tool_calls as Message['tool_calls'],
              created_at: msg.created_at ? new Date(msg.created_at) : new Date()
            }
            console.log('[ChatContainer] Adding message:', storeMsg)
            appendMessage(storeMsg)
          }
          break
        case 'workspace':
          if (Array.isArray(data.files)) {
            setWorkspaceFiles(
              data.files.map((f) => ({
                path: f.path,
                is_dir: f.is_dir,
                size: f.size
              }))
            )
          }
          if (data.path) {
            setWorkspacePath(data.path)
          }
          break
        case 'subagents':
          if (Array.isArray(data.subagents)) {
            setSubagents(
              data.subagents.map((s) => ({
                id: s.id || crypto.randomUUID(),
                name: s.name || 'Subagent',
                description: s.description || '',
                status: (s.status || 'pending') as 'pending' | 'running' | 'completed' | 'failed',
                startedAt: s.startedAt,
                completedAt: s.completedAt
              }))
            )
          }
          break
        case 'interrupt':
          if (data.request) {
            setPendingApproval(data.request as Parameters<typeof setPendingApproval>[0])
          }
          break
      }
    },
    [setWorkspaceFiles, setWorkspacePath, setSubagents, setPendingApproval, appendMessage]
  )

  // Use the useStream hook with our custom transport
  const stream = useStream<DeepAgent>({
    transport,
    threadId,
    messagesKey: 'messages',
    onCustomEvent: (data): void => {
      handleCustomEvent(data as CustomEventData)
    },
    onError: (error): void => {
      console.error('[ChatContainer] Stream error:', error)
    }
  })

  // Sync todos from stream state
  const agentValues = stream.values as unknown as InferAgentState<DeepAgent>
  useEffect(() => {
    const todos = agentValues.todos
    if (Array.isArray(todos)) {
      setTodos(
        todos.map((t: { id?: string; content?: string; status?: string }) => ({
          id: t.id || crypto.randomUUID(),
          content: t.content || '',
          status: (t.status || 'pending') as 'pending' | 'in_progress' | 'completed' | 'cancelled'
        }))
      )
    }
  }, [agentValues.todos, setTodos])

  // Persist messages and refresh threads when stream completes
  const prevLoadingRef = useRef(false)
  useEffect(() => {
    if (prevLoadingRef.current && !stream.isLoading) {
      // Stream just completed - persist streaming messages to store
      const streamMsgs = stream.messages || []
      for (const msg of streamMsgs) {
        if (msg.id) {
          const storeMsg: Message = {
            id: msg.id,
            role: (msg.type === 'human' ? 'user' : 'assistant') as Message['role'],
            content: typeof msg.content === 'string' ? msg.content : '',
            created_at: new Date()
          }
          appendMessage(storeMsg)
        }
      }
      loadThreads()
    }
    prevLoadingRef.current = stream.isLoading
  }, [stream.isLoading, stream.messages, loadThreads, appendMessage])

  // Combine store messages with streaming messages
  const displayMessages = useMemo(() => {
    // Get IDs of messages already in the store
    const storeMessageIds = new Set(storeMessages.map((m) => m.id))

    // Get streaming messages that aren't in the store yet
    const streamingMsgs: Message[] = (stream.messages || [])
      .filter((m): m is typeof m & { id: string } => !!m.id && !storeMessageIds.has(m.id))
      .map((m) => ({
        id: m.id,
        role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
        content: typeof m.content === 'string' ? m.content : '',
        created_at: new Date()
      }))

    return [...storeMessages, ...streamingMsgs]
  }, [storeMessages, stream.messages])

  // Get the actual scrollable viewport element from Radix ScrollArea
  const getViewport = useCallback(() => {
    return scrollRef.current?.querySelector(
      '[data-radix-scroll-area-viewport]'
    ) as HTMLDivElement | null
  }, [])

  // Track scroll position to determine if user is at bottom
  const handleScroll = useCallback(() => {
    const viewport = getViewport()
    if (!viewport) return

    const { scrollTop, scrollHeight, clientHeight } = viewport
    // Consider "at bottom" if within 50px of the bottom
    const threshold = 50
    isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < threshold
  }, [getViewport])

  // Attach scroll listener to viewport
  useEffect(() => {
    const viewport = getViewport()
    if (!viewport) return

    viewport.addEventListener('scroll', handleScroll)
    return () => viewport.removeEventListener('scroll', handleScroll)
  }, [getViewport, handleScroll])

  // Auto-scroll on new messages only if already at bottom
  useEffect(() => {
    const viewport = getViewport()
    if (viewport && isAtBottomRef.current) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [displayMessages, stream.isLoading, getViewport])

  // Always scroll to bottom when switching threads
  useEffect(() => {
    const viewport = getViewport()
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight
      isAtBottomRef.current = true
    }
  }, [threadId, getViewport])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [threadId])

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    if (!input.trim() || stream.isLoading) return

    const message = input.trim()
    setInput('')

    // Check if this is the first message (for title generation)
    const isFirstMessage = storeMessages.length === 0

    // Add user message to store immediately
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
      created_at: new Date()
    }
    appendMessage(userMessage)

    // Generate title for first message
    if (isFirstMessage) {
      generateTitleForFirstMessage(threadId, message)
    }

    // Submit via useStream
    await stream.submit(
      {
        messages: [{ type: 'human', content: message }]
      },
      {
        config: {
          configurable: { thread_id: threadId }
        }
      }
    )
  }

  const handleKeyDown = (e: React.KeyboardEvent): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea based on content
  const adjustTextareaHeight = (): void => {
    const textarea = inputRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }

  useEffect(() => {
    adjustTextareaHeight()
  }, [input])

  const handleCancel = async (): Promise<void> => {
    await stream.stop()
  }

  return (
    <div className="flex flex-1 flex-col min-h-0 overflow-hidden">
      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0" ref={scrollRef}>
        <div className="p-4">
          <div className="max-w-3xl mx-auto space-y-4">
            {displayMessages.length === 0 && !stream.isLoading && (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <div className="text-section-header mb-2">NEW THREAD</div>
                <div className="text-sm">Start a conversation with the agent</div>
              </div>
            )}

            {displayMessages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {/* Streaming indicator - only show if no streaming messages yet */}
            {stream.isLoading && stream.messages.length === 0 && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="size-4 animate-spin" />
                Agent is thinking...
              </div>
            )}
          </div>
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex flex-col gap-2">
            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message..."
                disabled={stream.isLoading}
                className="flex-1 min-w-0 resize-none rounded-sm border border-border bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '200px' }}
              />
              <div className="flex items-center justify-center shrink-0 h-12">
                {stream.isLoading ? (
                  <Button type="button" variant="ghost" size="icon" onClick={handleCancel}>
                    <Square className="size-4" />
                  </Button>
                ) : (
                  <Button type="submit" variant="default" size="icon" disabled={!input.trim()}>
                    <Send className="size-4" />
                  </Button>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <ModelSwitcher />
              <div className="w-px h-4 bg-border" />
              <WorkspacePicker />
            </div>
          </div>
        </form>
      </div>

      {/* HITL Approval Dialog */}
      {pendingApproval && <ApprovalDialog request={pendingApproval} />}
    </div>
  )
}
