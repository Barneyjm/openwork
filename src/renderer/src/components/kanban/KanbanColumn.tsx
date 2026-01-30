import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type ColumnStatus = "pending" | "in_progress" | "interrupted" | "done"

interface KanbanColumnProps {
  title: string
  status: ColumnStatus
  count: number
  children: React.ReactNode
}

const columnConfig: Record<
  ColumnStatus,
  { badge: "outline" | "info" | "warning" | "nominal"; borderColor: string }
> = {
  pending: { badge: "outline", borderColor: "border-t-border" },
  in_progress: { badge: "info", borderColor: "border-t-status-info" },
  interrupted: { badge: "warning", borderColor: "border-t-status-warning" },
  done: { badge: "nominal", borderColor: "border-t-status-nominal" }
}

export function KanbanColumn({
  title,
  status,
  count,
  children
}: KanbanColumnProps): React.JSX.Element {
  const config = columnConfig[status]

  return (
    <div
      className={cn(
        "flex flex-col min-w-[200px] w-[200px] flex-1 bg-muted/30 rounded-sm border border-border border-t-2",
        config.borderColor
      )}
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <span className="text-section-header">{title}</span>
        <Badge variant={config.badge}>{count}</Badge>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-2 space-y-2">{children}</div>
      </ScrollArea>
    </div>
  )
}
