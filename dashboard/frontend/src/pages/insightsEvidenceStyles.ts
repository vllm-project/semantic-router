// The class names the routing evidence of a record shares: its tables, its
// disclosures and the metric lists inside them. Each element keeps its plain
// markup, so a table, a summary or a header cell reads the same to a screen
// reader and a test, and the container styles it.

export const EVIDENCE = 'flex min-w-0 flex-col gap-3'

export const INTRO = 'text-xs leading-relaxed text-muted-foreground'

export const MUTED = 'text-xs text-subtle-foreground'

export const PILL =
  'inline-flex h-5 shrink-0 items-center rounded-full bg-secondary px-2 text-[11px] font-medium text-muted-foreground'

export const CARD =
  'flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface px-3.5 py-3'

export const TABLE_WRAP = 'max-w-full overflow-x-auto rounded-lg border border-border'

export const TABLE = [
  'w-full border-collapse text-[12.5px]',
  '[&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:align-top [&_th]:font-medium',
  '[&_td]:px-3 [&_td]:py-1.5 [&_td]:align-top [&_td]:font-mono [&_td]:text-xs [&_td]:tabular',
  '[&_thead_th]:whitespace-nowrap [&_thead_th]:bg-surface-subtle [&_thead_th]:text-xs [&_thead_th]:text-subtle-foreground',
  '[&_tbody_tr]:border-t [&_tbody_tr]:border-border [&_tbody_th]:font-normal',
  '[&_small]:ml-1.5 [&_small]:text-[11px] [&_small]:text-subtle-foreground',
].join(' ')

export const DETAIL = [
  'group/detail min-w-0 rounded-lg border border-border bg-surface-subtle/40',
  '[&>summary]:flex [&>summary]:cursor-pointer [&>summary]:list-none [&>summary]:items-center [&>summary]:justify-between [&>summary]:gap-2',
  '[&>summary]:rounded-lg [&>summary]:px-3 [&>summary]:py-2 [&>summary]:text-xs [&>summary]:font-medium [&>summary]:text-muted-foreground',
  '[&>summary:hover]:text-foreground [&>summary::-webkit-details-marker]:hidden',
  '[&>summary:focus-visible]:outline-none [&>summary:focus-visible]:ring-[3px] [&>summary:focus-visible]:ring-ring/40',
].join(' ')

export const DETAIL_BODY = 'flex min-w-0 flex-col gap-3 border-t border-border px-3 py-3'

export const CHEVRON = 'size-3.5 shrink-0 transition-transform group-open/detail:rotate-180'

export const METRICS = [
  'grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3',
  '[&_dt]:text-xs [&_dt]:text-subtle-foreground [&_dd]:text-[13px] [&_dd]:font-medium [&_dd]:text-foreground',
].join(' ')

export const RAW =
  'max-h-80 overflow-auto rounded-md bg-surface-sunken px-3 py-2 font-mono text-[11.5px] leading-relaxed'
