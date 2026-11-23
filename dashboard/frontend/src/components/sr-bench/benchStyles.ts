import { cn, type StatusTone } from '../../ui'
import { active } from './model'

export const benchText = {
  h2: 'text-[15px] font-semibold leading-6 tracking-[-0.012em] text-foreground',
  h3: 'text-[13.5px] font-semibold leading-5 tracking-[-0.006em] text-foreground',
  h4: 'text-[13px] font-medium leading-5 text-foreground',
  body: 'text-[13px] leading-relaxed text-foreground/90',
  muted: 'text-xs leading-relaxed text-muted-foreground',
  field: 'flex min-w-0 flex-col gap-1.5 text-xs font-medium text-muted-foreground',
}

export const benchTable = cn(
  'w-full border-collapse whitespace-nowrap text-left text-xs',
  '[&_thead_th]:h-9 [&_thead_th]:border-b [&_thead_th]:border-border [&_thead_th]:bg-surface-subtle [&_thead_th]:px-3 [&_thead_th]:text-left [&_thead_th]:align-middle [&_thead_th]:text-[11.5px] [&_thead_th]:font-medium [&_thead_th]:text-muted-foreground',
  '[&_tbody_th]:px-3 [&_tbody_th]:py-2.5 [&_tbody_th]:text-left [&_tbody_th]:align-top [&_tbody_th]:font-medium [&_tbody_th]:text-foreground',
  '[&_td]:px-3 [&_td]:py-2.5 [&_td]:align-top [&_td]:text-foreground/90 [&_td]:tabular-nums',
  '[&_tbody_tr]:border-b [&_tbody_tr]:border-border [&_tbody_tr:last-child]:border-b-0 [&_tbody_tr]:transition-colors [&_tbody_tr:hover]:bg-surface-subtle/60',
  '[&_small]:mt-0.5 [&_small]:block [&_small]:max-w-[260px] [&_small]:overflow-hidden [&_small]:text-ellipsis [&_small]:text-[11px] [&_small]:font-normal [&_small]:text-muted-foreground',
  '[&_a]:font-medium [&_a]:text-primary [&_a]:underline-offset-4 hover:[&_a]:underline',
)

/** Definition grid for `<dl><div><dt/><dd/></div></dl>`; dt and dd stay attribute-free. */
export const factsClass =
  'my-3 grid min-w-0 grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-x-6 gap-y-3 [&_dt]:text-xs [&_dt]:text-muted-foreground [&_dd]:mt-0.5 [&_dd]:break-words [&_dd]:text-[13px] [&_dd]:text-foreground [&_dd]:tabular-nums'

/** Definition list of direct `<dt>`/`<dd>` pairs. */
export const identityClass =
  'my-3 grid min-w-0 grid-cols-[minmax(110px,max-content)_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-xs [&_dt]:text-muted-foreground [&_dd]:min-w-0 [&_dd]:break-words [&_dd]:text-foreground [&_code]:font-mono [&_code]:text-[11.5px]'

/** Recorded call fields; nested evidence values collapse to a single column. */
export const evidenceFieldsClass =
  'm-0 grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(220px,100%),1fr))] gap-4 [&>div]:min-w-0 [&>div]:break-words [&_dt]:mb-1 [&_dt]:text-xs [&_dt]:text-muted-foreground [&_dd]:text-[13px] [&_dd]:text-foreground [&_dd]:tabular-nums [&_dl]:grid-cols-1 [&_dl]:gap-3'

/** Recorded prose (responses, messages) keeps its line breaks in the body font. */
export const evidenceTextClass =
  'my-1 max-h-[360px] min-w-0 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-sunken px-3 py-2.5 font-sans text-[13px] leading-relaxed text-foreground/90'

export const codeBlockClass =
  'my-2 max-h-[360px] min-w-0 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-sunken p-3 font-mono text-[11.5px] leading-relaxed text-foreground/90'

export const linkButtonClass =
  'inline-flex max-w-full items-center gap-1 rounded-sm text-left font-medium text-foreground underline-offset-4 outline-none transition-colors hover:text-primary hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40'

export const choiceCardClass =
  'flex min-w-0 cursor-pointer items-start gap-3 rounded-lg border border-border bg-surface px-3.5 py-3 text-[13px] text-foreground shadow-xs transition-colors hover:border-border-strong has-[[data-state=checked]]:border-primary/60 has-[[data-state=checked]]:bg-primary-soft/50 has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-60'

/** Line tabs shared by the view switcher, run sections and dataset sections. */
export const lineTabList =
  'flex min-w-0 items-center gap-1 overflow-x-auto border-b border-border [scrollbar-width:none]'

export const lineTab =
  "relative -mb-px inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-none border-b-2 border-transparent px-2.5 text-[13px] font-medium text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 aria-selected:border-foreground aria-selected:text-foreground aria-[current=page]:border-foreground aria-[current=page]:text-foreground [&_svg:not([class*='size-'])]:size-4"

export const countPill =
  'rounded-full bg-secondary px-1.5 font-mono text-[11px] font-normal leading-[18px] text-muted-foreground tabular-nums'

export function statusTone(status: string): StatusTone {
  if (status === 'completed') return 'success'
  if (active(status)) return 'info'
  if (status === 'failed') return 'danger'
  return 'warning'
}
