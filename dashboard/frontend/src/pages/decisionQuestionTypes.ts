import type { LucideIcon } from 'lucide-react'
import { CircleDot, Gauge, Highlighter, ListChecks, ToggleLeft } from 'lucide-react'
import { QUESTION_TYPES } from './systemOnePlayground'

export const QUESTION_TYPE_ORDER = QUESTION_TYPES.map((item) => item.type)

export function questionTypeLabel(type: string): string {
  return QUESTION_TYPES.find((item) => item.type === type)?.label ?? type
}

export function questionTypeDetail(type: string): string {
  return QUESTION_TYPES.find((item) => item.type === type)?.detail ?? ''
}

export const QUESTION_ICONS: Record<string, LucideIcon> = {
  choice: CircleDot,
  score: Gauge,
  noul: ToggleLeft,
  set: ListChecks,
  span: Highlighter,
}
