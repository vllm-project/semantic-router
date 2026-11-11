const PREFIX = 'systemone:'

type Option = { key: string; probability: number }

// One System One question as the record keeps it. A choice's options sum to 1; a set's
// or span's labels each carry their own probability; a noul is one probability; a
// score is an expected level, which the Router records with a different confidence.
export type SystemOneAnswer =
  | { question: string; kind: 'choice'; options: Option[] }
  | { question: string; kind: 'labels'; options: Option[] }
  | { question: string; kind: 'probability'; value: number }
  | { question: string; kind: 'score'; value: number }

const clamp = (value: number) => Math.min(1, Math.max(0, value))

// The answers in a record's signal values, keyed `systemone:<question>` and, for each
// option or label, `systemone:<question>:<option>`.
export function systemOneAnswers(
  values: Record<string, number> | undefined,
  confidences: Record<string, number> | undefined,
): SystemOneAnswer[] {
  const questions = new Map<string, { value?: number; options: Option[] }>()
  for (const [key, value] of Object.entries(values ?? {})) {
    if (!key.startsWith(PREFIX) || !Number.isFinite(value)) continue
    const rest = key.slice(PREFIX.length)
    const at = rest.indexOf(':')
    const question = at < 0 ? rest : rest.slice(0, at)
    const entry = questions.get(question) ?? { options: [] }
    if (at < 0) entry.value = value
    else entry.options.push({ key: rest.slice(at + 1), probability: clamp(value) })
    questions.set(question, entry)
  }
  return [...questions.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .flatMap(([question, { value, options }]): SystemOneAnswer[] => {
      if (options.length > 0) {
        const total = options.reduce((sum, one) => sum + one.probability, 0)
        const kind = Math.abs(total - 1) <= 0.02 ? 'choice' : 'labels'
        return [{ question, kind, options }]
      }
      if (value === undefined) return []
      const confidence = confidences?.[`${PREFIX}${question}`]
      const isScore =
        (confidence !== undefined && Math.abs(confidence - value) > 1e-9) || value < 0 || value > 1
      return [{ question, kind: isScore ? 'score' : 'probability', value }]
    })
}
