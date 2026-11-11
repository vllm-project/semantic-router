const PREFIX = 'systemone:'

// One System One question as the record keeps it: a choice's options with their
// probabilities, or a single value (a noul probability or a score).
export type SystemOneAnswer =
  | { question: string; kind: 'choice'; options: { key: string; probability: number }[] }
  | { question: string; kind: 'value'; value: number }

// The answers in a record's signal values, keyed `systemone:<question>` and, for a
// choice, `systemone:<question>:<option>`.
export function systemOneAnswers(values: Record<string, number> | undefined): SystemOneAnswer[] {
  const questions = new Map<string, { value?: number; options: Map<string, number> }>()
  for (const [key, value] of Object.entries(values ?? {})) {
    if (!key.startsWith(PREFIX) || !Number.isFinite(value)) continue
    const [question, option] = key.slice(PREFIX.length).split(':', 2)
    const entry = questions.get(question) ?? { options: new Map<string, number>() }
    if (option === undefined) entry.value = value
    else entry.options.set(option, value)
    questions.set(question, entry)
  }
  return [...questions.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .flatMap(([question, { value, options }]): SystemOneAnswer[] => {
      if (options.size > 0) {
        const sorted = [...options.entries()].map(([key, probability]) => ({ key, probability }))
        return [{ question, kind: 'choice', options: sorted }]
      }
      return value === undefined ? [] : [{ question, kind: 'value', value }]
    })
}
