// The kind of decision a name reads as, which sets the badge's color and icon.
export type Tone = 'guard' | 'privacy' | 'deep' | 'fast' | 'route'

// Whole words of the name only, so unblock_x is not a guard and the first kind found wins.
const TONES: [Tone, string[]][] = [
  ['guard', ['shield', 'guard', 'jailbreak', 'injection', 'block', 'harm', 'safety']],
  ['privacy', ['pii', 'mask', 'privacy', 'redact', 'secret', 'private']],
  ['deep', ['slow', 'hard', 'deep', 'reason', 'reasoning', 'escalate', 'complex']],
  ['fast', ['fast', 'quick', 'easy', 'light', 'simple', 'cheap']],
]

export function decisionTone(decision: string): Tone {
  const words = new Set(decision.toLowerCase().split(/[^a-z0-9]+/))
  return TONES.find(([, names]) => names.some((name) => words.has(name)))?.[0] ?? 'route'
}
