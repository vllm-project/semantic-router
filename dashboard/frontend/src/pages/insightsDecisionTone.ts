// The kind of decision a name reads as, which sets the badge's color and icon.
export type Tone = 'guard' | 'privacy' | 'deep' | 'fast' | 'route'

const TONES: [RegExp, Tone][] = [
  [/shield|guard|jailbreak|injection|block|refus|harm/, 'guard'],
  [/pii|mask|privacy|redact|secret|private/, 'privacy'],
  [/slow|hard|deep|reason|escalat|complex/, 'deep'],
  [/fast|quick|easy|light|simple|cheap/, 'fast'],
]

export function decisionTone(decision: string): Tone {
  const name = decision.toLowerCase()
  return TONES.find(([pattern]) => pattern.test(name))?.[1] ?? 'route'
}
