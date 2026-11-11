export type Risk = 'safe' | 'destructive' | 'exfiltration' | 'remote_code'

export type Outcome = 'approved' | 'declined'

export type Model = 'haiku' | 'sonnet' | 'opus'

export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max'

// The `steps` question's answer: the expected number of reasoning steps (0 to 2) and how sure the classifier is.
export type Steps = { score: number; confidence: number }

// Where a routed subagent's effort came from: Haiku's fixed medium, the `steps` answer, the band default when that answer was missing or unsure, or the effort the subagent would have had.
export type EffortFrom = 'haiku' | 'steps' | 'unsure' | 'floor' | 'escalated'

// Each application of the one local decision model.
export type App = 'router' | 'escalate' | 'redact' | 'shield' | 'guard' | 'check'

// What a pushback looked like: the feedback the prompt gave, or a repeat of the prior prompt, and the line it crossed.
export type Signal = { kind: 'wrong' | 'different' | 'repeat'; score: number; threshold: number }

// Dollars at list price over the routed subagents' turns, and the session model they were routed from.
export type Spend = { actual: number; baseline: number; turns: number; sessionModel?: string }

// The router's tier lines on the expected difficulty level (0 to 2): below the first easy, from the second on hard.
export type Tiers = { easyBelow: number; hardFrom: number }

// The tag a hidden piece of text is replaced with, without its brackets.
export type Tag = 'name' | 'email' | 'phone' | 'card' | 'ip' | 'date' | 'ssn' | 'iban' | 'address' | 'id' | 'secret'

export type Decision =
  | {
      kind: 'route'
      id: string
      at: string
      toolUseId?: string
      agentId?: string
      task: string
      level: number
      confidence: number
      // The model named for it, or, for an escalation that kept a session model above Opus, the one it ran on.
      model: string
      // The effort the first request ran at, known once it has.
      effort?: string
      // The effort the steps rule picked, before any floor; `effortFrom` says why.
      picked: Effort
      steps: Steps | null
      effortFrom: EffortFrom
      tiers: Tiers
      latencyMs: number
      // Set when the turn was escalated: what the router alone would have done.
      escalated?: { alone: Model | 'default' }
      // The session's model, which the subagent would have run on without the router: the baseline of the savings.
      sessionModel?: string
    }
  | {
      kind: 'keep'
      id: string
      at: string
      toolUseId?: string
      task: string
      level: number
      confidence: number
      model: string
      tiers: Tiers
      // The confidence the move needed: the router's line, or the higher one a smaller model than the session's needs.
      minConfidence: number
      // Why it stayed: the answer was too spread, too unsure for a smaller model, or the goal never uses one.
      reason?: 'unsure' | 'downgrade' | 'quality'
      latencyMs: number
    }
  | {
      kind: 'unavailable'
      app: App
      id: string
      at: string
      // Where the card is drawn: under a tool row, a prompt, or the end of an answer.
      toolUseId?: string
      prompt?: string
      answerTail?: string
      subject: string
      why: string
      // Set when the runtime could not be reached: the band says so once, and no row repeats it.
      offline?: true
      // Another app skipped with this one, for the same reason: shield, when redact's question on an output ran out of
      // time and shield did not ask, so one card says both.
      also?: App
      // Set when the request was cancelled with its call (the user pressed Esc): nothing went unchecked, so it is not kept.
      cancelled?: true
      // Set when the classifier answered but read only the first `chars` of an output of `of` characters.
      read?: { chars: number; of: number }
      // Set when the app waits until done and could not finish: the output was withheld from Claude, not let through.
      withheld?: true
    }
  | {
      kind: 'guard'
      id: string
      at: string
      command: string
      part: string
      parts: number
      verdict: 'pass' | 'ask'
      risk: Risk
      riskProbability: number
      review: number
      threshold: number
      latencyMs: number
      // The parts not checked: past the ones the guard checks one by one, or timed out or answered with an error
      // while the runtime was up. A command with any is held whatever its score.
      unchecked?: number
      // The fixed rule that holds a catastrophic command whatever its score (`alwaysHeld`).
      rule?: string
      outcome?: Outcome
    }
  | {
      kind: 'escalate'
      id: string
      at: string
      prompt: string
      signals: Signal[]
      // Every `user_feedback` probability, and the reask score; null where the question failed or was not asked.
      feedback: Record<string, number> | null
      reask: number | null
      latencyMs: number
    }
  | {
      kind: 'redact'
      id: string
      at: string
      toolUseId: string
      tool: string
      source: string
      counts: Partial<Record<Tag, number>>
      // The lowest probability of a span the classifier had hidden, 1 when only rules found what was hidden.
      score: number
      threshold: number
      latencyMs: number
      // The characters at the end of the output the classifier did not read in its time.
      unread?: number
      // The host of a remote runtime, which the originals went to; absent for one on this machine.
      checkedOn?: string
    }
  | {
      kind: 'shield'
      id: string
      at: string
      toolUseId: string
      tool: string
      source: string
      score: number
      threshold: number
      latencyMs: number
    }
  | {
      kind: 'check'
      id: string
      at: string
      verdict: 'grounded' | 'unverified'
      score: number
      threshold: number
      files: number
      commands: number
      // The turn's prompt, and the end of its answer: the last drawn block of the answer is found by it.
      prompt: string
      answerTail: string
      latencyMs: number
    }

// A pushback in force until the turn that answers it ends.
export type Escalation = { prompt: string; signals: Signal[]; turnId?: string }

// What the main loop's turn read so far, for check.
export type TurnReads = { prompt: string; files: number; commands: number; context: string }

// The decision log against the last calibration: when it was (null before the first), the decisions logged since, and whether this session reminded.
export type Calibration = { at: string | null; since: number; isReminded: boolean }

declare module 'claude-code' {
  interface PluginState {
    vsr: {
      decisions: Shaped<Decision[]>
      escalation: Escalation | null
      lastPrompt: string | null
      turn: TurnReads
      // The tags redact put in place of values this session; a Write or Edit adding one back is refused.
      hiddenTags: Tag[]
      // Set once the session's check of the runtime found it not answering; cleared by any answer.
      isOffline: boolean
      // When a request last went to a runtime known to be offline.
      offlineTry: number
      // The runtime's model when it answers no `score` question, so the router steps aside; empty otherwise.
      noScoreModel: string
      // The ids of requests vSR gave up on (past their wait) that the runtime may still be answering.
      abandoned: number[]
      calibration: Calibration
      // The answers given in the pane this session, by decision id.
      labels: Record<string, 'right' | 'wrong' | 'too-low' | 'too-high'>
      // What routed subagents cost at list price, and what the same tokens would have cost on the session's model.
      spend: Spend
      // The session model each routed subagent was routed from, by its agent id.
      routedFrom: Record<string, string>
    }
  }
}
