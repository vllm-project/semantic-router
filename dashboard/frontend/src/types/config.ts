export type DecisionConditionType =
  | 'keyword'
  | 'domain'
  | 'preference'
  | 'user_feedback'
  | 'reask'
  | 'embedding'
  | 'fact_check'
  | 'language'
  | 'context'
  | 'structure'
  | 'complexity'
  | 'modality'
  | 'authz'
  | 'jailbreak'
  | 'pii'
  | 'kb'
  | 'conversation'
  | 'event'
  | 'metadata'
  | 'classifier'
  | 'projection'

export interface Listener {
  name: string
  address: string
  port: number
  timeout?: string
}
