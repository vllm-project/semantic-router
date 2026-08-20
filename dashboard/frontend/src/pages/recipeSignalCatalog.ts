export interface RecipeSignalFamily {
  key: string
  type: string
}

// Keep this aligned with config.SupportedSignalTypes. Projections are composed
// in their own tab and are therefore intentionally not treated as signals here.
export const RECIPE_SIGNAL_FAMILIES: readonly RecipeSignalFamily[] = [
  { key: 'keywords', type: 'Keyword' },
  { key: 'embeddings', type: 'Embedding' },
  { key: 'domains', type: 'Domain' },
  { key: 'fact_check', type: 'Fact check' },
  { key: 'user_feedbacks', type: 'User feedback' },
  { key: 'reasks', type: 'Reask' },
  { key: 'preferences', type: 'Preference' },
  { key: 'language', type: 'Language' },
  { key: 'context', type: 'Context' },
  { key: 'structure', type: 'Structure' },
  { key: 'complexity', type: 'Complexity' },
  { key: 'modality', type: 'Modality' },
  { key: 'role_bindings', type: 'Authz' },
  { key: 'jailbreak', type: 'Jailbreak' },
  { key: 'pii', type: 'PII' },
  { key: 'kb', type: 'Knowledge base' },
  { key: 'conversation', type: 'Conversation' },
  { key: 'events', type: 'Event' },
  { key: 'metadata', type: 'Metadata' },
  { key: 'classifiers', type: 'Classifier' },
]
