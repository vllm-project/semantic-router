import React from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  AddSignalFormState,
  ConfigData,
  ComplexitySignal,
  ContextSignal,
  DomainSignal,
  EmbeddingSignal,
  FactCheckSignal,
  JailbreakSignal,
  KeywordSignal,
  LanguageSignal,
  ModalitySignal,
  PIISignal,
  PreferenceSignal,
  RoleBindingSignal,
  SignalType,
  Subject,
  UserFeedbackSignal,
} from './configPageSupport'
import { formatThreshold } from './configPageSupport'
import { hasFlatSignals } from '../types/config'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'

interface ConfigPageSignalsSectionProps {
  config: ConfigData | null
  isPythonCLI: boolean
  isReadonly: boolean
  signalsSearch: string
  onSignalsSearchChange: (value: string) => void
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
  listInputToArray: (input: string) => string[]
  removeSignalByName: (cfg: ConfigData, type: SignalType, targetName: string) => void
}

type UnifiedSignalData = Partial<
  KeywordSignal &
  EmbeddingSignal &
  DomainSignal &
  PreferenceSignal &
  FactCheckSignal &
  UserFeedbackSignal &
  LanguageSignal &
  ContextSignal &
  ComplexitySignal &
  ModalitySignal &
  RoleBindingSignal &
  JailbreakSignal &
  PIISignal
>

interface UnifiedSignal {
  name: string
  type: SignalType
  summary: string
  rawData: UnifiedSignalData
}

export default function ConfigPageSignalsSection({
  config,
  isPythonCLI,
  isReadonly,
  signalsSearch,
  onSignalsSearchChange,
  saveConfig,
  openEditModal,
  openViewModal,
  listInputToArray,
  removeSignalByName,
}: ConfigPageSignalsSectionProps) {
  const signals = config?.signals
  const flatSignals: ConfigData['signals'] | null = !signals && hasFlatSignals(config) ? {
    keywords: config?.keyword_rules,
    embeddings: config?.embedding_rules,
    domains: (config?.categories || []).map((category) => ({
      name: category.name,
      description: category.description || '',
      mmlu_categories: category.mmlu_categories,
    })),
    fact_check: config?.fact_check_rules,
    user_feedbacks: config?.user_feedback_rules,
    preferences: config?.preference_rules,
    language: config?.language_rules,
    context: config?.context_rules,
    complexity: config?.complexity_rules,
    modality: undefined,
    role_bindings: undefined,
    jailbreak: config?.jailbreak,
    pii: config?.pii,
  } : null
  const effectiveSignals = signals || flatSignals

  const allSignals: UnifiedSignal[] = []

  effectiveSignals?.keywords?.forEach(kw => {
    allSignals.push({
      name: kw.name,
      type: 'Keywords',
      summary: `${kw.operator}, ${kw.keywords.length} keywords${kw.case_sensitive ? ', case-sensitive' : ''}`,
      rawData: kw
    })
  })

  effectiveSignals?.embeddings?.forEach(emb => {
    allSignals.push({
      name: emb.name,
      type: 'Embeddings',
      summary: `Threshold: ${Math.round(emb.threshold * 100)}%, ${emb.candidates.length} items, ${emb.aggregation_method}`,
      rawData: emb
    })
  })

  effectiveSignals?.domains?.forEach(domain => {
    const categoryCount = domain.mmlu_categories?.length || 0
    allSignals.push({
      name: domain.name,
      type: 'Domain',
      summary: categoryCount > 0 ? `${categoryCount} MMLU categories` : (domain.description || 'No description'),
      rawData: domain
    })
  })

  effectiveSignals?.preferences?.forEach(pref => {
    const examplesCount = pref.examples?.length || 0
    const thresholdText = pref.threshold !== undefined ? ` • threshold ${formatThreshold(pref.threshold)}` : ''
    const examplesText = examplesCount > 0 ? ` • ${examplesCount} ${examplesCount === 1 ? 'example' : 'examples'}` : ''
    allSignals.push({
      name: pref.name,
      type: 'Preference',
      summary: `${pref.description || 'No description'}${examplesText}${thresholdText}`,
      rawData: pref
    })
  })

  effectiveSignals?.fact_check?.forEach(fc => {
    allSignals.push({
      name: fc.name,
      type: 'Fact Check',
      summary: fc.description || 'No description',
      rawData: fc
    })
  })

  effectiveSignals?.user_feedbacks?.forEach(uf => {
    allSignals.push({
      name: uf.name,
      type: 'User Feedback',
      summary: uf.description || 'No description',
      rawData: uf
    })
  })

  effectiveSignals?.language?.forEach(lang => {
    allSignals.push({
      name: lang.name,
      type: 'Language',
      summary: 'Language detection rule',
      rawData: lang
    })
  })

  effectiveSignals?.context?.forEach(ctx => {
    allSignals.push({
      name: ctx.name,
      type: 'Context',
      summary: `${ctx.min_tokens} to ${ctx.max_tokens} tokens`,
      rawData: ctx
    })
  })

  effectiveSignals?.complexity?.forEach(comp => {
    const hardCount = comp.hard?.candidates?.length || 0
    const easyCount = comp.easy?.candidates?.length || 0
    allSignals.push({
      name: comp.name,
      type: 'Complexity',
      summary: `Threshold: ${comp.threshold}, ${hardCount} hard / ${easyCount} easy candidates`,
      rawData: comp
    })
  })

  effectiveSignals?.modality?.forEach(modality => {
    allSignals.push({
      name: modality.name,
      type: 'Modality',
      summary: modality.description || 'Modality signal',
      rawData: modality,
    })
  })

  effectiveSignals?.role_bindings?.forEach(binding => {
    const subjectCount = binding.subjects?.length || 0
    allSignals.push({
      name: binding.name,
      type: 'Authz',
      summary: `${binding.role} • ${subjectCount} ${subjectCount === 1 ? 'subject' : 'subjects'}`,
      rawData: binding,
    })
  })

  effectiveSignals?.jailbreak?.forEach(jb => {
    const method = jb.method || 'classifier'
    allSignals.push({
      name: jb.name,
      type: 'Jailbreak',
      summary: `Method: ${method}, Threshold: ${jb.threshold}${jb.include_history ? ', includes history' : ''}`,
      rawData: jb
    })
  })

  effectiveSignals?.pii?.forEach(p => {
    const allowed = p.pii_types_allowed?.length || 0
    allSignals.push({
      name: p.name,
      type: 'PII',
      summary: `Threshold: ${p.threshold}${allowed > 0 ? `, ${allowed} types allowed` : ', deny all'}`,
      rawData: p
    })
  })

  const filteredSignals = allSignals.filter(signal =>
    signal.name.toLowerCase().includes(signalsSearch.toLowerCase()) ||
    signal.type.toLowerCase().includes(signalsSearch.toLowerCase()) ||
    signal.summary.toLowerCase().includes(signalsSearch.toLowerCase())
  )

  const signalsColumns: Column<UnifiedSignal>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>
    },
    {
      key: 'type',
      header: 'Type',
      width: '160px',
      sortable: true,
      render: (row) => <span className={styles.tableMetaBadge}>{row.type}</span>
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (row) => (
        <span style={{
          fontSize: '0.875rem',
          color: 'var(--color-text-secondary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          display: 'block'
        }}>
          {row.summary}
        </span>
      )
    }
  ]

  const handleViewSignal = (signal: UnifiedSignal) => {
    const sections: ViewSection[] = []

    sections.push({
      title: 'Basic Information',
      fields: [
        { label: 'Name', value: signal.name },
        { label: 'Type', value: signal.type },
        { label: 'Summary', value: signal.summary, fullWidth: true }
      ]
    })

    if (signal.type === 'Keywords') {
      sections.push({
        title: 'Keywords Configuration',
        fields: [
          { label: 'Operator', value: signal.rawData.operator },
          { label: 'Case Sensitive', value: signal.rawData.case_sensitive ? 'Yes' : 'No' },
          {
            label: 'Keywords',
            value: (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {(signal.rawData.keywords || []).map((kw: string, i: number) => (
                  <span key={i} style={{
                    padding: '0.25rem 0.75rem',
                    background: 'rgba(118, 185, 0, 0.1)',
                    borderRadius: '4px',
                    fontSize: '0.875rem',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    {kw}
                  </span>
                ))}
              </div>
            ),
            fullWidth: true
          }
        ]
      })
    } else if (signal.type === 'Embeddings') {
      sections.push({
        title: 'Embeddings Configuration',
        fields: [
          { label: 'Threshold', value: `${Math.round((signal.rawData.threshold || 0) * 100)}%` },
          { label: 'Aggregation Method', value: signal.rawData.aggregation_method },
          {
            label: 'Candidates',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {(signal.rawData.candidates || []).map((c: string, i: number) => (
                  <div key={i} style={{
                    padding: '0.5rem',
                    background: 'rgba(0, 212, 255, 0.1)',
                    borderRadius: '4px',
                    fontSize: '0.875rem'
                  }}>
                    {c}
                  </div>
                ))}
              </div>
            ),
            fullWidth: true
          }
        ]
      })
    } else if (signal.type === 'Domain') {
      sections.push({
        title: 'Domain Configuration',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          {
            label: 'MMLU Categories',
            value: signal.rawData.mmlu_categories?.length ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {signal.rawData.mmlu_categories.map((cat: string, i: number) => (
                  <span key={i} style={{
                    padding: '0.25rem 0.75rem',
                    background: 'rgba(147, 51, 234, 0.1)',
                    borderRadius: '4px',
                    fontSize: '0.875rem'
                  }}>
                    {cat}
                  </span>
                ))}
              </div>
            ) : 'No categories',
            fullWidth: true
          }
        ]
      })
    } else if (signal.type === 'Preference') {
      sections.push({
        title: 'Preference Configuration',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          { label: 'Threshold', value: signal.rawData.threshold !== undefined ? formatThreshold(signal.rawData.threshold) : 'Not set' },
          {
            label: 'Examples',
            value: signal.rawData.examples?.length ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>
                {signal.rawData.examples.map((ex: string, i: number) => (
                  <div key={i} style={{ padding: '0.35rem 0.5rem', background: 'rgba(234, 179, 8, 0.1)', borderRadius: 6 }}>
                    {ex}
                  </div>
                ))}
              </div>
            ) : 'No examples provided',
            fullWidth: true
          }
        ]
      })
    } else if (signal.type === 'Language') {
      sections.push({
        title: 'Language Signal',
        fields: [
          { label: 'Language Code', value: signal.rawData.name || 'N/A', fullWidth: true },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
        ]
      })
    } else if (signal.type === 'Context') {
      sections.push({
        title: 'Context Signal',
        fields: [
          { label: 'Min Tokens', value: signal.rawData.min_tokens || 'N/A', fullWidth: true },
          { label: 'Max Tokens', value: signal.rawData.max_tokens || 'N/A', fullWidth: true },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
        ]
      })
    } else if (signal.type === 'Complexity') {
      const fields: Array<{ label: string; value: React.ReactNode; fullWidth?: boolean }> = [
        { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true }
      ]

      if (signal.rawData.composer) {
        fields.push({
          label: 'Composer',
          value: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div><strong>Operator:</strong> {signal.rawData.composer.operator}</div>
              <div><strong>Conditions:</strong></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginLeft: '1rem' }}>
                {signal.rawData.composer.conditions.map((cond: { type: string; name: string }, i: number) => (
                  <div key={i} style={{
                    padding: '0.5rem',
                    background: 'rgba(255, 165, 0, 0.1)',
                    borderRadius: '4px',
                    fontSize: '0.875rem',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    {cond.type}: {cond.name}
                  </div>
                ))}
              </div>
            </div>
          ),
          fullWidth: true
        })
      }

      fields.push(
        {
          label: 'Hard Candidates',
          value: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              {(signal.rawData.hard?.candidates || []).map((c: string, i: number) => (
                <div key={i}>• {c}</div>
              ))}
            </div>
          ),
          fullWidth: true
        },
        {
          label: 'Easy Candidates',
          value: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              {(signal.rawData.easy?.candidates || []).map((c: string, i: number) => (
                <div key={i}>• {c}</div>
              ))}
            </div>
          ),
          fullWidth: true
        },
        { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
      )

      sections.push({
        title: 'Complexity Signal',
        fields
      })
    } else if (signal.type === 'Modality') {
      sections.push({
        title: 'Modality Signal',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ]
      })
    } else if (signal.type === 'Authz') {
      sections.push({
        title: 'Role Binding',
        fields: [
          { label: 'Role', value: signal.rawData.role || 'N/A' },
          {
            label: 'Subjects',
            value: signal.rawData.subjects?.length ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                {signal.rawData.subjects.map((subject: Subject, i: number) => (
                  <div key={i}>{subject.kind}:{subject.name}</div>
                ))}
              </div>
            ) : 'No subjects',
            fullWidth: true,
          },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ]
      })
    } else if (signal.type === 'Jailbreak') {
      const fields = [
        { label: 'Method', value: signal.rawData.method || 'classifier', fullWidth: true },
        { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true },
        { label: 'Include History', value: signal.rawData.include_history ? 'Yes' : 'No', fullWidth: true },
      ]
      if (signal.rawData.method === 'contrastive') {
        fields.push(
          { label: 'Jailbreak Patterns', value: (signal.rawData.jailbreak_patterns || []).length + ' patterns', fullWidth: true },
          { label: 'Benign Patterns', value: (signal.rawData.benign_patterns || []).length + ' patterns', fullWidth: true },
        )
      }
      fields.push({ label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true })
      sections.push({ title: 'Jailbreak Signal', fields })
    } else if (signal.type === 'PII') {
      sections.push({
        title: 'PII Signal',
        fields: [
          { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true },
          { label: 'Allowed PII Types', value: signal.rawData.pii_types_allowed?.join(', ') || 'None (deny all)', fullWidth: true },
          { label: 'Include History', value: signal.rawData.include_history ? 'Yes' : 'No', fullWidth: true },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
        ]
      })
    } else {
      sections.push({
        title: 'Details',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
        ]
      })
    }

    openViewModal(`Signal: ${signal.name}`, sections, () => handleEditSignal(signal))
  }

  const openSignalEditor = (mode: 'add' | 'edit', signal?: UnifiedSignal) => {
    const defaultForm: AddSignalFormState = {
      type: 'Keywords',
      name: '',
      description: '',
      operator: 'AND',
      keywords: '',
      case_sensitive: false,
      threshold: 0.8,
      candidates: '',
      aggregation_method: 'mean',
      mmlu_categories: '',
      preference_examples: '',
      preference_threshold: undefined,
      min_tokens: '0',
      max_tokens: '8K',
      complexity_threshold: 0.1,
      role: '',
      subjects: '',
      hard_candidates: '',
      easy_candidates: '',
      composer_operator: 'AND',
      composer_conditions: '',
      jailbreak_threshold: 0.65,
      jailbreak_method: 'classifier',
      include_history: false,
      jailbreak_patterns: '',
      benign_patterns: '',
      pii_threshold: 0.5,
      pii_types_allowed: '',
      pii_include_history: false
    }

    const initialData: AddSignalFormState = mode === 'edit' && signal ? {
      type: signal.type,
      name: signal.name,
      description: signal.rawData.description || '',
      operator: signal.rawData.operator || 'AND',
      keywords: (signal.rawData.keywords || []).join('\n'),
      case_sensitive: !!signal.rawData.case_sensitive,
      threshold: signal.rawData.threshold ?? 0.8,
      candidates: (signal.rawData.candidates || []).join('\n'),
      aggregation_method: signal.rawData.aggregation_method || 'mean',
      mmlu_categories: (signal.rawData.mmlu_categories || []).join('\n'),
      preference_examples: (signal.rawData.examples || []).join('\n'),
      preference_threshold: signal.rawData.threshold,
      min_tokens: signal.rawData.min_tokens || '0',
      max_tokens: signal.rawData.max_tokens || '8K',
      complexity_threshold: signal.rawData.threshold ?? 0.1,
      role: signal.type === 'Authz' ? signal.rawData.role || '' : '',
      subjects: signal.type === 'Authz'
        ? (signal.rawData.subjects || []).map((subject: Subject) => `${subject.kind}:${subject.name}`).join('\n')
        : '',
      hard_candidates: (signal.rawData.hard?.candidates || []).join('\n'),
      easy_candidates: (signal.rawData.easy?.candidates || []).join('\n'),
      composer_operator: signal.rawData.composer?.operator || 'AND',
      composer_conditions: signal.rawData.composer?.conditions?.map((c: { type: string; name: string }) => `${c.type}:${c.name}`).join('\n') || '',
      jailbreak_threshold: signal.rawData.threshold ?? 0.65,
      jailbreak_method: signal.rawData.method || 'classifier',
      include_history: !!signal.rawData.include_history,
      jailbreak_patterns: (signal.rawData.jailbreak_patterns || []).join('\n'),
      benign_patterns: (signal.rawData.benign_patterns || []).join('\n'),
      pii_threshold: signal.rawData.threshold ?? 0.5,
      pii_types_allowed: (signal.rawData.pii_types_allowed || []).join('\n'),
      pii_include_history: !!signal.rawData.include_history
    } : defaultForm

    const conditionallyHideFieldExceptType = (type: SignalType) => {
      return (formData: AddSignalFormState) => formData.type !== type
    }

    const fields: FieldConfig<AddSignalFormState>[] = [
      {
        name: 'type',
        label: 'Type',
        type: 'select',
        options: ['Keywords', 'Embeddings', 'Domain', 'Preference', 'Fact Check', 'User Feedback', 'Language', 'Context', 'Complexity', 'Modality', 'Authz', 'Jailbreak', 'PII'],
        required: true,
        description: 'Fields are validated based on the selected type.'
      },
      {
        name: 'name',
        label: 'Name',
        type: 'text',
        required: true,
        placeholder: 'Enter a unique signal name here'
      },
      {
        name: 'description',
        label: 'Description',
        type: 'textarea',
        placeholder: 'Optional description for this signal'
      },
      {
        name: 'preference_examples',
        label: 'Examples (preference only)',
        type: 'textarea',
        placeholder: 'One example per line to represent this preference',
        description: 'Few-shot hints sent to the contrastive preference classifier.',
        shouldHide: conditionallyHideFieldExceptType('Preference')
      },
      {
        name: 'preference_threshold',
        label: 'Threshold (preference only)',
        type: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        placeholder: 'e.g., 0.35',
        description: 'Override the global preference threshold for this specific rule.',
        shouldHide: conditionallyHideFieldExceptType('Preference')
      },
      {
        name: 'operator',
        label: 'Operator (keywords only)',
        type: 'select',
        options: ['AND', 'OR'],
        description: 'Used when type is Keywords',
        shouldHide: conditionallyHideFieldExceptType('Keywords')
      },
      {
        name: 'case_sensitive',
        label: 'Case Sensitive (keywords only)',
        type: 'boolean',
        description: 'Whether keyword matching is case sensitive',
        shouldHide: conditionallyHideFieldExceptType('Keywords')
      },
      {
        name: 'keywords',
        label: 'Keywords',
        type: 'textarea',
        placeholder: 'Comma or newline separated keywords',
        shouldHide: conditionallyHideFieldExceptType('Keywords')
      },
      {
        name: 'threshold',
        label: 'Threshold (embeddings only)',
        type: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        placeholder: '0.80',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      },
      {
        name: 'aggregation_method',
        label: 'Aggregation Method (embeddings only)',
        type: 'text',
        placeholder: 'mean',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      },
      {
        name: 'candidates',
        label: 'Candidates (embeddings only)',
        type: 'textarea',
        placeholder: 'One candidate per line or comma separated',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      },
      {
        name: 'mmlu_categories',
        label: 'MMLU Categories (domains only)',
        type: 'textarea',
        placeholder: 'Comma or newline separated categories',
        shouldHide: conditionallyHideFieldExceptType('Domain')
      },
      {
        name: 'min_tokens',
        label: 'Minimum Tokens (context only)',
        type: 'text',
        placeholder: 'e.g., 0, 8K, 1M',
        description: 'Minimum token count (supports K/M suffixes)',
        shouldHide: conditionallyHideFieldExceptType('Context')
      },
      {
        name: 'max_tokens',
        label: 'Maximum Tokens (context only)',
        type: 'text',
        placeholder: 'e.g., 8K, 1024K',
        description: 'Maximum token count (supports K/M suffixes)',
        shouldHide: conditionallyHideFieldExceptType('Context')
      },
      {
        name: 'complexity_threshold',
        label: 'Threshold (complexity only)',
        type: 'number',
        placeholder: 'e.g., 0.1',
        description: 'Similarity difference threshold for hard/easy classification',
        shouldHide: conditionallyHideFieldExceptType('Complexity')
      },
      {
        name: 'composer_operator',
        label: 'Composer Operator (complexity only)',
        type: 'select',
        options: ['AND', 'OR'],
        description: 'Logical operator for composer conditions (recommended to filter based on other signals)',
        shouldHide: conditionallyHideFieldExceptType('Complexity')
      },
      {
        name: 'composer_conditions',
        label: 'Composer Conditions (complexity only)',
        type: 'textarea',
        placeholder: 'One condition per line in format type:name, e.g.:\ndomain:computer science\nkeyword:coding',
        description: 'Filter this complexity signal based on other signals (RECOMMENDED). Format: type:name per line',
        shouldHide: conditionallyHideFieldExceptType('Complexity')
      },
      {
        name: 'hard_candidates',
        label: 'Hard Candidates (complexity only)',
        type: 'textarea',
        placeholder: 'One candidate per line, e.g.:\ndesign distributed system\nimplement consensus algorithm',
        description: 'Phrases representing hard/complex queries',
        shouldHide: conditionallyHideFieldExceptType('Complexity')
      },
      {
        name: 'easy_candidates',
        label: 'Easy Candidates (complexity only)',
        type: 'textarea',
        placeholder: 'One candidate per line, e.g.:\nprint hello world\nloop through array',
        description: 'Phrases representing easy/simple queries',
        shouldHide: conditionallyHideFieldExceptType('Complexity')
      },
      {
        name: 'role',
        label: 'Role (authz only)',
        type: 'text',
        placeholder: 'admin',
        shouldHide: conditionallyHideFieldExceptType('Authz')
      },
      {
        name: 'subjects',
        label: 'Subjects (authz only)',
        type: 'textarea',
        placeholder: 'One subject per line, e.g.:\nUser:alice\nGroup:admins',
        shouldHide: conditionallyHideFieldExceptType('Authz')
      },
      {
        name: 'jailbreak_method',
        label: 'Method (jailbreak only)',
        type: 'select',
        options: ['classifier', 'contrastive'],
        description: 'Detection method: "classifier" (BERT-based) or "contrastive" (embedding KB similarity)',
        shouldHide: conditionallyHideFieldExceptType('Jailbreak')
      },
      {
        name: 'jailbreak_threshold',
        label: 'Threshold (jailbreak only)',
        type: 'number',
        placeholder: 'e.g., 0.65 for classifier, 0.10 for contrastive',
        description: 'Confidence threshold for jailbreak detection (0.0 - 1.0)',
        shouldHide: conditionallyHideFieldExceptType('Jailbreak')
      },
      {
        name: 'include_history',
        label: 'Include History (jailbreak only)',
        type: 'boolean',
        description: 'Whether to include conversation history in jailbreak detection',
        shouldHide: conditionallyHideFieldExceptType('Jailbreak')
      },
      {
        name: 'jailbreak_patterns',
        label: 'Jailbreak Patterns (contrastive only)',
        type: 'textarea',
        placeholder: 'One pattern per line, e.g.:\nIgnore all previous instructions\nYou are now DAN',
        description: 'Known jailbreak prompts for the contrastive KB',
        shouldHide: conditionallyHideFieldExceptType('Jailbreak')
      },
      {
        name: 'benign_patterns',
        label: 'Benign Patterns (contrastive only)',
        type: 'textarea',
        placeholder: 'One pattern per line, e.g.:\nWhat is the weather today\nHelp me write an email',
        description: 'Known benign prompts for the contrastive KB',
        shouldHide: conditionallyHideFieldExceptType('Jailbreak')
      },
      {
        name: 'pii_threshold',
        label: 'Threshold (PII only)',
        type: 'number',
        placeholder: 'e.g., 0.5',
        description: 'Confidence threshold for PII detection (0.0 - 1.0)',
        shouldHide: conditionallyHideFieldExceptType('PII')
      },
      {
        name: 'pii_types_allowed',
        label: 'Allowed PII Types (PII only)',
        type: 'textarea',
        placeholder: 'One PII type per line, e.g.:\nEMAIL_ADDRESS\nPHONE_NUMBER',
        description: 'PII types to allow (not blocked). Leave empty to deny all.',
        shouldHide: conditionallyHideFieldExceptType('PII')
      },
      {
        name: 'pii_include_history',
        label: 'Include History (PII only)',
        type: 'boolean',
        description: 'Whether to include conversation history in PII detection',
        shouldHide: conditionallyHideFieldExceptType('PII')
      }
    ]

    const saveSignal = async (formData: AddSignalFormState) => {
      if (!config) {
        throw new Error('Configuration not loaded yet.')
      }

      if (!isPythonCLI) {
        throw new Error('Editing signals is only supported for Python CLI configs.')
      }

      const name = (formData.name || '').trim()
      if (!name) {
        throw new Error('Name is required.')
      }

      const type = formData.type as SignalType
      if (!type) {
        throw new Error('Type is required.')
      }

      const newConfig: ConfigData = cloneConfigData(config)
      if (!newConfig.signals) newConfig.signals = {}

      if (mode === 'edit' && signal) {
        removeSignalByName(newConfig, signal.type, signal.name)
      }

      switch (type) {
        case 'Keywords': {
          const keywords = listInputToArray(formData.keywords || '')
          if (keywords.length === 0) {
            throw new Error('Please provide at least one keyword.')
          }
          newConfig.signals.keywords = [
            ...(newConfig.signals.keywords || []),
            {
              name,
              operator: formData.operator,
              keywords,
              case_sensitive: !!formData.case_sensitive
            }
          ]
          break
        }
        case 'Embeddings': {
          const candidates = listInputToArray(formData.candidates || '')
          if (candidates.length === 0) {
            throw new Error('Please provide at least one candidate string.')
          }
          const threshold = Number.isFinite(formData.threshold)
            ? Math.max(0, Math.min(1, formData.threshold))
            : 0
          newConfig.signals.embeddings = [
            ...(newConfig.signals.embeddings || []),
            {
              name,
              threshold,
              candidates,
              aggregation_method: formData.aggregation_method || 'mean'
            }
          ]
          break
        }
        case 'Domain': {
          const mmlu_categories = listInputToArray(formData.mmlu_categories || '')
          newConfig.signals.domains = [
            ...(newConfig.signals.domains || []),
            {
              name,
              description: formData.description,
              mmlu_categories
            }
          ]
          break
        }
        case 'Preference': {
          const examples = listInputToArray(formData.preference_examples || '')
          const hasThreshold = Number.isFinite(formData.preference_threshold)
          const threshold = hasThreshold ? Math.max(0, Math.min(1, Number(formData.preference_threshold))) : undefined

          const preferenceRule: { name: string; description: string; examples?: string[]; threshold?: number } = {
            name,
            description: formData.description || ''
          }

          if (examples.length > 0) {
            preferenceRule.examples = examples
          }

          if (threshold !== undefined && threshold > 0) {
            preferenceRule.threshold = threshold
          }

          newConfig.signals.preferences = [
            ...(newConfig.signals.preferences || []),
            preferenceRule
          ]
          break
        }
        case 'Fact Check': {
          newConfig.signals.fact_check = [
            ...(newConfig.signals.fact_check || []),
            {
              name,
              description: formData.description
            }
          ]
          break
        }
        case 'User Feedback': {
          newConfig.signals.user_feedbacks = [
            ...(newConfig.signals.user_feedbacks || []),
            {
              name,
              description: formData.description
            }
          ]
          break
        }
        case 'Language': {
          newConfig.signals.language = [
            ...(newConfig.signals.language || []),
            {
              name
            }
          ]
          break
        }
        case 'Context': {
          const min_tokens = (formData.min_tokens || '0').trim()
          const max_tokens = (formData.max_tokens || '8K').trim()
          if (!min_tokens || !max_tokens) {
            throw new Error('Both min_tokens and max_tokens are required.')
          }
          newConfig.signals.context = [
            ...(newConfig.signals.context || []),
            {
              name,
              min_tokens,
              max_tokens,
              description: formData.description || undefined
            }
          ]
          break
        }
        case 'Complexity': {
          const complexity_threshold = formData.complexity_threshold ?? 0.1
          const hard_candidates = (formData.hard_candidates || '').trim()
          const easy_candidates = (formData.easy_candidates || '').trim()

          if (!hard_candidates || !easy_candidates) {
            throw new Error('Both hard and easy candidates are required.')
          }

          const hardList = hard_candidates.split('\n').map(c => c.trim()).filter(c => c.length > 0)
          const easyList = easy_candidates.split('\n').map(c => c.trim()).filter(c => c.length > 0)

          if (hardList.length === 0 || easyList.length === 0) {
            throw new Error('Both hard and easy candidates must have at least one entry.')
          }

          const composerConditionsText = (formData.composer_conditions || '').trim()
          let composer = undefined
          if (composerConditionsText) {
            const conditions = composerConditionsText
              .split('\n')
              .map(line => line.trim())
              .filter(line => line.length > 0)
              .map(line => {
                const parts = line.split(':')
                if (parts.length !== 2) {
                  throw new Error(`Invalid composer condition format: "${line}". Expected format: type:name`)
                }
                return {
                  type: parts[0].trim(),
                  name: parts[1].trim()
                }
              })

            if (conditions.length > 0) {
              composer = {
                operator: formData.composer_operator || 'AND',
                conditions
              }
            }
          }

          newConfig.signals.complexity = [
            ...(newConfig.signals.complexity || []),
            {
              name,
              threshold: complexity_threshold,
              hard: {
                candidates: hardList
              },
              easy: {
                candidates: easyList
              },
              description: formData.description || undefined,
              ...(composer && { composer })
            }
          ]
          break
        }
        case 'Modality': {
          newConfig.signals.modality = [
            ...(newConfig.signals.modality || []),
            {
              name,
              description: formData.description || undefined,
            }
          ]
          break
        }
        case 'Authz': {
          const role = (formData.role || '').trim()
          if (!role) {
            throw new Error('Role is required for authz signals.')
          }
          const subjects = listInputToArray(formData.subjects || '').map((entry) => {
            const [kindRaw, ...nameParts] = entry.split(':')
            const kind = (kindRaw || '').trim()
            const subjectName = nameParts.join(':').trim()
            if ((kind !== 'User' && kind !== 'Group') || !subjectName) {
              throw new Error(`Invalid authz subject "${entry}". Expected User:name or Group:name.`)
            }
            return { kind, name: subjectName } as Subject
          })
          if (subjects.length === 0) {
            throw new Error('At least one subject is required for authz signals.')
          }
          newConfig.signals.role_bindings = [
            ...(newConfig.signals.role_bindings || []),
            {
              name,
              role,
              subjects,
              description: formData.description || undefined,
            }
          ]
          break
        }
        case 'Jailbreak': {
          const jailbreak_threshold = formData.jailbreak_threshold ?? 0.65
          if (jailbreak_threshold < 0 || jailbreak_threshold > 1) {
            throw new Error('Jailbreak threshold must be between 0.0 and 1.0.')
          }
          const method = formData.jailbreak_method || 'classifier'
          const jailbreakEntry: JailbreakSignal = {
            name,
            threshold: jailbreak_threshold,
            include_history: formData.include_history || false,
            description: formData.description || undefined
          }
          if (method !== 'classifier') {
            jailbreakEntry.method = method
          }
          if (method === 'contrastive') {
            const jailbreakPatterns = (formData.jailbreak_patterns || '').trim()
            const benignPatternsText = (formData.benign_patterns || '').trim()
            if (jailbreakPatterns) {
              jailbreakEntry.jailbreak_patterns = jailbreakPatterns.split('\n').map((p: string) => p.trim()).filter((p: string) => p.length > 0)
            }
            if (benignPatternsText) {
              jailbreakEntry.benign_patterns = benignPatternsText.split('\n').map((p: string) => p.trim()).filter((p: string) => p.length > 0)
            }
          }
          newConfig.signals.jailbreak = [
            ...(newConfig.signals.jailbreak || []),
            jailbreakEntry
          ]
          break
        }
        case 'PII': {
          const pii_threshold = formData.pii_threshold ?? 0.5
          if (pii_threshold < 0 || pii_threshold > 1) {
            throw new Error('PII threshold must be between 0.0 and 1.0.')
          }
          const pii_types_allowed = (formData.pii_types_allowed || '').trim()
          const allowedList = pii_types_allowed
            ? pii_types_allowed.split('\n').map(t => t.trim()).filter(t => t.length > 0)
            : undefined
          newConfig.signals.pii = [
            ...(newConfig.signals.pii || []),
            {
              name,
              threshold: pii_threshold,
              pii_types_allowed: allowedList,
              include_history: formData.pii_include_history || false,
              description: formData.description || undefined
            }
          ]
          break
        }
        default:
          throw new Error('Unsupported signal type.')
      }

      await saveConfig(newConfig)
    }

    openEditModal<AddSignalFormState>(
      mode === 'add' ? 'Add Signal' : `Edit Signal: ${signal?.name}`,
      initialData,
      fields,
      saveSignal,
      mode
    )
  }

  const handleEditSignal = (signal: UnifiedSignal) => {
    openSignalEditor('edit', signal)
  }

  const handleDeleteSignal = async (signal: UnifiedSignal) => {
    if (confirm(`Are you sure you want to delete signal "${signal.name}"?`)) {
      if (!config || !isPythonCLI) {
        alert('Deleting signals is only supported for Python CLI configs.')
        return
      }

      const newConfig: ConfigData = cloneConfigData(config)
      removeSignalByName(newConfig, signal.type, signal.name)

      await saveConfig(newConfig)
    }
  }

  return (
    <ConfigPageManagerLayout title="Signals" description="Review the signal catalog that drives semantic routing, guardrails, and context-aware behavior.">
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <TableHeader title="Signals" count={allSignals.length} searchPlaceholder="Search signals..." searchValue={signalsSearch} onSearchChange={onSignalsSearchChange} onAdd={() => openSignalEditor('add')} addButtonText="Add Signal" disabled={isReadonly} variant="embedded" />
          <DataTable
            columns={signalsColumns}
            data={filteredSignals}
            keyExtractor={(row) => `${row.type}-${row.name}`}
            onView={handleViewSignal}
            onEdit={handleEditSignal}
            onDelete={handleDeleteSignal}
            emptyMessage={signalsSearch ? 'No signals match your search' : 'No signals configured'}
            className={styles.managerTable}
            readonly={isReadonly}
          />
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
