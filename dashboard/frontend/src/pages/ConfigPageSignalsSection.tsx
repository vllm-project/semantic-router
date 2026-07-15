import React from 'react'
import styles from './ConfigPage.module.css'
import signalStyles from './ConfigPageSignalsSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import ConfirmDialog from '../components/ConfirmDialog'
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
  KBSignal,
  KeywordSignal,
  LanguageSignal,
  ModalitySignal,
  PIISignal,
  PreferenceSignal,
  ReaskSignal,
  RoleBindingSignal,
  SignalType,
  StructureSignal,
  UserFeedbackSignal,
} from './configPageSupport'
import { formatThreshold } from './configPageSupport'
import { hasFlatSignals } from '../types/config'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'
import { buildSignalFormFields } from './configPageSignalFormFields'
import {
  SignalConditionsEditor,
  SignalStringListEditor,
  SignalStructureFeatureEditor,
  SignalStructurePredicateEditor,
  SignalSubjectsEditor,
} from './configPageSignalStructuredEditors'
import {
  DEFAULT_STRUCTURE_FEATURE,
  DEFAULT_STRUCTURE_PREDICATE,
  getSignalReferenceCount,
  normalizeConditions,
  normalizeStringList,
  normalizeStructureFeature,
  normalizeStructurePredicate,
  normalizeSubjects,
} from './configPageSignalFormSupport'

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
    ReaskSignal &
    LanguageSignal &
    ContextSignal &
    StructureSignal &
    ComplexitySignal &
    ModalitySignal &
    RoleBindingSignal &
    JailbreakSignal &
    PIISignal &
    KBSignal
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
  removeSignalByName,
}: ConfigPageSignalsSectionProps) {
  const [selectedSignalKeys, setSelectedSignalKeys] = React.useState<Set<string>>(new Set())
  const [signalsPendingDelete, setSignalsPendingDelete] = React.useState<UnifiedSignal[]>([])
  const [deletePending, setDeletePending] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [actionError, setActionError] = React.useState<string | null>(null)
  const signals = config?.signals
  const flatSignals: ConfigData['signals'] | null =
    !signals && hasFlatSignals(config)
      ? {
          keywords: config?.keyword_rules,
          embeddings: config?.embedding_rules,
          domains: (config?.categories || []).map((category) => ({
            name: category.name,
            description: category.description || '',
            mmlu_categories: category.mmlu_categories,
          })),
          fact_check: config?.fact_check_rules,
          user_feedbacks: config?.user_feedback_rules,
          reasks: config?.reask_rules,
          preferences: config?.preference_rules,
          language: config?.language_rules,
          context: config?.context_rules,
          structure: config?.structure_rules,
          complexity: config?.complexity_rules,
          modality: undefined,
          role_bindings: undefined,
          jailbreak: config?.jailbreak,
          pii: config?.pii,
        }
      : null
  const effectiveSignals = signals || flatSignals

  const allSignals: UnifiedSignal[] = []

  effectiveSignals?.keywords?.forEach((kw) => {
    allSignals.push({
      name: kw.name,
      type: 'Keywords',
      summary: `${kw.operator}, ${kw.keywords.length} keywords${kw.case_sensitive ? ', case-sensitive' : ''}`,
      rawData: kw,
    })
  })

  effectiveSignals?.embeddings?.forEach((emb) => {
    allSignals.push({
      name: emb.name,
      type: 'Embeddings',
      summary: `Threshold: ${Math.round(emb.threshold * 100)}%, ${emb.candidates.length} items, ${emb.aggregation_method}`,
      rawData: emb,
    })
  })

  effectiveSignals?.domains?.forEach((domain) => {
    const categoryCount = domain.mmlu_categories?.length || 0
    allSignals.push({
      name: domain.name,
      type: 'Domain',
      summary:
        categoryCount > 0
          ? `${categoryCount} MMLU categories`
          : domain.description || 'No description',
      rawData: domain,
    })
  })

  effectiveSignals?.preferences?.forEach((pref) => {
    const examplesCount = pref.examples?.length || 0
    const thresholdText =
      pref.threshold !== undefined ? ` • threshold ${formatThreshold(pref.threshold)}` : ''
    const examplesText =
      examplesCount > 0 ? ` • ${examplesCount} ${examplesCount === 1 ? 'example' : 'examples'}` : ''
    allSignals.push({
      name: pref.name,
      type: 'Preference',
      summary: `${pref.description || 'No description'}${examplesText}${thresholdText}`,
      rawData: pref,
    })
  })

  effectiveSignals?.fact_check?.forEach((fc) => {
    allSignals.push({
      name: fc.name,
      type: 'Fact Check',
      summary: fc.description || 'No description',
      rawData: fc,
    })
  })

  effectiveSignals?.user_feedbacks?.forEach((uf) => {
    allSignals.push({
      name: uf.name,
      type: 'User Feedback',
      summary: uf.description || 'No description',
      rawData: uf,
    })
  })

  effectiveSignals?.reasks?.forEach((reask) => {
    const lookback = reask.lookback_turns ?? 1
    const threshold = reask.threshold ?? 0.8
    allSignals.push({
      name: reask.name,
      type: 'Reask',
      summary: `${reask.description || 'No description'} • lookback ${lookback} • threshold ${formatThreshold(threshold)}`,
      rawData: reask,
    })
  })

  effectiveSignals?.language?.forEach((lang) => {
    allSignals.push({
      name: lang.name,
      type: 'Language',
      summary: 'Language detection rule',
      rawData: lang,
    })
  })

  effectiveSignals?.context?.forEach((ctx) => {
    allSignals.push({
      name: ctx.name,
      type: 'Context',
      summary: `${ctx.min_tokens} to ${ctx.max_tokens} tokens`,
      rawData: ctx,
    })
  })

  effectiveSignals?.structure?.forEach((structure) => {
    allSignals.push({
      name: structure.name,
      type: 'Structure',
      summary: `${structure.feature?.type || 'unknown'} from ${structure.feature?.source?.type || 'unknown'}`,
      rawData: structure,
    })
  })

  effectiveSignals?.complexity?.forEach((comp) => {
    const hardCount = comp.hard?.candidates?.length || 0
    const easyCount = comp.easy?.candidates?.length || 0
    allSignals.push({
      name: comp.name,
      type: 'Complexity',
      summary: `Threshold: ${comp.threshold}, ${hardCount} hard / ${easyCount} easy candidates`,
      rawData: comp,
    })
  })

  effectiveSignals?.modality?.forEach((modality) => {
    allSignals.push({
      name: modality.name,
      type: 'Modality',
      summary: modality.description || 'Modality signal',
      rawData: modality,
    })
  })

  effectiveSignals?.role_bindings?.forEach((binding) => {
    const subjectCount = binding.subjects?.length || 0
    allSignals.push({
      name: binding.name,
      type: 'Authz',
      summary: `${binding.role} • ${subjectCount} ${subjectCount === 1 ? 'subject' : 'subjects'}`,
      rawData: binding,
    })
  })

  effectiveSignals?.jailbreak?.forEach((jb) => {
    const method = jb.method || 'classifier'
    allSignals.push({
      name: jb.name,
      type: 'Jailbreak',
      summary: `Method: ${method}, Threshold: ${jb.threshold}${jb.include_history ? ', includes history' : ''}`,
      rawData: jb,
    })
  })

  effectiveSignals?.pii?.forEach((p) => {
    const allowed = p.pii_types_allowed?.length || 0
    allSignals.push({
      name: p.name,
      type: 'PII',
      summary: `Threshold: ${p.threshold}${allowed > 0 ? `, ${allowed} types allowed` : ', deny all'}`,
      rawData: p,
    })
  })

  effectiveSignals?.kb?.forEach((kbSignal) => {
    const match = kbSignal.match || 'best'
    allSignals.push({
      name: kbSignal.name,
      type: 'KB',
      summary: `${kbSignal.kb} • ${kbSignal.target.kind}:${kbSignal.target.value} • ${match}`,
      rawData: kbSignal,
    })
  })

  const filteredSignals = allSignals.filter(
    (signal) =>
      signal.name.toLowerCase().includes(signalsSearch.toLowerCase()) ||
      signal.type.toLowerCase().includes(signalsSearch.toLowerCase()) ||
      signal.summary.toLowerCase().includes(signalsSearch.toLowerCase()),
  )

  const signalKey = (signal: UnifiedSignal) => `${signal.type}-${signal.name}`
  const signalReferenceCount = (signal: UnifiedSignal) =>
    getSignalReferenceCount(config, signal.type, signal.name)

  const signalsColumns: Column<UnifiedSignal>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>,
    },
    {
      key: 'type',
      header: 'Type',
      width: '160px',
      sortable: true,
      render: (row) => <span className={styles.tableMetaBadge}>{row.type}</span>,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (row) => (
        <span
          style={{
            fontSize: '0.875rem',
            color: 'var(--color-text-secondary)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            display: 'block',
          }}
        >
          {row.summary}
        </span>
      ),
    },
  ]

  const handleViewSignal = (signal: UnifiedSignal) => {
    const sections: ViewSection[] = []

    sections.push({
      title: 'Basic Information',
      fields: [
        { label: 'Name', value: signal.name },
        { label: 'Type', value: signal.type },
        { label: 'Summary', value: signal.summary, fullWidth: true },
      ],
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
                  <span
                    key={i}
                    style={{
                      padding: '0.25rem 0.75rem',
                      background: 'rgba(143, 148, 156, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    {kw}
                  </span>
                ))}
              </div>
            ),
            fullWidth: true,
          },
        ],
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
                  <div
                    key={i}
                    style={{
                      padding: '0.5rem',
                      background: 'rgba(166, 171, 179, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                    }}
                  >
                    {c}
                  </div>
                ))}
              </div>
            ),
            fullWidth: true,
          },
        ],
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
                  <span
                    key={i}
                    style={{
                      padding: '0.25rem 0.75rem',
                      background: 'rgba(147, 51, 234, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                    }}
                  >
                    {cat}
                  </span>
                ))}
              </div>
            ) : (
              'No categories'
            ),
            fullWidth: true,
          },
        ],
      })
    } else if (signal.type === 'Preference') {
      sections.push({
        title: 'Preference Configuration',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          {
            label: 'Threshold',
            value:
              signal.rawData.threshold !== undefined
                ? formatThreshold(signal.rawData.threshold)
                : 'Not set',
          },
          {
            label: 'Examples',
            value: signal.rawData.examples?.length ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9rem',
                }}
              >
                {signal.rawData.examples.map((ex: string, i: number) => (
                  <div
                    key={i}
                    style={{
                      padding: '0.35rem 0.5rem',
                      background: 'rgba(234, 179, 8, 0.1)',
                      borderRadius: 6,
                    }}
                  >
                    {ex}
                  </div>
                ))}
              </div>
            ) : (
              'No examples provided'
            ),
            fullWidth: true,
          },
        ],
      })
    } else if (signal.type === 'Reask') {
      sections.push({
        title: 'Reask Signal',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          {
            label: 'Threshold',
            value:
              signal.rawData.threshold !== undefined
                ? formatThreshold(signal.rawData.threshold)
                : 'Default (80%)',
          },
          { label: 'Lookback Turns', value: signal.rawData.lookback_turns?.toString() || '1' },
        ],
      })
    } else if (signal.type === 'Language') {
      sections.push({
        title: 'Language Signal',
        fields: [
          { label: 'Language Code', value: signal.rawData.name || 'N/A', fullWidth: true },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'Context') {
      sections.push({
        title: 'Context Signal',
        fields: [
          { label: 'Min Tokens', value: signal.rawData.min_tokens || 'N/A', fullWidth: true },
          { label: 'Max Tokens', value: signal.rawData.max_tokens || 'N/A', fullWidth: true },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'Structure') {
      sections.push({
        title: 'Structure Signal',
        fields: [
          { label: 'Feature Type', value: signal.rawData.feature?.type || 'N/A' },
          { label: 'Source Type', value: signal.rawData.feature?.source?.type || 'N/A' },
          {
            label: 'Feature',
            value: (
              <SignalStructureFeatureEditor
                value={signal.rawData.feature}
                onChange={() => undefined}
                readOnly
              />
            ),
            fullWidth: true,
          },
          {
            label: 'Predicate',
            value: signal.rawData.predicate ? (
              <SignalStructurePredicateEditor
                value={signal.rawData.predicate}
                onChange={() => undefined}
                readOnly
              />
            ) : (
              'None'
            ),
            fullWidth: true,
          },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'Complexity') {
      const fields: Array<{ label: string; value: React.ReactNode; fullWidth?: boolean }> = [
        {
          label: 'Threshold',
          value: signal.rawData.threshold?.toString() || 'N/A',
          fullWidth: true,
        },
      ]

      if (signal.rawData.composer) {
        fields.push({
          label: 'Composer',
          value: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div>
                <strong>Operator:</strong> {signal.rawData.composer.operator}
              </div>
              <SignalConditionsEditor
                value={signal.rawData.composer.conditions}
                onChange={() => undefined}
                readOnly
              />
            </div>
          ),
          fullWidth: true,
        })
      }

      fields.push(
        {
          label: 'Hard Candidates',
          value: (
            <SignalStringListEditor
              value={signal.rawData.hard?.candidates}
              onChange={() => undefined}
              addLabel=""
              emptyLabel="No hard candidates."
              itemLabel="Hard candidate"
              readOnly
            />
          ),
          fullWidth: true,
        },
        {
          label: 'Easy Candidates',
          value: (
            <SignalStringListEditor
              value={signal.rawData.easy?.candidates}
              onChange={() => undefined}
              addLabel=""
              emptyLabel="No easy candidates."
              itemLabel="Easy candidate"
              readOnly
            />
          ),
          fullWidth: true,
        },
        { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
      )

      sections.push({
        title: 'Complexity Signal',
        fields,
      })
    } else if (signal.type === 'Modality') {
      sections.push({
        title: 'Modality Signal',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'Authz') {
      sections.push({
        title: 'Role Binding',
        fields: [
          { label: 'Role', value: signal.rawData.role || 'N/A' },
          {
            label: 'Subjects',
            value: (
              <SignalSubjectsEditor
                value={signal.rawData.subjects}
                onChange={() => undefined}
                readOnly
              />
            ),
            fullWidth: true,
          },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'Jailbreak') {
      const fields: Array<{ label: string; value: React.ReactNode; fullWidth?: boolean }> = [
        { label: 'Method', value: signal.rawData.method || 'classifier', fullWidth: true },
        {
          label: 'Threshold',
          value: signal.rawData.threshold?.toString() || 'N/A',
          fullWidth: true,
        },
        {
          label: 'Include History',
          value: signal.rawData.include_history ? 'Yes' : 'No',
          fullWidth: true,
        },
      ]
      if (signal.rawData.method === 'contrastive') {
        fields.push(
          {
            label: 'Jailbreak Patterns',
            value: (
              <SignalStringListEditor
                value={signal.rawData.jailbreak_patterns}
                onChange={() => undefined}
                addLabel=""
                emptyLabel="No jailbreak patterns."
                itemLabel="Jailbreak pattern"
                readOnly
              />
            ),
            fullWidth: true,
          },
          {
            label: 'Benign Patterns',
            value: (
              <SignalStringListEditor
                value={signal.rawData.benign_patterns}
                onChange={() => undefined}
                addLabel=""
                emptyLabel="No benign patterns."
                itemLabel="Benign pattern"
                readOnly
              />
            ),
            fullWidth: true,
          },
        )
      }
      fields.push({
        label: 'Description',
        value: signal.rawData.description || 'N/A',
        fullWidth: true,
      })
      sections.push({ title: 'Jailbreak Signal', fields })
    } else if (signal.type === 'PII') {
      sections.push({
        title: 'PII Signal',
        fields: [
          {
            label: 'Threshold',
            value: signal.rawData.threshold?.toString() || 'N/A',
            fullWidth: true,
          },
          {
            label: 'Allowed PII Types',
            value: (
              <SignalStringListEditor
                value={signal.rawData.pii_types_allowed}
                onChange={() => undefined}
                addLabel=""
                emptyLabel="None (deny all)"
                itemLabel="PII type"
                readOnly
              />
            ),
            fullWidth: true,
          },
          {
            label: 'Include History',
            value: signal.rawData.include_history ? 'Yes' : 'No',
            fullWidth: true,
          },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
      })
    } else if (signal.type === 'KB') {
      sections.push({
        title: 'Knowledge Base Signal',
        fields: [
          { label: 'Knowledge Base', value: signal.rawData.kb || 'N/A' },
          { label: 'Target Kind', value: signal.rawData.target?.kind || 'N/A' },
          { label: 'Target Value', value: signal.rawData.target?.value || 'N/A', fullWidth: true },
          { label: 'Match', value: signal.rawData.match || 'best' },
        ],
      })
    } else {
      sections.push({
        title: 'Details',
        fields: [
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
        ],
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
      keywords: [],
      case_sensitive: false,
      threshold: 0.8,
      candidates: [],
      aggregation_method: 'mean',
      mmlu_categories: [],
      preference_examples: [],
      preference_threshold: undefined,
      min_tokens: '0',
      max_tokens: '8K',
      structure_feature: structuredClone(DEFAULT_STRUCTURE_FEATURE),
      structure_predicate: { ...DEFAULT_STRUCTURE_PREDICATE },
      complexity_threshold: 0.1,
      role: '',
      subjects: [],
      hard_candidates: [],
      easy_candidates: [],
      composer_operator: 'AND',
      composer_conditions: [],
      jailbreak_threshold: 0.65,
      jailbreak_method: 'classifier',
      include_history: false,
      jailbreak_patterns: [],
      benign_patterns: [],
      pii_threshold: 0.5,
      pii_types_allowed: [],
      pii_include_history: false,
      kb_name: '',
      target_kind: 'group',
      target_value: '',
      kb_match: 'best',
    }

    const initialData: AddSignalFormState =
      mode === 'edit' && signal
        ? {
            type: signal.type,
            name: signal.name,
            description: signal.rawData.description || '',
            operator: signal.rawData.operator || 'AND',
            keywords: [...(signal.rawData.keywords || [])],
            case_sensitive: !!signal.rawData.case_sensitive,
            threshold: signal.rawData.threshold ?? 0.8,
            candidates: [...(signal.rawData.candidates || [])],
            aggregation_method: signal.rawData.aggregation_method || 'mean',
            mmlu_categories: [...(signal.rawData.mmlu_categories || [])],
            preference_examples: [...(signal.rawData.examples || [])],
            preference_threshold: signal.rawData.threshold,
            lookback_turns: signal.rawData.lookback_turns,
            min_tokens: signal.rawData.min_tokens || '0',
            max_tokens: signal.rawData.max_tokens || '8K',
            structure_feature:
              signal.type === 'Structure' ? signal.rawData.feature : defaultForm.structure_feature,
            structure_predicate:
              signal.type === 'Structure'
                ? signal.rawData.predicate
                : defaultForm.structure_predicate,
            complexity_threshold: signal.rawData.threshold ?? 0.1,
            role: signal.type === 'Authz' ? signal.rawData.role || '' : '',
            subjects: signal.type === 'Authz' ? [...(signal.rawData.subjects || [])] : [],
            hard_candidates: [...(signal.rawData.hard?.candidates || [])],
            easy_candidates: [...(signal.rawData.easy?.candidates || [])],
            composer_operator: signal.rawData.composer?.operator || 'AND',
            composer_conditions: [...(signal.rawData.composer?.conditions || [])],
            jailbreak_threshold: signal.rawData.threshold ?? 0.65,
            jailbreak_method: signal.rawData.method || 'classifier',
            include_history: !!signal.rawData.include_history,
            jailbreak_patterns: [...(signal.rawData.jailbreak_patterns || [])],
            benign_patterns: [...(signal.rawData.benign_patterns || [])],
            pii_threshold: signal.rawData.threshold ?? 0.5,
            pii_types_allowed: [...(signal.rawData.pii_types_allowed || [])],
            pii_include_history: !!signal.rawData.include_history,
            kb_name: signal.type === 'KB' ? signal.rawData.kb || '' : '',
            target_kind: signal.type === 'KB' ? signal.rawData.target?.kind || 'group' : 'group',
            target_value: signal.type === 'KB' ? signal.rawData.target?.value || '' : '',
            kb_match: signal.type === 'KB' ? signal.rawData.match || 'best' : 'best',
          }
        : defaultForm

    const fields = buildSignalFormFields()

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
          const keywords = normalizeStringList(formData.keywords, 'Keywords', true)
          newConfig.signals.keywords = [
            ...(newConfig.signals.keywords || []),
            {
              name,
              operator: formData.operator,
              keywords,
              case_sensitive: !!formData.case_sensitive,
            },
          ]
          break
        }
        case 'Embeddings': {
          const candidates = normalizeStringList(formData.candidates, 'Candidates', true)
          const threshold = Number.isFinite(formData.threshold)
            ? Math.max(0, Math.min(1, formData.threshold))
            : 0
          newConfig.signals.embeddings = [
            ...(newConfig.signals.embeddings || []),
            {
              name,
              threshold,
              candidates,
              aggregation_method: formData.aggregation_method || 'mean',
            },
          ]
          break
        }
        case 'Domain': {
          const mmlu_categories = normalizeStringList(formData.mmlu_categories, 'MMLU categories')
          newConfig.signals.domains = [
            ...(newConfig.signals.domains || []),
            {
              name,
              description: formData.description,
              mmlu_categories,
            },
          ]
          break
        }
        case 'Preference': {
          const examples = normalizeStringList(formData.preference_examples, 'Preference examples')
          const hasThreshold = Number.isFinite(formData.preference_threshold)
          const threshold = hasThreshold
            ? Math.max(0, Math.min(1, Number(formData.preference_threshold)))
            : undefined

          const preferenceRule: {
            name: string
            description: string
            examples?: string[]
            threshold?: number
          } = {
            name,
            description: formData.description || '',
          }

          if (examples.length > 0) {
            preferenceRule.examples = examples
          }

          if (threshold !== undefined && threshold > 0) {
            preferenceRule.threshold = threshold
          }

          newConfig.signals.preferences = [...(newConfig.signals.preferences || []), preferenceRule]
          break
        }
        case 'Fact Check': {
          newConfig.signals.fact_check = [
            ...(newConfig.signals.fact_check || []),
            {
              name,
              description: formData.description,
            },
          ]
          break
        }
        case 'User Feedback': {
          newConfig.signals.user_feedbacks = [
            ...(newConfig.signals.user_feedbacks || []),
            {
              name,
              description: formData.description,
            },
          ]
          break
        }
        case 'Reask': {
          const threshold = Number.isFinite(formData.threshold)
            ? Math.max(0, Math.min(1, Number(formData.threshold)))
            : undefined
          const lookback_turns = Number.isFinite(formData.lookback_turns)
            ? Math.max(1, Math.trunc(Number(formData.lookback_turns)))
            : undefined

          newConfig.signals.reasks = [
            ...(newConfig.signals.reasks || []),
            {
              name,
              description: formData.description || undefined,
              threshold,
              lookback_turns,
            },
          ]
          break
        }
        case 'Language': {
          newConfig.signals.language = [
            ...(newConfig.signals.language || []),
            {
              name,
            },
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
              description: formData.description || undefined,
            },
          ]
          break
        }
        case 'Structure': {
          const feature = normalizeStructureFeature(formData.structure_feature)
          const predicate = normalizeStructurePredicate(feature, formData.structure_predicate)

          newConfig.signals.structure = [
            ...(newConfig.signals.structure || []),
            {
              name,
              description: formData.description || undefined,
              feature,
              ...(predicate ? { predicate } : {}),
            },
          ]
          break
        }
        case 'Complexity': {
          const complexity_threshold = formData.complexity_threshold ?? 0.1
          const hardList = normalizeStringList(formData.hard_candidates, 'Hard candidates', true)
          const easyList = normalizeStringList(formData.easy_candidates, 'Easy candidates', true)
          const conditions = normalizeConditions(formData.composer_conditions)
          const composer =
            conditions.length > 0
              ? {
                  operator: formData.composer_operator || 'AND',
                  conditions,
                }
              : undefined

          newConfig.signals.complexity = [
            ...(newConfig.signals.complexity || []),
            {
              name,
              threshold: complexity_threshold,
              hard: {
                candidates: hardList,
              },
              easy: {
                candidates: easyList,
              },
              description: formData.description || undefined,
              ...(composer && { composer }),
            },
          ]
          break
        }
        case 'Modality': {
          newConfig.signals.modality = [
            ...(newConfig.signals.modality || []),
            {
              name,
              description: formData.description || undefined,
            },
          ]
          break
        }
        case 'Authz': {
          const role = (formData.role || '').trim()
          if (!role) {
            throw new Error('Role is required for authz signals.')
          }
          const subjects = normalizeSubjects(formData.subjects)
          newConfig.signals.role_bindings = [
            ...(newConfig.signals.role_bindings || []),
            {
              name,
              role,
              subjects,
              description: formData.description || undefined,
            },
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
            description: formData.description || undefined,
          }
          if (method !== 'classifier') {
            jailbreakEntry.method = method
          }
          if (method === 'contrastive') {
            const jailbreakPatterns = normalizeStringList(
              formData.jailbreak_patterns,
              'Jailbreak patterns',
            )
            const benignPatterns = normalizeStringList(formData.benign_patterns, 'Benign patterns')
            if (jailbreakPatterns.length > 0) jailbreakEntry.jailbreak_patterns = jailbreakPatterns
            if (benignPatterns.length > 0) jailbreakEntry.benign_patterns = benignPatterns
          }
          newConfig.signals.jailbreak = [...(newConfig.signals.jailbreak || []), jailbreakEntry]
          break
        }
        case 'PII': {
          const pii_threshold = formData.pii_threshold ?? 0.5
          if (pii_threshold < 0 || pii_threshold > 1) {
            throw new Error('PII threshold must be between 0.0 and 1.0.')
          }
          const normalizedAllowedTypes = normalizeStringList(
            formData.pii_types_allowed,
            'Allowed PII types',
          )
          const allowedList = normalizedAllowedTypes.length > 0 ? normalizedAllowedTypes : undefined
          newConfig.signals.pii = [
            ...(newConfig.signals.pii || []),
            {
              name,
              threshold: pii_threshold,
              pii_types_allowed: allowedList,
              include_history: formData.pii_include_history || false,
              description: formData.description || undefined,
            },
          ]
          break
        }
        case 'KB': {
          const kb = (formData.kb_name || '').trim()
          const targetKind = formData.target_kind || 'group'
          const targetValue = (formData.target_value || '').trim()
          const match = formData.kb_match || 'best'
          if (!kb) {
            throw new Error('Knowledge base is required for KB signals.')
          }
          if (!targetValue) {
            throw new Error('Target value is required for KB signals.')
          }
          newConfig.signals.kb = [
            ...(newConfig.signals.kb || []),
            {
              name,
              kb,
              target: {
                kind: targetKind,
                value: targetValue,
              },
              match,
            },
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
      mode,
    )
  }

  const handleEditSignal = (signal: UnifiedSignal) => {
    openSignalEditor('edit', signal)
  }

  const handleDeleteSignal = (signal: UnifiedSignal) => {
    const referenceCount = signalReferenceCount(signal)
    if (referenceCount > 0) {
      setActionError(
        `Signal "${signal.name}" has ${referenceCount} active reference${referenceCount === 1 ? '' : 's'}. Update those decisions, projections, or composers before deleting it.`,
      )
      return
    }
    if (!config || !isPythonCLI) {
      setActionError('Deleting signals is only supported for Python CLI configs.')
      return
    }
    setActionError(null)
    setDeleteError(null)
    setSignalsPendingDelete([signal])
  }

  const handleBulkDeleteSignals = () => {
    if (!config || !isPythonCLI || selectedSignalKeys.size === 0) return
    const selectedSignals = allSignals.filter((signal) => selectedSignalKeys.has(signalKey(signal)))
    const safeSignals = selectedSignals.filter((signal) => signalReferenceCount(signal) === 0)
    if (safeSignals.length !== selectedSignals.length) {
      setActionError(
        'One or more selected signals are now referenced. Refresh the selection and try again.',
      )
      return
    }
    setActionError(null)
    setDeleteError(null)
    setSignalsPendingDelete(safeSignals)
  }

  const confirmDeleteSignals = async () => {
    if (!config || !isPythonCLI || signalsPendingDelete.length === 0 || deletePending) return

    setDeletePending(true)
    setDeleteError(null)
    const newConfig: ConfigData = cloneConfigData(config)
    signalsPendingDelete.forEach((signal) =>
      removeSignalByName(newConfig, signal.type, signal.name),
    )
    try {
      await saveConfig(newConfig)
      setSelectedSignalKeys(new Set())
      setSignalsPendingDelete([])
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete signals')
    } finally {
      setDeletePending(false)
    }
  }

  return (
    <ConfigPageManagerLayout
      title="Signals"
      description="Review the signal catalog that drives semantic routing, guardrails, and context-aware behavior."
    >
      <div className={styles.sectionPanel}>
        {actionError ? (
          <div className={styles.error} role="alert">
            {actionError}
          </div>
        ) : null}
        <div className={styles.sectionTableBlock}>
          <TableHeader
            title="Signals"
            count={filteredSignals.length}
            searchPlaceholder="Search signals..."
            searchValue={signalsSearch}
            onSearchChange={onSignalsSearchChange}
            onAdd={() => openSignalEditor('add')}
            addButtonText="Add Signal"
            disabled={isReadonly || !isPythonCLI}
            variant="embedded"
          />
          {selectedSignalKeys.size > 0 ? (
            <div className={signalStyles.bulkBar} role="status">
              <div className={signalStyles.bulkCopy}>
                <strong>
                  {selectedSignalKeys.size} signal{selectedSignalKeys.size === 1 ? '' : 's'}{' '}
                  selected
                </strong>
                <span className={signalStyles.bulkHint}>
                  Referenced signals cannot be selected or deleted.
                </span>
              </div>
              <div className={signalStyles.bulkActions}>
                <button
                  type="button"
                  className={signalStyles.clearButton}
                  onClick={() => setSelectedSignalKeys(new Set())}
                >
                  Clear
                </button>
                <button
                  type="button"
                  className={signalStyles.deleteButton}
                  onClick={handleBulkDeleteSignals}
                >
                  Delete selected
                </button>
              </div>
            </div>
          ) : null}
          <DataTable
            columns={signalsColumns}
            data={filteredSignals}
            keyExtractor={signalKey}
            onView={handleViewSignal}
            onEdit={handleEditSignal}
            onDelete={handleDeleteSignal}
            emptyMessage={signalsSearch ? 'No signals match your search' : 'No signals configured'}
            className={styles.managerTable}
            readonly={isReadonly || !isPythonCLI}
            pagination={{
              pageSize: 25,
              pageSizeOptions: [10, 25, 50],
              itemLabel: 'signals',
              resetKey: signalsSearch,
            }}
            selection={
              !isReadonly && isPythonCLI
                ? {
                    selectedKeys: selectedSignalKeys,
                    onChange: setSelectedSignalKeys,
                    isRowDisabled: (signal) => signalReferenceCount(signal) > 0,
                    label: 'signal',
                  }
                : undefined
            }
          />
        </div>
      </div>

      <ConfirmDialog
        isOpen={signalsPendingDelete.length > 0}
        title={signalsPendingDelete.length === 1 ? 'Delete signal' : 'Delete signals'}
        description={
          signalsPendingDelete.length === 1 ? (
            <>
              Delete <strong>{signalsPendingDelete[0]?.name}</strong> from the routing signal
              catalog?
            </>
          ) : (
            <>
              Delete <strong>{signalsPendingDelete.length}</strong> selected signals from the
              routing catalog?
            </>
          )
        }
        details={deleteError ? <span role="alert">{deleteError}</span> : undefined}
        confirmLabel={signalsPendingDelete.length === 1 ? 'Delete signal' : 'Delete signals'}
        pending={deletePending}
        onCancel={() => {
          if (!deletePending) {
            setSignalsPendingDelete([])
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDeleteSignals}
      />
    </ConfigPageManagerLayout>
  )
}
