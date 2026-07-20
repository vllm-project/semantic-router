import type { FieldConfig } from '../components/EditModal'
import ConfigPageDomainCategoryPicker from './ConfigPageDomainCategoryPicker'
import {
  SignalConditionsEditor,
  SignalStringListEditor,
  SignalStructureFeatureEditor,
  SignalStructurePredicateEditor,
  SignalSubjectsEditor,
} from './configPageSignalStructuredEditors'
import { readStringList } from './configPageSignalFormSupport'
import type { AddSignalFormState, SignalType } from './configPageSupport'

const signalTypes: SignalType[] = [
  'Keywords',
  'Embeddings',
  'Domain',
  'Preference',
  'Fact Check',
  'User Feedback',
  'Reask',
  'Language',
  'Context',
  'Structure',
  'Complexity',
  'Modality',
  'Authz',
  'Jailbreak',
  'PII',
  'KB',
]

const hideUnless = (type: SignalType) => (formData: AddSignalFormState) => formData.type !== type

interface ListFieldOptions {
  name: string
  label: string
  signalType: SignalType
  addLabel: string
  emptyLabel: string
  itemLabel: string
  placeholder?: string
  description?: string
  shouldHide?: (formData: AddSignalFormState) => boolean
}

function stringListField({
  name,
  label,
  signalType,
  addLabel,
  emptyLabel,
  itemLabel,
  placeholder,
  description,
  shouldHide,
}: ListFieldOptions): FieldConfig<AddSignalFormState> {
  return {
    name,
    label,
    type: 'custom',
    customRender: (value, onChange) => (
      <SignalStringListEditor
        value={value}
        onChange={onChange}
        addLabel={addLabel}
        emptyLabel={emptyLabel}
        itemLabel={itemLabel}
        placeholder={placeholder}
      />
    ),
    description,
    shouldHide: shouldHide ?? hideUnless(signalType),
  }
}

export function buildSignalFormFields(): FieldConfig<AddSignalFormState>[] {
  return [
    {
      name: 'type',
      label: 'Type',
      type: 'select',
      options: signalTypes,
      required: true,
      description: 'Fields are validated based on the selected type.',
    },
    {
      name: 'name',
      label: 'Name',
      type: 'text',
      required: true,
      placeholder: 'Unique signal name',
    },
    {
      name: 'description',
      label: 'Description',
      type: 'textarea',
      placeholder: 'Optional signal description',
    },
    stringListField({
      name: 'preference_examples',
      label: 'Examples (preference only)',
      signalType: 'Preference',
      addLabel: 'Add example',
      emptyLabel: 'No preference examples configured.',
      itemLabel: 'Example',
      placeholder: 'A representative preference prompt',
      description: 'Few-shot hints sent to the contrastive preference classifier.',
    }),
    {
      name: 'preference_threshold',
      label: 'Threshold (preference only)',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.35',
      description: 'Override the global preference threshold for this rule.',
      shouldHide: hideUnless('Preference'),
    },
    {
      name: 'threshold',
      label: 'Threshold (reask only)',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.80',
      shouldHide: hideUnless('Reask'),
    },
    {
      name: 'lookback_turns',
      label: 'Lookback Turns (reask only)',
      type: 'number',
      min: 1,
      step: 1,
      placeholder: '1',
      shouldHide: hideUnless('Reask'),
    },
    {
      name: 'operator',
      label: 'Operator (keywords only)',
      type: 'select',
      options: ['AND', 'OR'],
      shouldHide: hideUnless('Keywords'),
    },
    {
      name: 'case_sensitive',
      label: 'Case Sensitive (keywords only)',
      type: 'boolean',
      shouldHide: hideUnless('Keywords'),
    },
    stringListField({
      name: 'keywords',
      label: 'Keywords',
      signalType: 'Keywords',
      addLabel: 'Add keyword',
      emptyLabel: 'At least one keyword is required.',
      itemLabel: 'Keyword',
      placeholder: 'urgent',
    }),
    {
      name: 'threshold',
      label: 'Threshold (embeddings only)',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.80',
      shouldHide: hideUnless('Embeddings'),
    },
    {
      name: 'aggregation_method',
      label: 'Aggregation Method (embeddings only)',
      type: 'select',
      options: ['mean', 'max', 'any'],
      shouldHide: hideUnless('Embeddings'),
    },
    stringListField({
      name: 'candidates',
      label: 'Candidates (embeddings only)',
      signalType: 'Embeddings',
      addLabel: 'Add candidate',
      emptyLabel: 'At least one embedding candidate is required.',
      itemLabel: 'Candidate',
      placeholder: 'A representative request',
    }),
    {
      name: 'mmlu_categories',
      label: 'MMLU Categories (domains only)',
      type: 'custom',
      customRender: (value, onChange) => (
        <ConfigPageDomainCategoryPicker value={readStringList(value)} onChange={onChange} />
      ),
      shouldHide: hideUnless('Domain'),
    },
    {
      name: 'min_tokens',
      label: 'Minimum Tokens (context only)',
      type: 'text',
      placeholder: '0, 8K, or 1M',
      shouldHide: hideUnless('Context'),
    },
    {
      name: 'max_tokens',
      label: 'Maximum Tokens (context only)',
      type: 'text',
      placeholder: '8K or 1024K',
      shouldHide: hideUnless('Context'),
    },
    {
      name: 'structure_feature',
      label: 'Feature (structure only)',
      type: 'custom',
      customRender: (value, onChange) => (
        <SignalStructureFeatureEditor value={value} onChange={onChange} />
      ),
      description: 'Choose how the request shape is measured and which typed source it scans.',
      shouldHide: hideUnless('Structure'),
    },
    {
      name: 'structure_predicate',
      label: 'Predicate (structure only)',
      type: 'custom',
      customRender: (value, onChange) => (
        <SignalStructurePredicateEditor value={value} onChange={onChange} />
      ),
      description: 'Set numeric bounds. Exists features ignore predicate bounds.',
      shouldHide: hideUnless('Structure'),
    },
    {
      name: 'complexity_threshold',
      label: 'Threshold (complexity only)',
      type: 'number',
      placeholder: '0.1',
      shouldHide: hideUnless('Complexity'),
    },
    {
      name: 'composer_operator',
      label: 'Composer Operator (complexity only)',
      type: 'select',
      options: ['AND', 'OR'],
      shouldHide: hideUnless('Complexity'),
    },
    {
      name: 'composer_conditions',
      label: 'Composer Conditions (complexity only)',
      type: 'custom',
      customRender: (value, onChange) => (
        <SignalConditionsEditor value={value} onChange={onChange} />
      ),
      description: 'Optionally gate this complexity signal on typed signal references.',
      shouldHide: hideUnless('Complexity'),
    },
    stringListField({
      name: 'hard_candidates',
      label: 'Hard Candidates (complexity only)',
      signalType: 'Complexity',
      addLabel: 'Add hard candidate',
      emptyLabel: 'At least one hard candidate is required.',
      itemLabel: 'Hard candidate',
      placeholder: 'Design a distributed system',
    }),
    stringListField({
      name: 'easy_candidates',
      label: 'Easy Candidates (complexity only)',
      signalType: 'Complexity',
      addLabel: 'Add easy candidate',
      emptyLabel: 'At least one easy candidate is required.',
      itemLabel: 'Easy candidate',
      placeholder: 'Print hello world',
    }),
    {
      name: 'role',
      label: 'Role (authz only)',
      type: 'text',
      placeholder: 'admin',
      shouldHide: hideUnless('Authz'),
    },
    {
      name: 'subjects',
      label: 'Subjects (authz only)',
      type: 'custom',
      customRender: (value, onChange) => <SignalSubjectsEditor value={value} onChange={onChange} />,
      shouldHide: hideUnless('Authz'),
    },
    {
      name: 'jailbreak_method',
      label: 'Method (jailbreak only)',
      type: 'select',
      options: ['classifier', 'contrastive'],
      shouldHide: hideUnless('Jailbreak'),
    },
    {
      name: 'jailbreak_threshold',
      label: 'Threshold (jailbreak only)',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.65',
      shouldHide: hideUnless('Jailbreak'),
    },
    {
      name: 'include_history',
      label: 'Include History (jailbreak only)',
      type: 'boolean',
      shouldHide: hideUnless('Jailbreak'),
    },
    stringListField({
      name: 'jailbreak_patterns',
      label: 'Jailbreak Patterns (contrastive only)',
      signalType: 'Jailbreak',
      addLabel: 'Add jailbreak pattern',
      emptyLabel: 'No jailbreak patterns configured.',
      itemLabel: 'Jailbreak pattern',
      placeholder: 'Ignore all previous instructions',
      shouldHide: (formData) =>
        formData.type !== 'Jailbreak' || formData.jailbreak_method !== 'contrastive',
    }),
    stringListField({
      name: 'benign_patterns',
      label: 'Benign Patterns (contrastive only)',
      signalType: 'Jailbreak',
      addLabel: 'Add benign pattern',
      emptyLabel: 'No benign patterns configured.',
      itemLabel: 'Benign pattern',
      placeholder: 'Help me write an email',
      shouldHide: (formData) =>
        formData.type !== 'Jailbreak' || formData.jailbreak_method !== 'contrastive',
    }),
    {
      name: 'pii_threshold',
      label: 'Threshold (PII only)',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.5',
      shouldHide: hideUnless('PII'),
    },
    stringListField({
      name: 'pii_types_allowed',
      label: 'Allowed PII Types (PII only)',
      signalType: 'PII',
      addLabel: 'Allow PII type',
      emptyLabel: 'No allowed PII types; all detected types are denied.',
      itemLabel: 'PII type',
      placeholder: 'EMAIL_ADDRESS',
    }),
    {
      name: 'pii_include_history',
      label: 'Include History (PII only)',
      type: 'boolean',
      shouldHide: hideUnless('PII'),
    },
    {
      name: 'kb_name',
      label: 'Knowledge Base (KB only)',
      type: 'text',
      placeholder: 'privacy_kb',
      shouldHide: hideUnless('KB'),
    },
    {
      name: 'target_kind',
      label: 'Target Kind (KB only)',
      type: 'select',
      options: ['group', 'label'],
      shouldHide: hideUnless('KB'),
    },
    {
      name: 'target_value',
      label: 'Target Value (KB only)',
      type: 'text',
      placeholder: 'privacy_policy',
      shouldHide: hideUnless('KB'),
    },
    {
      name: 'kb_match',
      label: 'Match (KB only)',
      type: 'select',
      options: ['best', 'threshold'],
      shouldHide: hideUnless('KB'),
    },
  ]
}
