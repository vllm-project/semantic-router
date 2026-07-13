import type { FieldConfig } from '../components/EditModal'
import { routerStructuredField } from './configPageRouterStructuredFields'
import type { RouterSystemKey } from './configPageRouterDefaultsSupport'

export function featureFieldsForKey(key: RouterSystemKey): FieldConfig[] | undefined {
  switch (key) {
    case 'prompt_guard':
      return [
        { name: 'enabled', label: 'Enable Prompt Guard', type: 'boolean' },
        { name: 'model_ref', label: 'Model Ref', type: 'text', placeholder: 'prompt_guard' },
        {
          name: 'model_id',
          label: 'Model ID Override',
          type: 'text',
          placeholder: 'models/mmbert32k-jailbreak-detector-merged',
        },
        { name: 'threshold', label: 'Threshold', type: 'percentage', placeholder: '70' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'use_mmbert_32k', label: 'Use mmBERT 32K', type: 'boolean' },
        { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean' },
        {
          name: 'jailbreak_mapping_path',
          label: 'Mapping Path',
          type: 'text',
          placeholder: 'models/.../jailbreak_type_mapping.json',
        },
      ]
    case 'classifier':
      return [
        routerStructuredField(key, 'domain'),
        routerStructuredField(key, 'pii'),
        routerStructuredField(key, 'mcp'),
        routerStructuredField(key, 'preference'),
      ]
    case 'hallucination_mitigation':
      return [
        { name: 'enabled', label: 'Enable Hallucination Mitigation', type: 'boolean' },
        {
          name: 'on_hallucination_detected',
          label: 'On Detection Action',
          type: 'text',
          placeholder: 'block',
        },
        routerStructuredField(key, 'fact_check'),
        routerStructuredField(key, 'detector'),
        routerStructuredField(key, 'explainer'),
      ]
    case 'feedback_detector':
      return [
        { name: 'enabled', label: 'Enable Feedback Detector', type: 'boolean' },
        { name: 'model_ref', label: 'Model Ref', type: 'text', placeholder: 'feedback_detector' },
        {
          name: 'model_id',
          label: 'Model ID Override',
          type: 'text',
          placeholder: 'models/mmbert32k-feedback-detector-merged',
        },
        { name: 'threshold', label: 'Threshold', type: 'percentage', placeholder: '70' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'use_mmbert_32k', label: 'Use mmBERT 32K', type: 'boolean' },
        { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean' },
      ]
    case 'external_models':
      return [routerStructuredField(key, 'items')]
    case 'system_models':
      return [
        {
          name: 'prompt_guard',
          label: 'Prompt Guard Binding',
          type: 'text',
          placeholder: 'models/mmbert32k-jailbreak-detector-merged',
        },
        {
          name: 'domain_classifier',
          label: 'Domain Classifier Binding',
          type: 'text',
          placeholder: 'models/mmbert32k-intent-classifier-merged',
        },
        {
          name: 'pii_classifier',
          label: 'PII Classifier Binding',
          type: 'text',
          placeholder: 'models/mmbert32k-pii-detector-merged',
        },
        {
          name: 'fact_check_classifier',
          label: 'Fact Check Binding',
          type: 'text',
          placeholder: 'models/mmbert32k-factcheck-classifier-merged',
        },
        {
          name: 'hallucination_detector',
          label: 'Hallucination Detector Binding',
          type: 'text',
          placeholder: 'models/mom-halugate-detector',
        },
        {
          name: 'hallucination_explainer',
          label: 'Hallucination Explainer Binding',
          type: 'text',
          placeholder: 'models/mom-halugate-explainer',
        },
        {
          name: 'feedback_detector',
          label: 'Feedback Detector Binding',
          type: 'text',
          placeholder: 'models/mmbert32k-feedback-detector-merged',
        },
      ]
    case 'embedding_models':
      return [
        {
          name: 'qwen3_model_path',
          label: 'Qwen3 Model Path',
          type: 'text',
          placeholder: 'models/mom-embedding-pro',
        },
        {
          name: 'gemma_model_path',
          label: 'Gemma Model Path',
          type: 'text',
          placeholder: 'models/mom-embedding-flash',
        },
        {
          name: 'mmbert_model_path',
          label: 'mmBERT Model Path',
          type: 'text',
          placeholder: 'models/mmbert-embed-32k-2d-matryoshka',
        },
        {
          name: 'multimodal_model_path',
          label: 'Multimodal Model Path',
          type: 'text',
          placeholder: 'models/mom-embedding-multimodal',
        },
        {
          name: 'bert_model_path',
          label: 'BERT Model Path',
          type: 'text',
          placeholder: 'models/mom-embedding-bert',
        },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        routerStructuredField(key, 'embedding_config'),
        routerStructuredField(key, 'endpoint'),
      ]
    case 'prompt_compression':
      return [
        { name: 'enabled', label: 'Enable Prompt Compression', type: 'boolean' },
        {
          name: 'profile',
          label: 'Profile',
          type: 'select',
          options: ['default', 'coding', 'medical', 'security', 'multi_turn'],
        },
        { name: 'max_tokens', label: 'Max Tokens', type: 'number', placeholder: '512' },
        { name: 'min_length', label: 'Min Length', type: 'number', placeholder: '64' },
        routerStructuredField(key, 'skip_signals'),
        {
          name: 'textrank_weight',
          label: 'TextRank Weight',
          type: 'number',
          step: 0.01,
          placeholder: '1.0',
        },
        {
          name: 'position_weight',
          label: 'Position Weight',
          type: 'number',
          step: 0.01,
          placeholder: '1.0',
        },
        {
          name: 'tfidf_weight',
          label: 'TFIDF Weight',
          type: 'number',
          step: 0.01,
          placeholder: '1.0',
        },
        {
          name: 'novelty_weight',
          label: 'Novelty Weight',
          type: 'number',
          step: 0.01,
          placeholder: '0.05',
        },
        {
          name: 'position_depth',
          label: 'Position Depth',
          type: 'number',
          step: 0.01,
          placeholder: '1.0',
        },
        {
          name: 'preserve_first_n',
          label: 'Preserve First Sentences',
          type: 'number',
          placeholder: '3',
        },
        {
          name: 'preserve_last_n',
          label: 'Preserve Last Sentences',
          type: 'number',
          placeholder: '2',
        },
      ]
    case 'modality_detector':
      return [
        { name: 'enabled', label: 'Enable Modality Detector', type: 'boolean' },
        routerStructuredField(key, 'prompt_prefixes'),
        {
          name: 'method',
          label: 'Detection Method',
          type: 'select',
          options: ['classifier', 'keyword', 'hybrid'],
        },
        routerStructuredField(key, 'classifier'),
        routerStructuredField(key, 'keywords'),
        routerStructuredField(key, 'both_keywords'),
        {
          name: 'confidence_threshold',
          label: 'Confidence Threshold',
          type: 'percentage',
          placeholder: '80',
        },
        {
          name: 'lower_threshold_ratio',
          label: 'Lower Threshold Ratio',
          type: 'percentage',
          placeholder: '60',
        },
      ]
    case 'observability':
      return [routerStructuredField(key, 'metrics'), routerStructuredField(key, 'tracing')]
    case 'looper':
      return [
        { name: 'enabled', label: 'Enable Looper', type: 'boolean' },
        {
          name: 'endpoint',
          label: 'Endpoint',
          type: 'text',
          placeholder: 'http://localhost:8899/v1/chat/completions',
        },
        {
          name: 'timeout_seconds',
          label: 'Timeout (seconds)',
          type: 'number',
          placeholder: '1200',
        },
        routerStructuredField(key, 'headers'),
      ]
    case 'clear_route_cache':
      return [
        { name: 'value', label: 'Clear Route Cache For Auxiliary Mutations', type: 'boolean' },
      ]
    case 'model_selection':
      return [
        { name: 'enabled', label: 'Enable Model Selection', type: 'boolean' },
        {
          name: 'default_algorithm',
          label: 'Method',
          type: 'select',
          options: ['knn', 'kmeans', 'svm', 'router_dc', 'automix', 'hybrid'],
          required: true,
        },
        {
          name: 'models_path',
          label: 'ML Models Path',
          type: 'text',
          placeholder: 'models/model_selection',
        },
        routerStructuredField(key, 'knn'),
        routerStructuredField(key, 'kmeans'),
        routerStructuredField(key, 'svm'),
        routerStructuredField(key, 'router_dc'),
        routerStructuredField(key, 'automix'),
        routerStructuredField(key, 'hybrid'),
      ]
    case 'api':
      return [routerStructuredField(key, 'batch_classification')]
  }

  return undefined
}
