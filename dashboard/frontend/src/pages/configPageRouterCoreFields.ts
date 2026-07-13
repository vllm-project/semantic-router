import type { FieldConfig } from '../components/EditModal'
import { routerStructuredField } from './configPageRouterStructuredFields'
import type { RouterSystemKey } from './configPageRouterDefaultsSupport'

export function coreFieldsForKey(key: RouterSystemKey): FieldConfig[] | undefined {
  switch (key) {
    case 'router_core':
      return [
        {
          name: 'config_source',
          label: 'Config Source',
          type: 'select',
          options: ['file', 'kubernetes'],
          required: true,
        },
        {
          name: 'strategy',
          label: 'Routing Strategy',
          type: 'text',
          placeholder: 'static, router_dc, automix...',
        },
        {
          name: 'auto_model_name',
          label: 'Auto Model Name',
          type: 'text',
          placeholder: 'vllm-sr/auto',
        },
        routerStructuredField(key, 'auto_model_names'),
        {
          name: 'include_config_models_in_list',
          label: 'Include Config Models In List',
          type: 'boolean',
        },
        routerStructuredField(key, 'streamed_body'),
      ]
    case 'response_api':
      return [
        { name: 'enabled', label: 'Enable Response API', type: 'boolean' },
        {
          name: 'store_backend',
          label: 'Store Backend',
          type: 'select',
          options: ['memory', 'milvus', 'redis'],
          required: true,
        },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '86400' },
        { name: 'max_responses', label: 'Max Responses', type: 'number', placeholder: '1000' },
      ]
    case 'router_replay':
      return [
        { name: 'enabled', label: 'Enable Router Replay', type: 'boolean' },
        {
          name: 'store_backend',
          label: 'Store Backend',
          type: 'select',
          options: ['memory', 'redis', 'postgres', 'milvus'],
          required: true,
        },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '2592000' },
        { name: 'async_writes', label: 'Async Writes', type: 'boolean' },
      ]
    case 'authz':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        routerStructuredField(key, 'identity'),
        routerStructuredField(key, 'providers'),
      ]
    case 'ratelimit':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        routerStructuredField(key, 'providers'),
      ]
    case 'memory':
      return [
        { name: 'enabled', label: 'Enable Memory', type: 'boolean' },
        { name: 'auto_store', label: 'Auto Store Facts', type: 'boolean' },
        routerStructuredField(key, 'milvus'),
        { name: 'embedding_model', label: 'Embedding Model', type: 'text', placeholder: 'bert' },
        {
          name: 'default_retrieval_limit',
          label: 'Default Retrieval Limit',
          type: 'number',
          placeholder: '5',
        },
        {
          name: 'default_similarity_threshold',
          label: 'Similarity Threshold',
          type: 'percentage',
          placeholder: '70',
        },
        {
          name: 'extraction_batch_size',
          label: 'Extraction Batch Size',
          type: 'number',
          placeholder: '10',
        },
        { name: 'hybrid_search', label: 'Hybrid Search', type: 'boolean' },
        { name: 'hybrid_mode', label: 'Hybrid Mode', type: 'text', placeholder: 'rerank' },
        { name: 'adaptive_threshold', label: 'Adaptive Threshold', type: 'boolean' },
        routerStructuredField(key, 'quality_scoring'),
        routerStructuredField(key, 'reflection'),
      ]
    case 'semantic_cache':
      return [
        { name: 'enabled', label: 'Enable Semantic Cache', type: 'boolean' },
        {
          name: 'backend_type',
          label: 'Backend Type',
          type: 'select',
          options: ['memory', 'milvus', 'redis'],
          required: true,
        },
        {
          name: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'percentage',
          placeholder: '80',
        },
        { name: 'max_entries', label: 'Max Entries', type: 'number', placeholder: '1000' },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '3600' },
        {
          name: 'eviction_policy',
          label: 'Eviction Policy',
          type: 'select',
          options: ['fifo', 'lru', 'lfu'],
        },
        {
          name: 'embedding_model',
          label: 'Embedding Model Override',
          type: 'text',
          placeholder: 'mmbert',
        },
        routerStructuredField(key, 'redis'),
        routerStructuredField(key, 'milvus'),
      ]
    case 'vector_store':
      return [
        { name: 'enabled', label: 'Enable Vector Store', type: 'boolean' },
        {
          name: 'backend_type',
          label: 'Backend Type',
          type: 'select',
          options: ['memory', 'milvus', 'llama_stack'],
          required: true,
        },
        {
          name: 'file_storage_dir',
          label: 'File Storage Dir',
          type: 'text',
          placeholder: '/var/lib/vsr/data',
        },
        {
          name: 'max_file_size_mb',
          label: 'Max File Size (MB)',
          type: 'number',
          placeholder: '50',
        },
        {
          name: 'embedding_model',
          label: 'Embedding Model',
          type: 'select',
          options: ['bert', 'qwen3', 'gemma', 'mmbert', 'multimodal'],
        },
        {
          name: 'embedding_dimension',
          label: 'Embedding Dimension',
          type: 'number',
          placeholder: '384',
        },
        { name: 'ingestion_workers', label: 'Ingestion Workers', type: 'number', placeholder: '2' },
        routerStructuredField(key, 'supported_formats'),
        routerStructuredField(key, 'memory'),
        routerStructuredField(key, 'milvus'),
        routerStructuredField(key, 'llama_stack'),
      ]
    case 'tools':
      return [
        { name: 'enabled', label: 'Enable Tool Auto Selection', type: 'boolean' },
        { name: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
        {
          name: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'percentage',
          placeholder: '20',
        },
        {
          name: 'tools_db_path',
          label: 'Tools DB Path',
          type: 'text',
          placeholder: 'config/tools_db.json',
        },
        { name: 'fallback_to_empty', label: 'Fallback To Empty', type: 'boolean' },
        routerStructuredField(key, 'advanced_filtering'),
      ]
  }

  return undefined
}
