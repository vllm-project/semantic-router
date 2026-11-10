/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Overview',
      collapsed: false,
      items: [
        'overview/goals',
        'overview/semantic-router-overview',
        'overview/component-architecture',
        'overview/use-cases',
        'overview/signal-driven-decisions',
        'overview/mom-model-family',
      ],
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'installation/installation',
        'installation/agent-harness',
        'installation/agent',
        'installation/shell-completion',
        {
          type: 'category',
          label: 'Plan a Deployment',
          items: [
            'installation/deployment-options',
            'installation/deployment-architecture',
            'installation/gateway-modes',
            'installation/support-matrix',
          ],
        },
        {
          type: 'category',
          label: 'Compatibility',
          items: [
            'installation/protocol-compatibility',
            'installation/backend-target-compatibility',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'System One',
      collapsed: false,
      link: { type: 'doc', id: 'system-one/overview' },
      items: [
        'system-one/concepts',
        'system-one/api',
        'system-one/deployments',
        'system-one/router',
        'system-one/custom-questions',
        'system-one/performance',
        'system-one/reference',
      ],
    },
    {
      type: 'category',
      label: 'Capabilities',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Entrypoints',
          link: {
            type: 'doc',
            id: 'tutorials/global/models-entrypoints-serving',
          },
          items: [
            'tutorials/global/entrypoints-and-recipes',
            'tutorials/global/entrypoints',
            'tutorials/global/recipes',
          ],
        },
        {
          type: 'category',
          label: 'Signals',
          items: [
            'tutorials/signal/overview',
            'tutorials/signal/systemone',
            {
              type: 'category',
              label: 'Application',
              items: [
                'tutorials/signal/application/domain',
                'tutorials/signal/application/fact-check',
                'tutorials/signal/application/user-feedback',
                'tutorials/signal/application/jailbreak',
                'tutorials/signal/application/safety',
                'tutorials/signal/application/pii',
                'tutorials/signal/application/hallucination',
                'tutorials/signal/application/refusal',
                'tutorials/signal/application/turn-relation',
                'tutorials/signal/application/complexity',
                'tutorials/signal/application/modality',
              ],
            },
            {
              type: 'category',
              label: 'Heuristic',
              items: [
                'tutorials/signal/heuristic/authz',
                'tutorials/signal/heuristic/context',
                'tutorials/signal/heuristic/conversation',
                'tutorials/signal/heuristic/input-modality',
                'tutorials/signal/heuristic/keyword',
                'tutorials/signal/heuristic/language',
                'tutorials/signal/heuristic/metadata',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Projections',
          items: [
            'tutorials/projection/overview',
            'tutorials/projection/partitions',
            'tutorials/projection/scores',
            'tutorials/projection/mappings',
          ],
        },
        {
          type: 'category',
          label: 'Decisions',
          items: [
            'tutorials/decision/overview',
            'tutorials/decision/single',
            'tutorials/decision/and',
            'tutorials/decision/or',
            'tutorials/decision/not',
            'tutorials/decision/composite',
            'tutorials/decision/retention',
            'tutorials/decision/output',
          ],
        },
        {
          type: 'category',
          label: 'Algorithms',
          items: [
            'tutorials/algorithm/overview',
            {
              type: 'category',
              label: 'Selectors',
              items: [
                'tutorials/algorithm/selection/static',
                'tutorials/algorithm/selection/random',
                'tutorials/algorithm/selection/multi-factor',
                'tutorials/algorithm/selection/latency-aware',
                'tutorials/algorithm/selection/systemone',
              ],
            },
            {
              type: 'category',
              label: 'Executors',
              items: [
                'tutorials/algorithm/looper/escalate',
                'tutorials/algorithm/looper/fusion',
                'tutorials/algorithm/looper/ratings',
                'tutorials/algorithm/looper/remom',
                'tutorials/algorithm/looper/workflows',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Learning',
          items: [
            'tutorials/learning/overview',
            'tutorials/learning/adaptations',
            'tutorials/learning/protection',
            'tutorials/learning/memory-and-replay',
            'tutorials/learning/decision-adaptations',
            'tutorials/learning/experience',
          ],
        },
        {
          type: 'category',
          label: 'Plugins',
          items: [
            'tutorials/plugin/overview',
            {
              type: 'category',
              label: 'Response and Mutation',
              items: [
                'tutorials/plugin/fast-response',
                'tutorials/plugin/header-mutation',
                'tutorials/plugin/masking',
                'tutorials/plugin/request-params',
                'tutorials/plugin/system-prompt',
                'tutorials/plugin/tools',
              ],
            },
            {
              type: 'category',
              label: 'Retrieval and Memory',
              items: [
                'tutorials/plugin/context',
                'tutorials/plugin/router-replay',
                'tutorials/plugin/response-cache',
                'tutorials/plugin/prompt-cache',
              ],
            },
            {
              type: 'category',
              label: 'Safety and Generation',
              items: [
                'tutorials/plugin/content-safety',
                'tutorials/plugin/response-guard',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Shared Services',
          link: {
            type: 'doc',
            id: 'tutorials/global/overview',
          },
          items: [
            'tutorials/global/api-and-observability',
            'tutorials/global/stores-and-tools',
            'tutorials/global/model-runtime',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Model Runtime',
      collapsed: false,
      link: { type: 'doc', id: 'model-runtime/overview' },
      items: [
        'model-runtime/quickstart',
        'model-runtime/choose-a-model',
        'model-runtime/guides/embeddings',
        'model-runtime/guides/multimodal',
        'model-runtime/deploy',
        'model-runtime/engine-args',
        'installation/runtime/external',
        'model-runtime/migrate',
        'model-runtime/troubleshooting',
        'model-runtime/plugins',
        'model-runtime/reference',
      ],
    },
    {
      type: 'category',
      label: 'Deploy & Operate',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Configure',
          link: {
            type: 'doc',
            id: 'installation/configuration',
          },
          items: [
            'installation/configuration-contract',
            'installation/configuration-workflows',
            'installation/configuration-management',
            'installation/recipe-lifecycle',
            {
              type: 'category',
              label: 'Models',
              link: {
                type: 'doc',
                id: 'installation/model-configuration',
              },
              items: [
                'installation/catalog-backed-models',
                'installation/custom-models',
                'installation/model-reasoning',
                'installation/model-configuration-patterns',
              ],
            },
            {
              type: 'category',
              label: 'Router Model Catalog',
              link: {
                type: 'doc',
                id: 'installation/choose-a-model-for-a-signal',
              },
              items: [
                'installation/model-catalog',
                'installation/model-resource-migration',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Deploy',
          items: [
            {
              type: 'category',
              label: 'Docker',
              link: {
                type: 'doc',
                id: 'installation/docker',
              },
              items: [
                'installation/ollama',
              ],
            },
            {
              type: 'category',
              label: 'Kubernetes',
              items: [
                'installation/k8s/operator',
                {
                  type: 'category',
                  label: 'Gateways',
                  link: {
                    type: 'doc',
                    id: 'installation/k8s/gateways',
                  },
                  items: [
                    'installation/k8s/ai-gateway',
                    'installation/k8s/agentgateway',
                    'installation/k8s/streamed-extproc',
                    'installation/k8s/istio',
                    'installation/k8s/gateway-api-inference-extension',
                    'installation/k8s/gateway-testing',
                  ],
                },
                {
                  type: 'category',
                  label: 'Inference Platforms',
                  link: {
                    type: 'doc',
                    id: 'installation/k8s/inference-platforms',
                  },
                  items: [
                    'installation/k8s/production-stack',
                    'installation/k8s/aibrix',
                    'installation/k8s/llm-d',
                    {
                      type: 'doc',
                      id: 'installation/k8s/dynamo',
                      label: 'Integrate with NVIDIA Dynamo',
                    },
                  ],
                },
              ],
            },
            {
              type: 'category',
              label: 'Hardware',
              items: [
                'installation/amd-rocm',
                'installation/nvidia-cuda',
                'installation/ascend-npu',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Data & Storage',
          link: {
            type: 'doc',
            id: 'installation/storage-overview',
          },
          items: [
            'installation/valkey-memory',
            'installation/qdrant',
            'installation/milvus',
          ],
        },
        {
          type: 'category',
          label: 'Security',
          items: [
            'installation/security-hardening',
          ],
        },
        {
          type: 'category',
          label: 'Operations',
          items: [
            'installation/k8s/operator-operations',
            'installation/upgrade-rollback',
            'installation/migrate-to-v0-5',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Proposals',
      collapsed: false,
      link: {
        type: 'doc',
        id: 'proposals/index',
      },
      items: [
        {
          type: 'category',
          label: 'Routing & Selection',
          items: [
            'proposals/decision-model-first-class',
            'proposals/decision-ranking-semantics',
            'proposals/batch-and-capacity-aware-routing',
            'proposals/router-learning-memory-and-adaptations',
            'proposals/success-constrained-selection',
            'proposals/prompt-classification-routing',
          ],
        },
        {
          type: 'category',
          label: 'Workflows, Memory & Tools',
          items: [
            'proposals/router-flow-workflows',
            'proposals/agent-based-routing',
            'proposals/deliberation-algorithms',
            'proposals/agentic-memory',
            'proposals/agentic-rag',
            'proposals/advanced-tool-filtering',
          ],
        },
        {
          type: 'category',
          label: 'Safety & Resilience',
          items: [
            'proposals/model-execution-fallback',
            'proposals/Prism-153key',
            'proposals/hallucination-mitigation-milestone',
          ],
        },
        {
          type: 'category',
          label: 'Configuration & Protocols',
          items: [
            'proposals/open-intelligence-index-and-model-arena',
            'proposals/unified-model-catalog-and-evaluation-index',
            'proposals/standalone-mode',
            'proposals/unified-config-contract-v0-3',
            'proposals/multi-protocol-adaptor',
          ],
        },
        {
          type: 'category',
          label: 'Serving Integrations',
          items: [
            'proposals/production-stack-integration',
            'proposals/nvidia-dynamo-integration',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Model Training',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Start Here',
          collapsed: false,
          items: [
            'training/training-overview',
          ],
        },
        {
          type: 'category',
          label: 'Embedding Models',
          items: [
            'training/mmbert-32k-models',
            'training/router-embeddings-huggingface',
          ],
        },
        {
          type: 'category',
          label: 'Evaluate and Select',
          items: [
            'training/model-performance-eval',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Benchmarking',
      collapsed: false,
      items: [
        'benchmarking/overview',
        'benchmarking/open-intelligence-index',
        'benchmarking/agent-evaluation-loop',
        'benchmarking/custom-evaluations',
        'benchmarking/agent-routing-protection',
        {
          type: 'category',
          label: 'sr-bench 1.0',
          link: {
            type: 'doc',
            id: 'benchmarking/sr-bench/index',
          },
          items: [
            'benchmarking/sr-bench/shared-worker',
            'benchmarking/sr-bench/tasks-and-targets',
            'benchmarking/sr-bench/plan-and-run',
            'benchmarking/sr-bench/iterate',
            'benchmarking/sr-bench/results',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/cli',
        'api/router',
        'api/headers',
        'api/configuration-schema',
        'api/apiserver',
        'api/openapi',
        'api/session-identification',
        'api/semantic-router-crd',
        'api/crd-reference',
      ],
    },
    {
      type: 'category',
      label: 'Troubleshooting',
      collapsed: false,
      items: [
        'troubleshooting/network-tips',
        'troubleshooting/container-connectivity',
        'troubleshooting/common-errors',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      collapsed: false,
      items: [
        'community/overview',
        'community/model-provider-day-0-support',
        'community/development',
        'community/documentation',
        'community/translation-guide',
        'community/code-style',
      ],
    },
    'faq',
  ],
}

export default sidebars
