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
        'installation/deployment-options',
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
            {
              type: 'category',
              label: 'Heuristic',
              items: [
                'tutorials/signal/heuristic/authz',
                'tutorials/signal/heuristic/context',
                'tutorials/signal/heuristic/conversation',
                'tutorials/signal/heuristic/keyword',
                'tutorials/signal/heuristic/language',
                'tutorials/signal/heuristic/event',
                'tutorials/signal/heuristic/metadata',
                'tutorials/signal/heuristic/structure',
              ],
            },
            {
              type: 'category',
              label: 'Learned',
              items: [
                'tutorials/signal/learned/classifier',
                'tutorials/signal/learned/complexity',
                'tutorials/signal/learned/domain',
                'tutorials/signal/learned/embedding',
                'tutorials/signal/learned/modality',
                'tutorials/signal/learned/fact-check',
                'tutorials/signal/learned/jailbreak',
                'tutorials/signal/learned/pii',
                'tutorials/signal/learned/preference',
                'tutorials/signal/learned/reask',
                'tutorials/signal/learned/kb',
                'tutorials/signal/learned/user-feedback',
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
            'tutorials/decision/multi-step-pipeline',
            'tutorials/decision/retention',
          ],
        },
        {
          type: 'category',
          label: 'Algorithms',
          items: [
            'tutorials/algorithm/overview',
            {
              type: 'category',
              label: 'Selection',
              items: [
                'tutorials/algorithm/selection/automix',
                'tutorials/algorithm/selection/hybrid',
                'tutorials/algorithm/selection/kmeans',
                'tutorials/algorithm/selection/knn',
                'tutorials/algorithm/selection/latency-aware',
                'tutorials/algorithm/selection/mlp',
                'tutorials/algorithm/selection/multi-factor',
                'tutorials/algorithm/selection/prompt',
                'tutorials/algorithm/selection/router-dc',
                'tutorials/algorithm/selection/static',
                'tutorials/algorithm/selection/svm',
              ],
            },
            {
              type: 'category',
              label: 'Looper',
              items: [
                'tutorials/algorithm/looper/confidence',
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
                'tutorials/plugin/image-gen',
                'tutorials/plugin/context-compression',
                'tutorials/plugin/request-params',
                'tutorials/plugin/system-prompt',
                'tutorials/plugin/tool-selection',
                'tutorials/plugin/tools',
              ],
            },
            {
              type: 'category',
              label: 'Retrieval and Memory',
              items: [
                'tutorials/plugin/memory',
                'tutorials/plugin/rag',
                'tutorials/plugin/router-replay',
                'tutorials/plugin/response-cache',
              ],
            },
            {
              type: 'category',
              label: 'Safety and Generation',
              items: [
                'tutorials/plugin/content-safety',
                'tutorials/plugin/hallucination',
                'tutorials/plugin/response-jailbreak',
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
            'tutorials/global/safety-models-and-policy',
            'tutorials/global/remote-embeddings',
          ],
        },
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
            'installation/configuration-workflows',
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
          ],
        },
        {
          type: 'category',
          label: 'Runtime Reference',
          items: [
            'installation/native-backends',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Fleet Simulator',
      collapsed: false,
      items: [
        'fleet-sim/overview',
        'fleet-sim/getting-started',
        'fleet-sim/dashboard-integration',
        'fleet-sim/use-cases',
        'fleet-sim/sim-algorithms',
        'fleet-sim/power-model',
        'fleet-sim/guide',
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
            'proposals/batch-and-capacity-aware-routing',
            'proposals/router-learning-memory-and-adaptations',
            'proposals/prompt-classification-routing',
          ],
        },
        {
          type: 'category',
          label: 'Workflows, Memory & Tools',
          items: [
            'proposals/router-flow-workflows',
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
        'training/training-overview',
        'training/model-performance-eval',
        'training/ml-model-selection',
      ],
    },
    {
      type: 'category',
      label: 'Benchmarking',
      collapsed: false,
      items: [
        'benchmarking/overview',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/router',
        'api/apiserver',
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
        'troubleshooting/vsr-headers',
        'troubleshooting/common-errors',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      collapsed: false,
      items: [
        'community/overview',
        'community/development',
        'community/documentation',
        'community/translation-guide',
        'community/code-style',
      ],
    },
  ],
}

export default sidebars
