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
      items: [
        'overview/goals',
        'overview/semantic-router-overview',
        'overview/collective-intelligence',
        'overview/signal-driven-decisions',
        'overview/mom-model-family',
      ],
    },
    {
      type: 'category',
      label: 'Installation',
      items: [
        'installation/installation',
        'installation/configuration',
        {
          type: 'category',
          label: 'Install with Gateways',
          items: [
            'installation/k8s/ai-gateway',
            'installation/k8s/istio',
            'installation/k8s/gateway-api-inference-extension',
          ],
        },
        {
          type: 'category',
          label: 'Install with Frameworks',
          items: [
            'installation/k8s/production-stack',
            'installation/k8s/aibrix',
            'installation/k8s/llm-d',
            'installation/k8s/dynamo',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Capacities',
      items: [
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
                'tutorials/signal/heuristic/keyword',
                'tutorials/signal/heuristic/language',
                'tutorials/signal/heuristic/modality',
              ],
            },
            {
              type: 'category',
              label: 'Learned',
              items: [
                'tutorials/signal/learned/complexity',
                'tutorials/signal/learned/domain',
                'tutorials/signal/learned/embedding',
                'tutorials/signal/learned/fact-check',
                'tutorials/signal/learned/jailbreak',
                'tutorials/signal/learned/pii',
                'tutorials/signal/learned/preference',
                'tutorials/signal/learned/user-feedback',
              ],
            },
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
          ],
        },
        {
          type: 'category',
          label: 'Algorithms',
          items: [
            'tutorials/algorithm/overview',
            'tutorials/algorithm/selection',
            'tutorials/algorithm/looper',
          ],
        },
        {
          type: 'category',
          label: 'Plugins',
          items: [
            'tutorials/plugin/overview',
            'tutorials/plugin/response-and-mutation',
            'tutorials/plugin/retrieval-and-memory',
            'tutorials/plugin/safety-and-generation',
          ],
        },
        {
          type: 'category',
          label: 'Global',
          items: [
            'tutorials/global/overview',
            'tutorials/global/api-and-observability',
            'tutorials/global/stores-and-tools',
            'tutorials/global/safety-models-and-policy',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Proposals',
      items: [
        'proposals/unified-config-contract-v0-3',
        'proposals/hallucination-mitigation-milestone',
        'proposals/prompt-classification-routing',
        'proposals/nvidia-dynamo-integration',
        'proposals/production-stack-integration',
        'proposals/multi-protocol-adaptor',
        'proposals/agentic-rag',
        'proposals/agentic-memory',
      ],
    },
    {
      type: 'category',
      label: 'Model Training',
      items: [
        'training/training-overview',
        'training/model-performance-eval',
        'training/ml-model-selection',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/router',
        'api/classification',
        'api/crd-reference',
      ],
    },
    {
      type: 'category',
      label: 'Troubleshooting',
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
      items: [
        'community/overview',
        'community/development',
        'community/documentation',
        'community/code-style',
      ],
    },
  ],
}

export default sidebars
