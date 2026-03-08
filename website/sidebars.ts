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
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Concepts',
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
      label: 'Get Started',
      items: [
        'installation/installation',
        'installation/configuration',
        'installation/docker-compose',
        'installation/milvus',
      ],
    },
    {
      type: 'category',
      label: 'Deploy & Integrate',
      items: [
        'installation/k8s/operator',
        {
          type: 'category',
          label: 'Gateways',
          items: [
            'installation/k8s/ai-gateway',
            'installation/k8s/istio',
            'installation/k8s/gateway-api-inference-extension',
          ],
        },
        {
          type: 'category',
          label: 'Frameworks',
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
      label: 'Capabilities',
      items: [
        {
          type: 'category',
          label: 'Intelligent Route',
          items: [
            'tutorials/intelligent-route/keyword-routing',
            'tutorials/intelligent-route/embedding-routing',
            'tutorials/intelligent-route/domain-routing',
            'tutorials/intelligent-route/context-routing',
            'tutorials/intelligent-route/complexity-routing',
            'tutorials/intelligent-route/fact-check-routing',
            'tutorials/intelligent-route/user-feedback-routing',
            'tutorials/intelligent-route/preference-routing',
            'tutorials/intelligent-route/mcp-routing',
            'tutorials/intelligent-route/lora-routing',
            'tutorials/intelligent-route/router-memory',
            {
              type: 'category',
              label: 'Model Selection',
              items: [
                'tutorials/intelligent-route/model-selection/overview',
                'tutorials/intelligent-route/model-selection/choosing-algorithm',
                'tutorials/intelligent-route/model-selection/static',
                'tutorials/intelligent-route/model-selection/elo',
                'tutorials/intelligent-route/model-selection/router-dc',
                'tutorials/intelligent-route/model-selection/automix',
                'tutorials/intelligent-route/model-selection/hybrid',
                'tutorials/intelligent-route/model-selection/thompson-sampling',
                'tutorials/intelligent-route/model-selection/gmtrouter',
                'tutorials/intelligent-route/model-selection/router-r1',
                'tutorials/intelligent-route/model-selection/troubleshooting',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Response API',
          items: [
            'tutorials/response-api/redis-storage',
            'tutorials/response-api/redis-cluster-storage',
          ],
        },
        {
          type: 'category',
          label: 'Semantic Cache',
          items: [
            'tutorials/semantic-cache/in-memory-cache',
            'tutorials/semantic-cache/redis-cache',
            'tutorials/semantic-cache/milvus-cache',
            'tutorials/semantic-cache/hybrid-cache',
          ],
        },
        {
          type: 'category',
          label: 'Content Safety',
          items: [
            'tutorials/content-safety/pii-detection',
            'tutorials/content-safety/jailbreak-protection',
            'tutorials/content-safety/hallucination-detection',
          ],
        },
        {
          type: 'category',
          key: 'capabilities-observability',
          label: 'Observability',
          items: [
            'tutorials/observability/metrics',
            'tutorials/observability/dashboard',
            'tutorials/observability/distributed-tracing',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        {
          type: 'category',
          key: 'operations-observability',
          label: 'Observability',
          items: [
            'tutorials/observability/metrics',
            'tutorials/observability/dashboard',
            'tutorials/observability/distributed-tracing',
          ],
        },
        {
          type: 'category',
          label: 'Performance Tuning',
          items: [
            'tutorials/performance-tuning/modernbert-32k-performance',
            {
              type: 'category',
              label: 'ModernBERT 32K Reference Notes',
              items: [
                'tutorials/performance-tuning/modernbert-32k-docs/modernbert-32k-deployment-guide',
                'tutorials/performance-tuning/modernbert-32k-docs/modernbert-32k-performance-validation',
                'tutorials/performance-tuning/modernbert-32k-docs/modernbert-32k-big-batch-test-plan',
                'tutorials/performance-tuning/modernbert-32k-docs/modernbert-32k-long-context-test-plan',
              ],
            },
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
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'api/router',
        'api/classification',
        'api/crd-reference',
      ],
    },
    {
      type: 'category',
      label: 'Research & Roadmap',
      items: [
        {
          type: 'category',
          label: 'Training',
          items: [
            'training/training-overview',
            'training/model-performance-eval',
            'training/ml-model-selection',
          ],
        },
        {
          type: 'category',
          label: 'Proposals',
          items: [
            'proposals/advanced-tool-filtering',
            'proposals/hallucination-mitigation-milestone',
            'proposals/prompt-classification-routing',
            'proposals/nvidia-dynamo-integration',
            'proposals/production-stack-integration',
            'proposals/multi-protocol-adaptor',
            'proposals/agentic-rag',
            'proposals/agentic-memory',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Contribute',
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
