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
          label: 'Intelligent Route',
          items: [
            'tutorials/intelligent-route/keyword-routing',
            'tutorials/intelligent-route/embedding-routing',
            'tutorials/intelligent-route/domain-routing',
            'tutorials/intelligent-route/fact-check-routing',
            'tutorials/intelligent-route/user-feedback-routing',
            'tutorials/intelligent-route/preference-routing',
            'tutorials/intelligent-route/mcp-routing',
            'tutorials/intelligent-route/lora-routing',
            'tutorials/intelligent-route/router-memory',
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
      label: 'Proposals',
      items: [
        'proposals/hallucination-mitigation-milestone',
        'proposals/prompt-classification-routing',
        'proposals/nvidia-dynamo-integration',
        'proposals/production-stack-integration',
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
      label: 'Cookbook',
      items: [
        'cookbook/classifier-tuning',
        'cookbook/pii-policy',
        'cookbook/vllm-endpoints',
        'cookbook/anthropic-models',
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
