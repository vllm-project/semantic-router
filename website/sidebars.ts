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
        'overview/semantic-router-overview',
        'overview/mixture-of-models',
        {
          type: 'category',
          label: 'Architecture',
          items: [
            'overview/architecture/system-architecture',
            'overview/architecture/envoy-extproc',
            'overview/architecture/router-implementation',
          ],
        },
        {
          type: 'category',
          label: 'Categories',
          items: [
            'overview/categories/overview',
            'overview/categories/supported-categories',
            'overview/categories/configuration',
            'overview/categories/technical-details',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Installation',
      items: [
        'installation/installation',
        'installation/kubernetes',
        'installation/docker-compose',
        'installation/configuration',
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
            'tutorials/intelligent-route/overview',
            'tutorials/intelligent-route/reasoning',
          ],
        },
        {
          type: 'category',
          label: 'Semantic Cache',
          items: [
            'tutorials/semantic-cache/overview',
            'tutorials/semantic-cache/in-memory-cache',
            'tutorials/semantic-cache/milvus-cache',
          ],
        },
        {
          type: 'category',
          label: 'Content Safety',
          items: [
            'tutorials/content-safety/overview',
            'tutorials/content-safety/pii-detection',
            'tutorials/content-safety/jailbreak-protection',
          ],
        },
        {
          type: 'category',
          label: 'Observability',
          items: [
            'tutorials/observability/overview',
            'tutorials/observability/metrics',
            'tutorials/observability/distributed-tracing',
            'tutorials/observability/open-webui-integration',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Proposals',
      items: [
        'proposals/prompt-classification-routing',
        'proposals/nvidia-dynamo-integration',
        'proposals/production-stack-integration',
      ],
    },
    {
      type: 'category',
      label: 'Model Training',
      items: [
        'training/training-overview',
        'training/model-performance-eval',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/router',
        'api/classification',
      ],
    },
    {
      type: 'category',
      label: 'Troubleshooting',
      items: [
        'troubleshooting/network-tips',
        'troubleshooting/container-connectivity',
        'troubleshooting/vsr-headers',
      ],
    },
  ],
}

export default sidebars
