export interface WorkGroupPerson {
  name: string
  avatar: string
  profile: string
}

export interface WorkGroup {
  id: string
  name: string
  label: string
  charterIssue: number
  goal: string
  scope: string[]
  leads?: WorkGroupPerson[]
  members?: WorkGroupPerson[]
}

export const workGroups: WorkGroup[] = [
  {
    id: 'mom-routing',
    name: 'MoM & Routing',
    label: 'wg/mom-routing',
    charterIssue: 2965,
    goal: 'Build measurable and continuously improving model pools, recipes, and multi-model routing.',
    scope: [
      'Model pools, recipes, and lifecycle',
      'Multi-model collaboration algorithms and strategies',
      'Model-pool optimization and cross-model efficiency',
    ],
  },
  {
    id: 'router-models-inference-runtime',
    name: 'Router Models & Inference Runtime',
    label: 'wg/router-models-inference-runtime',
    charterIssue: 2966,
    goal: 'Improve Router Models and provide an extensible inference runtime for the model ecosystem.',
    scope: [
      'Built-in model post-training and release',
      'Router-native model families beyond BERT',
      'Self-improvement, fine-tuning, and runtime contracts',
    ],
  },
  {
    id: 'data-plane-networking',
    name: 'Data Plane & Networking',
    label: 'wg/data-plane-networking',
    charterIssue: 2967,
    goal: 'Run a portable and reliable online request path across standalone and gateway-integrated deployments.',
    scope: [
      'OpenAI-compatible standalone data plane',
      'Envoy ExtProc, gateways, and networking integrations',
      'Performance optimization, streaming, dispatch, retries, and telemetry',
    ],
  },
  {
    id: 'enterprise-environment',
    name: 'Enterprise & Environment',
    label: 'wg/enterprise-environment',
    charterIssue: 2968,
    goal: 'Deliver production-grade enterprise capabilities across supported environments and hardware.',
    scope: [
      'Multi-tenancy, identity, API keys, quotas, and audit',
      'Stability, scalability, observability, and lifecycle operations',
      'Multi-environment and multi-hardware support',
    ],
  },
  {
    id: 'agentic-context',
    name: 'Agentic & Context',
    label: 'wg/agentic-context',
    charterIssue: 2987,
    goal: 'Keep long-running agent workloads context-efficient, state-aware, bounded, and safe.',
    scope: [
      'Context compression, pruning, retrieval, and memory',
      'Session budgets, retention, recovery, and state',
      'Long-session model and workflow switching',
    ],
  },
  {
    id: 'developer-experience-ecosystem',
    name: 'Developer Experience & Ecosystem',
    label: 'wg/developer-experience-ecosystem',
    charterIssue: 2970,
    goal: 'Make vLLM Semantic Router easy to adopt, configure, extend, deploy, tune, and operate.',
    scope: [
      'CLI, Dashboard, APIs, configuration, recipes, and errors',
      'Agent skill and ecosystem integrations for deployment, tuning, and operations',
      'Documentation, blogs, video tutorials, and use-case sharing',
    ],
  },
  {
    id: 'evaluation-quality',
    name: 'Evaluation & Quality',
    label: 'wg/evaluation-quality',
    charterIssue: 2969,
    goal: 'Provide common evaluation and quality gates across every project direction.',
    scope: [
      'MoM, Router Model, context, and workflow evaluation',
      'Model cards, benchmarks, and reproducibility',
      'CI, E2E, compatibility, and regression gates',
    ],
  },
]
