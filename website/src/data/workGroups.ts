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
    leads: [
      {
        name: 'Xunzhuo Liu',
        avatar: 'https://github.com/Xunzhuo.png',
        profile: 'https://github.com/Xunzhuo',
      },
    ],
    members: [
      {
        name: 'Haichen Zhang',
        avatar: '/img/team/haichen.jpeg',
        profile: 'https://github.com/haic0',
      },
      {
        name: 'raghavchitkara',
        avatar: 'https://github.com/raghavchitkara36.png',
        profile: 'https://github.com/raghavchitkara36',
      },
      {
        name: 'Cerdore',
        avatar: 'https://github.com/Cerdore.png',
        profile: 'https://github.com/Cerdore',
      },
      {
        name: 'Ramakrishnan Sathyavageeswaran',
        avatar: 'https://github.com/ramkrishs.png',
        profile: 'https://github.com/ramkrishs',
      },
      {
        name: 'Chlins Zhang',
        avatar: 'https://github.com/chlins.png',
        profile: 'https://github.com/chlins',
      },
      {
        name: 'yaojiejia',
        avatar: 'https://github.com/yaojiejia.png',
        profile: 'https://github.com/yaojiejia',
      },
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
    leads: [
      {
        name: 'Kun-Tai Wu',
        avatar: 'https://github.com/WUKUNTAI-0211.png',
        profile: 'https://github.com/WUKUNTAI-0211',
      },
      {
        name: 'Theo Hsiung',
        avatar: 'https://github.com/theohsiung.png',
        profile: 'https://github.com/theohsiung',
      },
      {
        name: 'Ádám Kovács',
        avatar: 'https://github.com/adaamko.png',
        profile: 'https://github.com/adaamko',
      },
      {
        name: 'Ramakrishnan Sathyavageeswaran',
        avatar: 'https://github.com/ramkrishs.png',
        profile: 'https://github.com/ramkrishs',
      },
    ],
    members: [
      {
        name: 'raghavchitkara',
        avatar: 'https://github.com/raghavchitkara36.png',
        profile: 'https://github.com/raghavchitkara36',
      },
      {
        name: 'Park Soobin',
        avatar: 'https://github.com/subin9.png',
        profile: 'https://github.com/subin9',
      },
      {
        name: 'Chlins Zhang',
        avatar: 'https://github.com/chlins.png',
        profile: 'https://github.com/chlins',
      },
      {
        name: 'yaojiejia',
        avatar: 'https://github.com/yaojiejia.png',
        profile: 'https://github.com/yaojiejia',
      },
      {
        name: 'Guan-Ming Chiu',
        avatar: 'https://github.com/guan404ming.png',
        profile: 'https://github.com/guan404ming',
      },
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
    leads: [
      {
        name: 'Yang Wu',
        avatar: 'https://github.com/drivebyer.png',
        profile: 'https://github.com/drivebyer',
      },
      {
        name: 'Xunzhuo Liu',
        avatar: 'https://github.com/Xunzhuo.png',
        profile: 'https://github.com/Xunzhuo',
      },
    ],
    members: [
      {
        name: 'raghavchitkara',
        avatar: 'https://github.com/raghavchitkara36.png',
        profile: 'https://github.com/raghavchitkara36',
      },
      {
        name: 'Zireael',
        avatar: 'https://github.com/ZireaelK.png',
        profile: 'https://github.com/ZireaelK',
      },
    ],
  },
  {
    id: 'enterprise-environment',
    name: 'Enterprise & Environment',
    label: 'wg/enterprise-environment',
    charterIssue: 2968,
    goal: 'Deliver production-grade security, operations, and deployments across supported environments and hardware.',
    scope: [
      'Management authentication, authorization, identity integration, and audit',
      'Existing Insights, production observability, workload simulation, and capacity planning',
      'Stable, scalable deployment APIs and reference stacks',
      'Multi-environment and multi-hardware support',
    ],
    leads: [
      {
        name: 'Aayush Saini',
        avatar: 'https://github.com/AayushSaini101.png',
        profile: 'https://github.com/AayushSaini101',
      },
      {
        name: 'Akshay Viswanathan',
        avatar: 'https://github.com/akshayv.png',
        profile: 'https://github.com/akshayv',
      },
    ],
    members: [
      {
        name: 'Abhinav Mahajan',
        avatar: 'https://github.com/abhinav-m22.png',
        profile: 'https://github.com/abhinav-m22',
      },
      {
        name: 'Aakanksha Bhende',
        avatar: 'https://github.com/aakankshabhende.png',
        profile: 'https://github.com/aakankshabhende',
      },
      {
        name: 'Pranav Thakur',
        avatar: 'https://github.com/pranavthakur0-0.png',
        profile: 'https://github.com/pranavthakur0-0',
      },
    ],
  },
  {
    id: 'agentic-context',
    name: 'Agentic & Context',
    label: 'wg/agentic-context',
    charterIssue: 2987,
    goal: 'Manage context and safely select, hand off, and compose agent backends for long-running workloads.',
    scope: [
      'Context optimization, memory, and session state',
      'Agent backend selection, handoff, and composition',
      'Bounded multi-agent collaboration and long-session model or workflow switching',
    ],
    leads: [
      {
        name: 'Xunzhuo Liu',
        avatar: 'https://github.com/Xunzhuo.png',
        profile: 'https://github.com/Xunzhuo',
      },
      {
        name: 'Aayush Saini',
        avatar: 'https://github.com/AayushSaini101.png',
        profile: 'https://github.com/AayushSaini101',
      },
    ],
    members: [
      {
        name: 'Abhinav Mahajan',
        avatar: 'https://github.com/abhinav-m22.png',
        profile: 'https://github.com/abhinav-m22',
      },
      {
        name: 'yaojiejia',
        avatar: 'https://github.com/yaojiejia.png',
        profile: 'https://github.com/yaojiejia',
      },
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
    leads: [
      {
        name: 'Aayush Saini',
        avatar: 'https://github.com/AayushSaini101.png',
        profile: 'https://github.com/AayushSaini101',
      },
      {
        name: 'Wilson Wu',
        avatar: 'https://github.com/wilsonwu.png',
        profile: 'https://github.com/wilsonwu',
      },
    ],
    members: [
      {
        name: 'Abhinav Mahajan',
        avatar: 'https://github.com/abhinav-m22.png',
        profile: 'https://github.com/abhinav-m22',
      },
      {
        name: 'Mahdi Ghodsi',
        avatar: 'https://github.com/Mahdi-CV.png',
        profile: 'https://github.com/Mahdi-CV',
      },
      {
        name: 'Aakanksha Bhende',
        avatar: 'https://github.com/aakankshabhende.png',
        profile: 'https://github.com/aakankshabhende',
      },
      {
        name: 'Eda Zhou',
        avatar: 'https://github.com/edamamez.png',
        profile: 'https://github.com/edamamez',
      },
    ],
  },
  {
    id: 'evaluation-quality',
    name: 'Evaluation & Quality',
    label: 'wg/evaluation-quality',
    charterIssue: 2969,
    goal: 'Provide common evaluation and quality gates across every project direction.',
    scope: [
      'MoM, Router Model, agent, context, and workflow evaluation',
      'Model cards, benchmarks, and reproducibility',
      'CI, E2E, compatibility, and regression gates',
    ],
    leads: [
      {
        name: 'Xunzhuo Liu',
        avatar: 'https://github.com/Xunzhuo.png',
        profile: 'https://github.com/Xunzhuo',
      },
      {
        name: 'FAUST',
        avatar: 'https://github.com/FAUST-BENCHOU.png',
        profile: 'https://github.com/FAUST-BENCHOU',
      },
    ],
  },
]
