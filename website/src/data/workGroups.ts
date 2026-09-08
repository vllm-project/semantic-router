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
      {
        name: 'Hui Ding',
        avatar: 'https://github.com/truddy0.png',
        profile: 'https://github.com/truddy0',
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
      {
        name: 'bugkeep',
        avatar: 'https://github.com/bugkeep.png',
        profile: 'https://github.com/bugkeep',
      },
      {
        name: 'JiaoliangYu',
        avatar: 'https://github.com/JiaoliangYu.png',
        profile: 'https://github.com/JiaoliangYu',
      },
      {
        name: 'Xuetao Li',
        avatar: 'https://github.com/Alanxtl.png',
        profile: 'https://github.com/Alanxtl',
      },
      {
        name: 'Nanasis',
        avatar: 'https://github.com/nanasis.png',
        profile: 'https://github.com/nanasis',
      },
      {
        name: 'karthikeyan1592',
        avatar: 'https://github.com/karthikeyan1592.png',
        profile: 'https://github.com/karthikeyan1592',
      },
      {
        name: 'Binbin Zhang',
        avatar: 'https://github.com/Bevisy.png',
        profile: 'https://github.com/Bevisy',
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
      {
        name: 'Hikari',
        avatar: 'https://github.com/altale.png',
        profile: 'https://github.com/altale',
      },
      {
        name: 'Binbin Zhang',
        avatar: 'https://github.com/Bevisy.png',
        profile: 'https://github.com/Bevisy',
      },
      {
        name: 'Xuge',
        avatar: 'https://github.com/xuuuge.png',
        profile: 'https://github.com/xuuuge',
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
      {
        name: 'kzos',
        avatar: 'https://github.com/kzos.png',
        profile: 'https://github.com/kzos',
      },
    ],
  },
  {
    id: 'agentic-context',
    name: 'Agentic & Context',
    label: 'wg/agentic-context',
    charterIssue: 2987,
    goal: 'Optimize bounded context, memory, and session continuity for long-running and agentic workloads.',
    scope: [
      'Context optimization, prompt-visible memory, and session state',
      'Session budgets, tool-loop continuity, and safe model or workflow switching',
      'Typed agent-aware boundaries and bounded collaboration receipts for external runtimes',
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
      {
        name: 'Binbin Zhang',
        avatar: 'https://github.com/Bevisy.png',
        profile: 'https://github.com/Bevisy',
      },
      {
        name: 'Shrek Luzz',
        avatar: 'https://github.com/Zheng-Lu.png',
        profile: 'https://github.com/Zheng-Lu',
      },
    ],
  },
  {
    id: 'developer-experience-ecosystem',
    name: 'Developer Experience & Ecosystem',
    label: 'wg/developer-experience-ecosystem',
    charterIssue: 2970,
    goal: 'Make vLLM Semantic Router easy to adopt, configure, extend, diagnose, and contribute to.',
    scope: [
      'First-run CLI, configuration, recipes, errors, and troubleshooting',
      'Dashboard workflows built on canonical Router and deployment contracts',
      'Reviewed agent-assisted workflows, documentation, localization, and ecosystem guides',
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
      {
        name: 'ZiYiMing',
        avatar: 'https://github.com/Zi-Yi-Ming.png',
        profile: 'https://github.com/Zi-Yi-Ming',
      },
    ],
  },
  {
    id: 'evaluation-quality',
    name: 'Evaluation & Quality',
    label: 'wg/evaluation-quality',
    charterIssue: 2969,
    goal: 'Make every supported capability measurable and every change verifiable.',
    scope: [
      'Decision-level routing and first-class MoM evaluation',
      'Performance coverage, reports, baselines, and hardware qualification',
      'CI, behavioral E2E, compatibility, security, and regression gates',
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
    members: [
      {
        name: 'Nanasis',
        avatar: 'https://github.com/nanasis.png',
        profile: 'https://github.com/nanasis',
      },
      {
        name: 'pikachu',
        avatar: 'https://github.com/yu3zhang1.png',
        profile: 'https://github.com/yu3zhang1',
      },
    ],
  },
]
