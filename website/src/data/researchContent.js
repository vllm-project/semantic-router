export const researchPapers = [
  {
    id: 'vsr-position-paper',
    type: 'paper',
    spotlight: true,
    categoryLabel: 'POSITION PAPER',
    title: 'vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models',
    authors: 'vLLM Semantic Router Team',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We introduce vLLM Semantic Router, a signal-driven decision routing framework for Mixture-of-Modality deployments that composes heterogeneous signals into deployment-specific routing policies across cost, privacy, latency, and safety constraints.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.04444', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 10,
  },
  {
    id: 'visual-confused-deputy',
    type: 'paper',
    title: 'Visual Confused Deputy: Exploiting and Defending Perception Failures in Computer-Using Agents',
    authors: 'Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We formalize the visual confused deputy as a security failure mode in computer-using agents and introduce a dual-channel guardrail that independently checks click targets and action reasoning before execution.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.14707', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 20,
  },
  {
    id: 'oats-tool-selection',
    type: 'paper',
    title: 'Outcome-Aware Tool Selection for Semantic Routers: Latency-Constrained Learning Without LLM Inference',
    authors: 'Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We introduce Outcome-Aware Tool Selection (OATS), an offline embedding refinement method that improves semantic-router tool ranking under single-digit millisecond CPU budgets without adding serving-time model inference.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.13426', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 30,
  },
  {
    id: 'adaptive-vlm-routing',
    type: 'paper',
    title: 'Adaptive Vision-Language Model Routing for Computer Use Agents',
    authors: 'Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We propose Adaptive VLM Routing (AVR), which estimates action difficulty and routes computer-use agent steps to the cheapest model that still satisfies a target reliability threshold.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.12823', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 40,
  },
  {
    id: 'routing-without-dedicated-gpu',
    type: 'paper',
    title: '98× Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router',
    authors: 'Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We combine Flash Attention, prompt compression, and near-streaming body processing to cut routing latency from seconds to tens of milliseconds while keeping the router lightweight enough to share hardware with serving.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.12646', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 50,
  },
  {
    id: 'inference-fleet-sim',
    type: 'paper',
    title: 'inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference',
    authors: 'Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We present a queueing-theory-grounded fleet planner and discrete-event simulator for sizing multi-pool LLM GPU fleets against P99 TTFT targets, without requiring hardware profiling runs up front.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.16054', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 55,
  },
  {
    id: 'fleetopt',
    type: 'paper',
    title: 'FleetOpt: Analytical Fleet Provisioning for LLM Inference with Compress-and-Route as Implementation Mechanism',
    authors: 'Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We derive the minimum-cost two-pool LLM fleet directly from the workload CDF and P99 TTFT target, then use Compress-and-Route to make the optimal boundary deployable in practice.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.16514', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 56,
  },
  {
    id: 'one-over-w-law',
    type: 'paper',
    title: 'The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency',
    authors: 'Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We derive the 1/W law showing that tokens per watt roughly halve whenever the serving context window doubles, making context-length routing topology a larger energy-efficiency lever than a pure GPU generation upgrade.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.17280', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 57,
  },
  {
    id: 'conflict-free-policy-languages',
    type: 'paper',
    title: 'Conflict-Free Policy Languages for Probabilistic ML Predicates: A Framework and Case Study with the Semantic Router DSL',
    authors: 'Xunzhuo Liu, Hao Wu, Huamin Chen, Bowei He, Xue Liu',
    venue: 'arXiv Technical Report',
    year: '2026',
    abstract: 'We show how probabilistic ML predicates in policy languages can silently co-fire on the same query, and implement conflict detection plus a softmax-based prevention mechanism in the Semantic Router DSL.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2603.18174', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 58,
  },
  {
    id: 'when-to-reason',
    type: 'paper',
    title: 'When to Reason: Semantic Router for vLLM',
    authors: 'Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen',
    venue: 'NeurIPS - MLForSys',
    year: '2025',
    abstract: 'We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2510.08731', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 60,
  },
  {
    id: 'category-aware-semantic-caching',
    type: 'paper',
    title: 'Category-Aware Semantic Caching for Heterogeneous LLM Workloads',
    authors: 'Chen Wang, Xunzhuo Liu, Yue Zhu, Alaa Youssef, Priya Nagpurkar, Huamin Chen',
    venue: '',
    year: '2025',
    abstract: 'We present a category-aware semantic caching where similarity thresholds, TTLs, and quotas vary by query category, with a hybrid architecture separating in-memory HNSW search from external document storage.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2510.26835', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 70,
  },
  {
    id: 'sirp',
    type: 'paper',
    title: 'Semantic Inference Routing Protocol (SIRP)',
    authors: 'Huamin Chen, Luay Jalil',
    venue: 'Internet Engineering Task Force (IETF)',
    year: '2025',
    abstract: 'This document specifies the Semantic Inference Routing Protocol (SIRP), a framework for content-level classification and semantic routing in AI inference systems.',
    links: [
      { type: 'paper', url: 'https://datatracker.ietf.org/doc/html/draft-chen-nmrg-semantic-inference-routing', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 80,
  },
  {
    id: 'multi-provider-inference-api',
    type: 'paper',
    title: 'Multi-Provider Extensions for Agentic AI Inference APIs',
    authors: 'H. Chen, L. Jalil, N. Cocker',
    venue: 'Internet Engineering Task Force (IETF) - Network Management Research Group',
    year: '2025',
    abstract: 'This document specifies multi-provider extensions for agentic AI inference APIs. Published: 20 October 2025. Intended Status: Informational. Expires: 23 April 2026.',
    links: [
      { type: 'paper', url: 'https://www.ietf.org/archive/id/draft-chen-nmrg-multi-provider-inference-api-00.html', label: 'Paper' },
    ],
    featured: true,
    sortOrder: 90,
  },
]

export const researchTalks = [
  {
    id: 'kubecon-intelligent-llm-routing',
    type: 'talk',
    title: 'Intelligent LLM Routing: A New Paradigm for Multi-Model AI Orchestration in Kubernetes',
    speakers: 'Chen Wang, Huamin Chen',
    venue: 'KubeCon NA 2025',
    organization: '',
    year: '2025',
    abstract: 'This research-driven talk introduces a novel architecture paradigm that complements recent advances in timely intelligent inference routing for large language models.',
    links: [
      { type: 'event', url: 'https://kccncna2025.sched.com/event/27FaI?iframe=no', label: 'Event page' },
    ],
    featured: true,
    sortOrder: 10,
  },
  {
    id: 'vllm-meetup-beijing',
    type: 'talk',
    title: 'vLLM Semantic Router: Unlock the Power of Intelligent Routing',
    speakers: 'Xunzhuo Liu',
    venue: 'vLLM Meetup Beijing',
    organization: '',
    year: '2025',
    abstract: 'A deep dive into vLLM Semantic Router capabilities, demonstrating how intelligent routing can unlock new possibilities for efficient LLM inference.',
    links: [
      { type: 'event', url: 'https://drive.google.com/drive/folders/1nQJ8ZkLSjKxvu36sSHaceVXtttbLvvu-', label: 'Watch recording' },
    ],
    featured: true,
    sortOrder: 20,
  },
  {
    id: 'vllm-office-hours',
    type: 'talk',
    title: 'AI-Powered vLLM Semantic Router',
    speakers: 'Huamin Chen',
    venue: 'vLLM Office Hours',
    organization: '',
    year: '2025',
    abstract: 'An overview of AI-powered features in vLLM Semantic Router, showcasing the latest developments and community contributions.',
    links: [
      { type: 'video', url: 'https://www.youtube.com/live/b-ciRqvbtsk', label: 'Watch recording' },
    ],
    featured: true,
    sortOrder: 30,
  },
]

export function sortResearchEntries(entries) {
  return [...entries].sort((a, b) => {
    if (Boolean(a.spotlight) !== Boolean(b.spotlight)) {
      return a.spotlight ? -1 : 1
    }

    if (Boolean(a.featured) !== Boolean(b.featured)) {
      return a.featured ? -1 : 1
    }

    if (a.year !== b.year) {
      return parseInt(b.year, 10) - parseInt(a.year, 10)
    }

    if ((a.sortOrder ?? 0) !== (b.sortOrder ?? 0)) {
      return (a.sortOrder ?? 0) - (b.sortOrder ?? 0)
    }

    return String(a.id).localeCompare(String(b.id))
  })
}
