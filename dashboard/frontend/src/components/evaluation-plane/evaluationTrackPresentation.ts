import type { EvaluationTrackId } from '../../types/evaluationPlane'

export const TRACK_PRESENTATION: Record<EvaluationTrackId, { label: string; description: string }> =
  {
    routing: {
      label: 'Routing',
      description:
        'Decision quality, coverage, abstention, fallbacks, and missed best-model opportunities.',
    },
    model_pool: {
      label: 'Model pool',
      description:
        'Quality and reliability of each model, complementary strengths, unique wins, and the best possible pool outcome.',
    },
    joint: {
      label: 'Routing and model pool',
      description:
        'End-to-end quality, reliability, latency, and cost, including the gap from the best available model.',
    },
    agentic: {
      label: 'Agent tasks',
      description:
        'Task completion, tool-use policy, state and privacy, recovery from failures, latency, and cost.',
    },
    multimodal: {
      label: 'Multimodal',
      description:
        'Input capability matching, grounded response quality, reliability, and privacy for text and non-text requests.',
    },
    preference: {
      label: 'Preference',
      description:
        'Offline preference agreement and statistically valid online preference outcomes.',
    },
    safety: {
      label: 'Safety',
      description: 'Policy adherence, correct blocking behavior, privacy, and unsafe regressions.',
    },
    capacity: {
      label: 'Capacity',
      description:
        'Throughput, tail latency, error bounds, stability, service-objective headroom, and test cost across repeated load levels.',
    },
  }
