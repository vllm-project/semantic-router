// topology/constants.ts - Constants and Color Schemes

import { SignalType, PluginType, AlgorithmType } from './types'

// ============== Signal Icons ==============
export const SIGNAL_ICONS: Record<SignalType, string> = {
  keyword: 'KW',
  embedding: 'EMB',
  domain: 'DOM',
  fact_check: 'FC',
  user_feedback: 'UF',
  reask: 'RA',
  preference: 'PREF',
  language: 'LANG',
  context: 'CTX',
  structure: 'STR',
  complexity: 'CPX',
  modality: 'MOD',
  session: 'SES',
  authz: 'AUTH',
  jailbreak: 'JB',
  pii: 'PII',
  kb: 'KB',
  projection: 'PRJ',
}

// ============== Signal Colors (Gray Nodes, Green Paths) ==============
export const SIGNAL_COLORS: Record<SignalType, { background: string; border: string }> = {
  keyword: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
  embedding: { background: '#4a5568', border: '#2d3748' },    // Dark Gray
  domain: { background: '#4a5568', border: '#2d3748' },       // Dark Gray
  fact_check: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  user_feedback: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
  reask: { background: '#4a5568', border: '#2d3748' },        // Dark Gray
  preference: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  language: { background: '#4a5568', border: '#2d3748' },     // Dark Gray
  context: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
  structure: { background: '#4a5568', border: '#2d3748' },    // Dark Gray
  complexity: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  modality: { background: '#4a5568', border: '#2d3748' },     // Dark Gray
  authz: { background: '#4a5568', border: '#2d3748' },        // Dark Gray
  jailbreak: { background: '#4a5568', border: '#2d3748' },    // Dark Gray
  pii: { background: '#4a5568', border: '#2d3748' },          // Dark Gray
  kb: { background: '#4a5568', border: '#2d3748' },           // Dark Gray
  projection: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
}

// ============== Signal Latency ==============
export const SIGNAL_LATENCY: Record<SignalType, string> = {
  keyword: '<1ms',
  embedding: '10-50ms',
  domain: '10-50ms',
  fact_check: '10-50ms',
  user_feedback: '10-50ms',
  reask: '10-50ms',
  preference: '200-500ms',
  language: '<1ms',
  context: '<1ms',
  structure: '<1ms',
  complexity: '50-100ms',
  modality: '50-100ms',
  authz: '<1ms',
  jailbreak: '10-50ms',
  pii: '10-50ms',
  kb: '10-50ms',
  projection: '<1ms',
}

// ============== Plugin Icons ==============
export const PLUGIN_ICONS: Record<PluginType, string> = {
  'semantic-cache': 'SC',
  'memory': 'MEM',
  'system_prompt': 'SP',
  'header_mutation': 'HM',
  'hallucination': 'HAL',
  'router_replay': 'RR',
  'rag': 'RAG',
  'image_gen': 'IMG',
  'fast_response': 'FR',
  'request_params': 'RP',
  'response_jailbreak': 'RJ',
  'tools': 'TL',
}

// ============== Plugin Colors (NVIDIA Dark Theme) ==============
export const PLUGIN_COLORS: Record<PluginType, { background: string; border: string }> = {
  'semantic-cache': { background: '#76b900', border: '#5a8f00' },  // NVIDIA Green
  'memory': { background: '#3f6b73', border: '#2e4f55' },
  'system_prompt': { background: '#8fd400', border: '#76b900' },   // Light Green
  'header_mutation': { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
  'hallucination': { background: '#556b7d', border: '#3d4a59' },   // Cool Gray
  'router_replay': { background: '#6ba300', border: '#5a8f00' },   // Green (consistent with other plugins)
  'rag': { background: '#2f855a', border: '#276749' },
  'image_gen': { background: '#7b5ea7', border: '#5b3f86' },
  'fast_response': { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  'request_params': { background: '#805ad5', border: '#6b46c1' },
  'response_jailbreak': { background: '#c05621', border: '#9c4221' },
  'tools': { background: '#5a6c7d', border: '#3d4a59' },
}

// ============== Algorithm Icons ==============
export const ALGORITHM_ICONS: Record<AlgorithmType, string> = {
  confidence: 'CF',
  concurrent: 'CC',
  sequential: 'SEQ',
  ratings: 'RT',
  static: 'ST',
  elo: 'ELO',
  router_dc: 'RDC',
  automix: 'AM',
  hybrid: 'HY',
  remom: 'RM',
  latency_aware: 'LAT',
}

// ============== Algorithm Colors (NVIDIA Dark Theme) ==============
export const ALGORITHM_COLORS: Record<AlgorithmType, { background: string; border: string }> = {
  confidence: { background: '#76b900', border: '#5a8f00' },    // NVIDIA Green
  concurrent: { background: '#5a6c7d', border: '#3d4a59' },    // Blue Gray
  sequential: { background: '#4a5568', border: '#2d3748' },    // Dark Gray
  ratings: { background: '#8fd400', border: '#76b900' },       // Light Green
  static: { background: '#606c7a', border: '#3d4a59' },        // Slate Gray
  elo: { background: '#718096', border: '#4a5568' },           // Medium Gray
  router_dc: { background: '#556b7d', border: '#3d4a59' },     // Cool Gray
  remom: { background: '#76b900', border: '#5a8f00' },         // NVIDIA Green (same as plugins)
  automix: { background: '#5d6d7e', border: '#3d4a59' },       // Steel Gray
  hybrid: { background: '#4a5568', border: '#2d3748' },        // Dark Gray
  session_aware: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray
  latency_aware: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray
}

// ============== Reasoning Effort Display (NVIDIA Dark Theme) ==============
export const REASONING_EFFORT_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'low': { icon: 'L', label: 'Low', color: '#8fd400' },       // Light Green
  'medium': { icon: 'M', label: 'Medium', color: '#76b900' }, // NVIDIA Green
  'high': { icon: 'H', label: 'High', color: '#5a8f00' },     // Dark Green
}

export const MODEL_NODE_WIDTH = 220

// ============== Global Plugin Display (NVIDIA Dark Theme) ==============
export const GLOBAL_PLUGIN_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'prompt_guard': { icon: 'PG', label: 'Jailbreak Guard', color: '#718096' },   // Medium Gray
  'pii_detection': { icon: 'PII', label: 'PII Detection', color: '#5a6c7d' },   // Blue Gray
  'semantic_cache': { icon: 'SC', label: 'Semantic Cache', color: '#76b900' },  // NVIDIA Green
}

// ============== Node Colors (Gray Nodes, Green Paths) ==============
export const NODE_COLORS = {
  client: { background: '#76b900', border: '#5a8f00' },        // NVIDIA Green (Client stays green)
  decision: {
    normal: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
    reasoning: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
    unreachable: { background: '#3d4a59', border: '#2d3748' }, // Very Dark Gray
  },
  model: {
    standard: { background: '#5a6c7d', border: '#3d4a59' },    // Blue Gray (Models gray)
    reasoning: { background: '#5a6c7d', border: '#3d4a59' },   // Blue Gray (Models gray)
  },
  classification: { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
  disabled: { background: '#3d4a59', border: '#2d3748' },      // Very Dark Gray
}

// ============== Edge Colors (Green Paths) ==============
export const EDGE_COLORS = {
  normal: '#76b900',      // NVIDIA Green (All paths green)
  reasoning: '#76b900',   // NVIDIA Green (All paths green)
  active: '#76b900',      // NVIDIA Green
  disabled: '#3d4a59',    // Very Dark Gray
  highlighted: '#8fd400', // Light Green (Highlighted paths brighter)
}

// ============== Layout Configuration ==============
export const LAYOUT_CONFIG = {
  columns: {
    client: { x: 50 },
    globalPlugins: { x: 200 },
    signals: { x: 350 },
    decisions: { x: 530 },
    algorithms: { x: 780 },
    plugins: { x: 930 },
    models: { x: 1100 },
  },
  nodeWidth: 180,
  nodeHeight: 100,
  verticalSpacing: 15,   // Minimum space between nodes in same column (reduced from 25)
  groupSpacing: 20,      // Extra space between signal groups (reduced from 35)
  // Base heights for different node types (actual height = base + content)
  decisionBaseHeight: 120,   // Decision nodes base
  decisionConditionHeight: 22, // Per condition line
  signalGroupBaseHeight: 80,   // Signal group base
  signalItemHeight: 20,        // Per signal item
  pluginChainBaseHeight: 60,   // Plugin chain base
  pluginItemHeight: 20,        // Per plugin item
}

// ============== Three-Layer LR Layout (Brain Page) ==============
// Keep these values configurable so dense topologies can be compacted
// without changing layout code.
export const TOPOLOGY_LAYER_LAYOUT = {
  x: {
    client: 0,
    signals: 340,
    projections: 640,
    decisions: 1040,
    algorithms: 1520,
    pluginChains: 1840,
    models: 2340,
  },
  framePadding: {
    client: 36,
    signals: 54,
    projections: 60,
    decisions: 84,
    algorithms: 68,
    pluginChains: 72,
    models: 88,
  },
  horizontalGap: {
    clientToSignals: 156,
    signalsToProjections: 198,
    projectionsToDecisions: 244,
    decisionsToAlgorithms: 286,
    algorithmsToPluginChains: 196,
    pluginChainsToModels: 248,
  },
  verticalSpacing: {
    client: { base: 0, min: 0, compactThreshold: 1, compactStep: 0 },
    signals: { base: 34, min: 14, compactThreshold: 8, compactStep: 2.5 },
    projections: { base: 38, min: 16, compactThreshold: 6, compactStep: 2.5 },
    decisions: { base: 42, min: 16, compactThreshold: 9, compactStep: 3 },
    algorithms: { base: 48, min: 20, compactThreshold: 4, compactStep: 4 },
    pluginChains: { base: 52, min: 20, compactThreshold: 5, compactStep: 4 },
    models: { base: 76, min: 36, compactThreshold: 9, compactStep: 2.5 },
  },
  lanes: {
    signals: {
      enableAt: 7,
      maxPerLane: 6,
      maxLanes: 2,
      laneGap: 212,
    },
    projections: {
      enableAt: 5,
      maxPerLane: 4,
      maxLanes: 2,
      laneGap: 228,
    },
    decisions: {
      enableAt: 7,
      maxPerLane: 6,
      maxLanes: 6,
      laneGap: 250,
    },
    algorithms: {
      laneGap: 220,
    },
    pluginChains: {
      laneGap: 220,
    },
    models: {
      enableAt: 10,
      maxPerLane: 8,
      maxLanes: 2,
      laneGap: 250,
    },
  },
} as const

// ============== Signal Types Array ==============
export const SIGNAL_TYPES: SignalType[] = [
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'reask',
  'preference',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
  'kb',
  'projection',
]

// ============== Plugin Types Array ==============
export const PLUGIN_TYPES: PluginType[] = [
  'semantic-cache',
  'memory',
  'system_prompt',
  'header_mutation',
  'hallucination',
  'router_replay',
  'rag',
  'image_gen',
  'fast_response',
  'request_params',
  'response_jailbreak',
  'tools',
]
