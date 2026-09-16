// topology/constants.ts - Constants and Color Schemes

import { SignalType, PluginType, AlgorithmType } from './types'
import {
  ALGORITHM_TYPES as ROUTER_ALGORITHM_TYPES,
  PLUGIN_TYPES as ROUTER_PLUGIN_TYPES,
  SIGNAL_TYPES as ROUTER_SIGNAL_TYPES,
} from '../../generated/routerConfigContract'

function compactSurfaceIcon(value: string): string {
  const words = value.split('_').filter(Boolean)
  if (words.length > 1) return words.map((word) => word[0]?.toUpperCase()).join('')
  return value.slice(0, 3).toUpperCase()
}

export const SIGNAL_TYPES: SignalType[] = [...ROUTER_SIGNAL_TYPES, 'projection']
export const PLUGIN_TYPES: PluginType[] = [...ROUTER_PLUGIN_TYPES]
export const ALGORITHM_TYPES: AlgorithmType[] = [...ROUTER_ALGORITHM_TYPES]
const ALGORITHM_DISPLAY_TYPES: AlgorithmType[] = [...ALGORITHM_TYPES, 'concurrent', 'sequential']

// ============== Signal Icons ==============
const SIGNAL_ICON_OVERRIDES: Partial<Record<SignalType, string>> = {
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
  authz: 'AUTH',
  jailbreak: 'JB',
  hallucination: 'HL',
  pii: 'PII',
  kb: 'KB',
  conversation: 'CONV',
  event: 'EVT',
  projection: 'PRJ',
}

export const SIGNAL_ICONS = Object.fromEntries(
  SIGNAL_TYPES.map((type) => [type, SIGNAL_ICON_OVERRIDES[type] || compactSurfaceIcon(type)]),
) as Record<SignalType, string>

// ============== Signal Colors (Gray Nodes, Alloy Paths) ==============
export const SIGNAL_COLORS = Object.fromEntries(
  SIGNAL_TYPES.map((type) => [type, { background: '#4a5568', border: '#2d3748' }]),
) as Record<SignalType, { background: string; border: string }>

// ============== Signal Latency ==============
const SIGNAL_LATENCY_OVERRIDES: Partial<Record<SignalType, string>> = {
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
  hallucination: '200-500ms',
  pii: '10-50ms',
  kb: '10-50ms',
  conversation: '<1ms',
  event: '<1ms',
  projection: '<1ms',
}

export const SIGNAL_LATENCY = Object.fromEntries(
  SIGNAL_TYPES.map((type) => [type, SIGNAL_LATENCY_OVERRIDES[type] || '~100ms']),
) as Record<SignalType, string>

// ============== Plugin Icons ==============
const PLUGIN_ICON_OVERRIDES: Partial<Record<PluginType, string>> = {
  response_cache: 'RC',
  memory: 'MEM',
  system_prompt: 'SP',
  header_mutation: 'HM',
  hallucination: 'HAL',
  router_replay: 'RR',
  rag: 'RAG',
  fast_response: 'FR',
  request_params: 'RP',
  response_jailbreak: 'RJ',
  tools: 'TL',
  tool_selection: 'TS',
  context_compression: 'CC',
  shadow_dispatch: 'SD',
}

export const PLUGIN_ICONS = Object.fromEntries(
  PLUGIN_TYPES.map((type) => [type, PLUGIN_ICON_OVERRIDES[type] || compactSurfaceIcon(type)]),
) as Record<PluginType, string>

// ============== Plugin Colors (Graphite Theme) ==============
const PLUGIN_COLOR_OVERRIDES: Partial<Record<PluginType, { background: string; border: string }>> =
  {
    response_cache: { background: '#8f949c', border: '#696d74' }, // Graphite Alloy
    memory: { background: '#3f6b73', border: '#2e4f55' },
    system_prompt: { background: '#c9cbd0', border: '#8f949c' }, // Light Alloy
    header_mutation: { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
    hallucination: { background: '#556b7d', border: '#3d4a59' }, // Cool Gray
    router_replay: { background: '#737780', border: '#696d74' }, // Green (consistent with other plugins)
    rag: { background: '#2f855a', border: '#276749' },
    fast_response: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
    request_params: { background: '#805ad5', border: '#6b46c1' },
    response_jailbreak: { background: '#c05621', border: '#9c4221' },
    tools: { background: '#5a6c7d', border: '#3d4a59' },
    tool_selection: { background: '#4b6f7f', border: '#344f5c' },
    context_compression: { background: '#606c7a', border: '#3d4a59' },
    shadow_dispatch: { background: '#5b5f7a', border: '#3f4259' },
  }

export const PLUGIN_COLORS = Object.fromEntries(
  PLUGIN_TYPES.map((type) => [
    type,
    PLUGIN_COLOR_OVERRIDES[type] || { background: '#607D8B', border: '#455A64' },
  ]),
) as Record<PluginType, { background: string; border: string }>

// ============== Algorithm Icons ==============
const ALGORITHM_ICON_OVERRIDES: Partial<Record<AlgorithmType, string>> = {
  confidence: 'CF',
  concurrent: 'CC',
  sequential: 'SEQ',
  ratings: 'RT',
  static: 'ST',
  router_dc: 'RDC',
  automix: 'AM',
  hybrid: 'HY',
  remom: 'RM',
  fusion: 'FU',
  workflows: 'FL',
  latency_aware: 'LAT',
  knn: 'KNN',
  kmeans: 'KM',
  svm: 'SVM',
  mlp: 'MLP',
  multi_factor: 'MF',
}

export const ALGORITHM_ICONS = Object.fromEntries(
  ALGORITHM_DISPLAY_TYPES.map((type) => [
    type,
    ALGORITHM_ICON_OVERRIDES[type] || compactSurfaceIcon(type),
  ]),
) as Record<AlgorithmType, string>

// ============== Algorithm Colors (Graphite Theme) ==============
const ALGORITHM_COLOR_OVERRIDES: Partial<
  Record<AlgorithmType, { background: string; border: string }>
> = {
  confidence: { background: '#8f949c', border: '#696d74' }, // Graphite Alloy
  concurrent: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray
  sequential: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
  ratings: { background: '#c9cbd0', border: '#8f949c' }, // Light Alloy
  static: { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
  router_dc: { background: '#556b7d', border: '#3d4a59' }, // Cool Gray
  remom: { background: '#8f949c', border: '#696d74' }, // Graphite Alloy (same as plugins)
  fusion: { background: '#60784f', border: '#455a36' }, // Muted Alloy
  workflows: { background: '#4f6f78', border: '#38545c' }, // Flow Teal
  automix: { background: '#5d6d7e', border: '#3d4a59' }, // Steel Gray
  hybrid: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
  latency_aware: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray
  knn: { background: '#687887', border: '#4b5968' },
  kmeans: { background: '#596b5d', border: '#3f5144' },
  svm: { background: '#6d687a', border: '#504a5c' },
  mlp: { background: '#586d7a', border: '#40515d' },
  multi_factor: { background: '#4e6f63', border: '#38534a' },
}

export const ALGORITHM_COLORS = Object.fromEntries(
  ALGORITHM_DISPLAY_TYPES.map((type) => [
    type,
    ALGORITHM_COLOR_OVERRIDES[type] || { background: '#607D8B', border: '#455A64' },
  ]),
) as Record<AlgorithmType, { background: string; border: string }>

// ============== Reasoning Effort Display (Graphite Theme) ==============
export const REASONING_EFFORT_DISPLAY: Record<
  string,
  { icon: string; label: string; color: string }
> = {
  low: { icon: 'L', label: 'Low', color: '#c9cbd0' }, // Light Alloy
  medium: { icon: 'M', label: 'Medium', color: '#8f949c' }, // Graphite Alloy
  high: { icon: 'H', label: 'High', color: '#696d74' }, // Dark Alloy
}

export const MODEL_NODE_WIDTH = 220

// ============== Global Plugin Display (Graphite Theme) ==============
export const GLOBAL_PLUGIN_DISPLAY: Record<string, { icon: string; label: string; color: string }> =
  {
    prompt_guard: { icon: 'PG', label: 'Jailbreak Guard', color: '#718096' }, // Medium Gray
    pii_detection: { icon: 'PII', label: 'PII Detection', color: '#5a6c7d' }, // Blue Gray
    response_cache: { icon: 'RC', label: 'Response Cache', color: '#8f949c' }, // Graphite Alloy
  }

// ============== Node Colors (Gray Nodes, Alloy Paths) ==============
export const NODE_COLORS = {
  client: { background: '#8f949c', border: '#696d74' }, // Graphite Alloy (Client stays alloy)
  decision: {
    normal: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
    reasoning: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
    unreachable: { background: '#3d4a59', border: '#2d3748' }, // Very Dark Gray
  },
  model: {
    standard: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray (Models gray)
    reasoning: { background: '#5a6c7d', border: '#3d4a59' }, // Blue Gray (Models gray)
  },
  classification: { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
  disabled: { background: '#3d4a59', border: '#2d3748' }, // Very Dark Gray
}

// ============== Edge Colors (Alloy Paths) ==============
export const EDGE_COLORS = {
  normal: '#8f949c', // Graphite Alloy (All paths use alloy)
  reasoning: '#8f949c', // Graphite Alloy (All paths use alloy)
  active: '#8f949c', // Graphite Alloy
  disabled: '#3d4a59', // Very Dark Gray
  highlighted: '#c9cbd0', // Light Alloy (Highlighted paths brighter)
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
  verticalSpacing: 15, // Minimum space between nodes in same column (reduced from 25)
  groupSpacing: 20, // Extra space between signal groups (reduced from 35)
  // Base heights for different node types (actual height = base + content)
  decisionBaseHeight: 120, // Decision nodes base
  decisionConditionHeight: 22, // Per condition line
  signalGroupBaseHeight: 80, // Signal group base
  signalItemHeight: 20, // Per signal item
  pluginChainBaseHeight: 60, // Plugin chain base
  pluginItemHeight: 20, // Per plugin item
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
