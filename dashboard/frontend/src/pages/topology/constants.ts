// topology/constants.ts - Constants and Color Schemes

import { SignalType, PluginType, AlgorithmType } from './types'

// ============== Signal Icons ==============
export const SIGNAL_ICONS: Record<SignalType, string> = {
  keyword: '🔑',
  embedding: '📐',
  domain: '🎯',
  fact_check: '✓',
  user_feedback: '💬',
  preference: '⚙️',
  language: '🌐',
  latency: '⚡',
  context: '📏',
  complexity: '🧠',
}

// ============== Signal Colors (Gray Nodes, Green Paths) ==============
export const SIGNAL_COLORS: Record<SignalType, { background: string; border: string }> = {
  keyword: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
  embedding: { background: '#4a5568', border: '#2d3748' },    // Dark Gray
  domain: { background: '#4a5568', border: '#2d3748' },       // Dark Gray
  fact_check: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  user_feedback: { background: '#4a5568', border: '#2d3748' }, // Dark Gray
  preference: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
  language: { background: '#4a5568', border: '#2d3748' },     // Dark Gray
  latency: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
  context: { background: '#4a5568', border: '#2d3748' },      // Dark Gray
  complexity: { background: '#4a5568', border: '#2d3748' },   // Dark Gray
}

// ============== Signal Latency ==============
export const SIGNAL_LATENCY: Record<SignalType, string> = {
  keyword: '<1ms',
  embedding: '10-50ms',
  domain: '10-50ms',
  fact_check: '10-50ms',
  user_feedback: '10-50ms',
  preference: '200-500ms',
  language: '<1ms',
  latency: '<1ms',
  context: '<1ms',
  complexity: '50-100ms',
}

// ============== Plugin Icons ==============
export const PLUGIN_ICONS: Record<PluginType, string> = {
  'semantic-cache': '⚡',
  'jailbreak': '🛡️',
  'pii': '🔒',
  'system_prompt': '📝',
  'header_mutation': '🔧',
  'hallucination': '🔍',
  'router_replay': '🔄',
}

// ============== Plugin Colors (NVIDIA Dark Theme) ==============
export const PLUGIN_COLORS: Record<PluginType, { background: string; border: string }> = {
  'semantic-cache': { background: '#76b900', border: '#5a8f00' },  // NVIDIA Green
  'jailbreak': { background: '#718096', border: '#4a5568' },       // Medium Gray
  'pii': { background: '#5a6c7d', border: '#3d4a59' },             // Blue Gray
  'system_prompt': { background: '#8fd400', border: '#76b900' },   // Light Green
  'header_mutation': { background: '#606c7a', border: '#3d4a59' }, // Slate Gray
  'hallucination': { background: '#556b7d', border: '#3d4a59' },   // Cool Gray
  'router_replay': { background: '#6ba300', border: '#5a8f00' },   // Green (consistent with other plugins)
}

// ============== Algorithm Icons ==============
export const ALGORITHM_ICONS: Record<AlgorithmType, string> = {
  confidence: '📈',
  concurrent: '⚡',
  sequential: '➡️',
  ratings: '⭐',
  static: '📌',
  elo: '🏆',
  router_dc: '🔀',
  automix: '🤖',
  hybrid: '🔄',
  remom: '🧠',  // Brain icon for reasoning/thinking
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
}

// ============== Reasoning Effort Display (NVIDIA Dark Theme) ==============
export const REASONING_EFFORT_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'low': { icon: '🔋', label: 'Low', color: '#8fd400' },       // Light Green
  'medium': { icon: '⚡', label: 'Medium', color: '#76b900' },  // NVIDIA Green
  'high': { icon: '🔥', label: 'High', color: '#5a8f00' },      // Dark Green
}

// ============== Global Plugin Display (NVIDIA Dark Theme) ==============
export const GLOBAL_PLUGIN_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'prompt_guard': { icon: '🛡️', label: 'Jailbreak Guard', color: '#718096' },   // Medium Gray
  'pii_detection': { icon: '🔒', label: 'PII Detection', color: '#5a6c7d' },     // Blue Gray
  'semantic_cache': { icon: '⚡', label: 'Semantic Cache', color: '#76b900' },   // NVIDIA Green
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

// ============== Signal Types Array ==============
export const SIGNAL_TYPES: SignalType[] = [
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'preference',
  'language',
  'latency',
  'context',
  'complexity',
]

// ============== Plugin Types Array ==============
export const PLUGIN_TYPES: PluginType[] = [
  'semantic-cache',
  'jailbreak',
  'pii',
  'system_prompt',
  'header_mutation',
  'hallucination',
  'router_replay',
]
