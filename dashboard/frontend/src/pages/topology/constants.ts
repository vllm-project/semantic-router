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
}

// ============== Signal Colors ==============
export const SIGNAL_COLORS: Record<SignalType, { background: string; border: string }> = {
  keyword: { background: '#4CAF50', border: '#388E3C' },
  embedding: { background: '#2196F3', border: '#1976D2' },
  domain: { background: '#9C27B0', border: '#7B1FA2' },
  fact_check: { background: '#FF9800', border: '#F57C00' },
  user_feedback: { background: '#E91E63', border: '#C2185B' },
  preference: { background: '#00BCD4', border: '#0097A7' },
  language: { background: '#795548', border: '#5D4037' },
}

// ============== Signal Latency ==============
export const SIGNAL_LATENCY: Record<SignalType, string> = {
  keyword: '<1ms',
  embedding: '10-50ms',
  domain: '50-100ms',
  fact_check: '50-100ms',
  user_feedback: '50-100ms',
  preference: '200-500ms',
  language: '<1ms',
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

// ============== Plugin Colors ==============
export const PLUGIN_COLORS: Record<PluginType, { background: string; border: string }> = {
  'semantic-cache': { background: '#00BCD4', border: '#0097A7' },
  'jailbreak': { background: '#FF9800', border: '#F57C00' },
  'pii': { background: '#9C27B0', border: '#7B1FA2' },
  'system_prompt': { background: '#4CAF50', border: '#388E3C' },
  'header_mutation': { background: '#795548', border: '#5D4037' },
  'hallucination': { background: '#F44336', border: '#D32F2F' },
  'router_replay': { background: '#607D8B', border: '#455A64' },
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
}

// ============== Algorithm Colors ==============
export const ALGORITHM_COLORS: Record<AlgorithmType, { background: string; border: string }> = {
  confidence: { background: '#FF5722', border: '#E64A19' },
  concurrent: { background: '#673AB7', border: '#512DA8' },
  sequential: { background: '#3F51B5', border: '#303F9F' },
  ratings: { background: '#009688', border: '#00796B' },
  static: { background: '#607D8B', border: '#455A64' },
  elo: { background: '#FFC107', border: '#FFA000' },
  router_dc: { background: '#E91E63', border: '#C2185B' },
  automix: { background: '#00BCD4', border: '#0097A7' },
  hybrid: { background: '#9C27B0', border: '#7B1FA2' },
}

// ============== Reasoning Effort Display ==============
export const REASONING_EFFORT_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'low': { icon: '🔋', label: 'Low', color: '#4CAF50' },
  'medium': { icon: '⚡', label: 'Medium', color: '#FF9800' },
  'high': { icon: '🔥', label: 'High', color: '#F44336' },
}

// ============== Global Plugin Display ==============
export const GLOBAL_PLUGIN_DISPLAY: Record<string, { icon: string; label: string; color: string }> = {
  'prompt_guard': { icon: '🛡️', label: 'Jailbreak Guard', color: '#FF9800' },
  'pii_detection': { icon: '🔒', label: 'PII Detection', color: '#9C27B0' },
  'semantic_cache': { icon: '⚡', label: 'Semantic Cache', color: '#00BCD4' },
}

// ============== Node Colors ==============
export const NODE_COLORS = {
  client: { background: '#4CAF50', border: '#45a049' },
  decision: {
    normal: { background: '#3F51B5', border: '#303F9F' },
    reasoning: { background: '#E91E63', border: '#C2185B' },
    unreachable: { background: '#795548', border: '#5D4037' },  // Brown/muted for unreachable
  },
  model: {
    standard: { background: '#607D8B', border: '#455A64' },
    reasoning: { background: '#E91E63', border: '#C2185B' },
  },
  classification: { background: '#673AB7', border: '#512DA8' },
  disabled: { background: '#757575', border: '#616161' },
}

// ============== Edge Colors ==============
export const EDGE_COLORS = {
  normal: '#999999',
  reasoning: '#E91E63',
  active: '#4CAF50',
  disabled: '#CCCCCC',
  highlighted: '#76b900',
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
  verticalSpacing: 25,   // Minimum space between nodes in same column
  groupSpacing: 35,      // Extra space between signal groups
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
