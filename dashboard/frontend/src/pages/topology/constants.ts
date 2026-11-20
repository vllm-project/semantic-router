// topology/constants.ts - Catalog constants the topology parsers read

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

// ============== Short Signal Codes ==============
const SIGNAL_ICON_OVERRIDES: Partial<Record<SignalType, string>> = {
  keyword: 'KW',
  domain: 'DOM',
  fact_check: 'FC',
  user_feedback: 'UF',
  language: 'LANG',
  context: 'CTX',
  complexity: 'CPX',
  modality: 'MOD',
  authz: 'AUTH',
  jailbreak: 'JB',
  hallucination: 'HL',
  pii: 'PII',
  conversation: 'CONV',
  projection: 'PRJ',
}

export const SIGNAL_ICONS = Object.fromEntries(
  SIGNAL_TYPES.map((type) => [type, SIGNAL_ICON_OVERRIDES[type] || compactSurfaceIcon(type)]),
) as Record<SignalType, string>

// ============== Signal Latency ==============
const SIGNAL_LATENCY_OVERRIDES: Partial<Record<SignalType, string>> = {
  keyword: '<1ms',
  domain: '10-50ms',
  fact_check: '10-50ms',
  user_feedback: '10-50ms',
  language: '<1ms',
  context: '<1ms',
  complexity: '50-100ms',
  modality: '50-100ms',
  authz: '<1ms',
  jailbreak: '10-50ms',
  hallucination: '200-500ms',
  pii: '10-50ms',
  conversation: '<1ms',
  projection: '<1ms',
}

export const SIGNAL_LATENCY = Object.fromEntries(
  SIGNAL_TYPES.map((type) => [type, SIGNAL_LATENCY_OVERRIDES[type] || '~100ms']),
) as Record<SignalType, string>

// ============== Short Plugin Codes ==============
const PLUGIN_ICON_OVERRIDES: Partial<Record<PluginType, string>> = {
  response_cache: 'RC',
  context: 'CTX',
  system_prompt: 'SP',
  header_mutation: 'HM',
  router_replay: 'RR',
  fast_response: 'FR',
  request_params: 'RP',
  response_guard: 'RG',
  tools: 'TL',
  prompt_cache: 'PC',
  masking: 'MSK',
}

export const PLUGIN_ICONS = Object.fromEntries(
  PLUGIN_TYPES.map((type) => [type, PLUGIN_ICON_OVERRIDES[type] || compactSurfaceIcon(type)]),
) as Record<PluginType, string>

// ============== Short Algorithm Codes ==============
const ALGORITHM_ICON_OVERRIDES: Partial<Record<AlgorithmType, string>> = {
  escalate: 'ES',
  ratings: 'RT',
  static: 'ST',
  remom: 'RM',
  fusion: 'FU',
  workflows: 'FL',
  multi_factor: 'MF',
}

export const ALGORITHM_ICONS = Object.fromEntries(
  ALGORITHM_TYPES.map((type) => [type, ALGORITHM_ICON_OVERRIDES[type] || compactSurfaceIcon(type)]),
) as Record<AlgorithmType, string>
