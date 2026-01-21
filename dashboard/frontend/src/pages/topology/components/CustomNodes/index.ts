// CustomNodes/index.ts - Export all custom node components

import { ClientNode } from './ClientNode'
import { GlobalPluginNode } from './GlobalPluginNode'
import { SignalGroupNode } from './SignalGroupNode'
import { DecisionNode } from './DecisionNode'
import { AlgorithmNode } from './AlgorithmNode'
import { PluginChainNode } from './PluginChainNode'
import { ModelNode } from './ModelNode'
import { DefaultRouteNode } from './DefaultRouteNode'
import { FallbackDecisionNode } from './FallbackDecisionNode'

export {
  ClientNode,
  GlobalPluginNode,
  SignalGroupNode,
  DecisionNode,
  AlgorithmNode,
  PluginChainNode,
  ModelNode,
  DefaultRouteNode,
  FallbackDecisionNode,
}

// Node types registry for ReactFlow
export const customNodeTypes = {
  clientNode: ClientNode,
  globalPluginNode: GlobalPluginNode,
  signalGroupNode: SignalGroupNode,
  decisionNode: DecisionNode,
  algorithmNode: AlgorithmNode,
  pluginChainNode: PluginChainNode,
  modelNode: ModelNode,
  defaultRouteNode: DefaultRouteNode,
  fallbackDecisionNode: FallbackDecisionNode,
}
