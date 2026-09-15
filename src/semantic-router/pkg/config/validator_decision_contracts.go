/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

// Per-decision contract validation. The checks are grouped by the question they
// answer so that adding one does not push the dispatcher over the complexity
// gate, and so a reader can tell which group a new rule belongs in.

func validateDecisionModelContracts(cfg *RouterConfig) error {
	for _, decision := range cfg.AllRoutingDecisions() {
		if err := validateDecisionReferences(cfg, decision); err != nil {
			return err
		}
		if err := validateDecisionExecution(cfg, decision); err != nil {
			return err
		}
	}
	return nil
}

// validateDecisionReferences checks what a decision points at: its rule tree,
// annotations, and the models and action it names.
func validateDecisionReferences(cfg *RouterConfig, decision Decision) error {
	if err := validateDecisionRuleNode(cfg, decision.Name, &decision.Rules, true); err != nil {
		return err
	}
	warnUnguardedClassifierConditions(decision)
	if err := validateDecisionAnnotations(decision); err != nil {
		return err
	}
	if err := validateDecisionModelRefs(cfg, decision); err != nil {
		return err
	}
	return validateDecisionAction(cfg, decision)
}

// validateDecisionExecution checks how a decision runs: its algorithm
// configuration and the model contracts that configuration depends on.
func validateDecisionExecution(cfg *RouterConfig, decision Decision) error {
	if err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm); err != nil {
		return err
	}
	if err := validateDecisionPromptModel(cfg, decision); err != nil {
		return err
	}
	if err := validateDecisionFusionFallbackTarget(cfg, decision); err != nil {
		return err
	}
	if err := validateDecisionWorkflowModelRefs(decision); err != nil {
		return err
	}
	if err := validateDecisionCandidateIterations(decision); err != nil {
		return err
	}
	return validateDecisionOutputContractSpec(decision)
}
