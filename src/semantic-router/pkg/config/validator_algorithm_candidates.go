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

import (
	"fmt"
	"strings"
)

func validateDecisionMinimumCandidates(
	decisionName string,
	modelRefs []ModelRef,
	algorithm *AlgorithmConfig,
) error {
	minimum := algorithm.MinimumCandidates
	if minimum < 0 {
		return fmt.Errorf(
			"decision '%s': algorithm.minimum_candidates must be >= 1 when set",
			decisionName,
		)
	}
	// Built-in Recipe assets are intentionally model-free. Enforce the
	// materialization contract as soon as an Entrypoint assigns candidates.
	if minimum == 0 || len(modelRefs) == 0 {
		return nil
	}
	configured := uniqueDecisionModelRefCount(modelRefs)
	if configured < minimum {
		return fmt.Errorf(
			"decision '%s': algorithm.minimum_candidates=%d requires at least %d unique modelRefs, got %d",
			decisionName,
			minimum,
			minimum,
			configured,
		)
	}
	return nil
}

func validateDecisionFusionAlgorithm(
	decisionName string,
	modelRefs []ModelRef,
	cfg *FusionAlgorithmConfig,
) error {
	if err := ValidateFusionAlgorithmConfig(cfg); err != nil {
		return wrapAlgorithmValidationError(decisionName, "fusion", err)
	}
	if cfg == nil || cfg.MinSuccessfulResponses == 0 {
		return nil
	}
	panelSize := uniqueDecisionModelRefCount(modelRefs)
	if len(cfg.AnalysisModels) > 0 {
		panelSize = uniqueNonEmptyStringCount(cfg.AnalysisModels)
	}
	if panelSize > 0 && cfg.MinSuccessfulResponses > panelSize {
		return wrapAlgorithmValidationError(
			decisionName,
			"fusion",
			fmt.Errorf(
				"min_successful_responses=%d exceeds the configured panel size %d",
				cfg.MinSuccessfulResponses,
				panelSize,
			),
		)
	}
	return nil
}

func validateDecisionWorkflowsAlgorithm(
	decisionName string,
	modelRefs []ModelRef,
	cfg *WorkflowsAlgorithmConfig,
) error {
	if err := ValidateWorkflowsAlgorithmConfig(cfg); err != nil {
		return wrapAlgorithmValidationError(decisionName, "workflows", err)
	}
	if cfg == nil || workflowMode(cfg.Mode) != WorkflowModeDynamic ||
		cfg.MinSuccessfulResponses == 0 || len(modelRefs) == 0 {
		return nil
	}
	workerCount := uniqueDecisionModelRefCount(modelRefs)
	if cfg.MinSuccessfulResponses > workerCount {
		return wrapAlgorithmValidationError(
			decisionName,
			"workflows",
			fmt.Errorf(
				"min_successful_responses=%d exceeds the configured worker pool size %d",
				cfg.MinSuccessfulResponses,
				workerCount,
			),
		)
	}
	return nil
}

func uniqueDecisionModelRefCount(modelRefs []ModelRef) int {
	seen := make(map[string]struct{}, len(modelRefs))
	for _, ref := range modelRefs {
		model := strings.TrimSpace(ref.Model)
		lora := strings.TrimSpace(ref.LoRAName)
		if model == "" {
			continue
		}
		seen[model+"\x00"+lora] = struct{}{}
	}
	return len(seen)
}

func uniqueNonEmptyStringCount(values []string) int {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized != "" {
			seen[normalized] = struct{}{}
		}
	}
	return len(seen)
}
