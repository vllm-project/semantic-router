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

package k8s

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type crdValidationContext struct {
	models     map[string]*v1alpha1.ModelConfig
	keywords   map[string]struct{}
	embeddings map[string]struct{}
	domains    map[string]struct{}
}

func validatePoolRoute(
	pool *v1alpha1.IntelligentPool,
	route *v1alpha1.IntelligentRoute,
	reasoningFamilies map[string]config.ReasoningFamilyConfig,
) error {
	ctx, err := buildCRDValidationContext(pool, route)
	if err != nil {
		return err
	}
	if err := validateDecisionReferences(route.Spec.Decisions, ctx); err != nil {
		return err
	}
	return validateReasoningFamilyReferences(pool.Spec.Models, reasoningFamilies)
}

func buildCRDValidationContext(
	pool *v1alpha1.IntelligentPool,
	route *v1alpha1.IntelligentRoute,
) (*crdValidationContext, error) {
	keywords, err := collectUniqueNames(route.Spec.Signals.Keywords, "keyword signal", func(signal v1alpha1.KeywordSignal) string {
		return signal.Name
	})
	if err != nil {
		return nil, err
	}
	embeddings, err := collectUniqueNames(route.Spec.Signals.Embeddings, "embedding signal", func(signal v1alpha1.EmbeddingSignal) string {
		return signal.Name
	})
	if err != nil {
		return nil, err
	}
	domains, err := collectUniqueNames(route.Spec.Signals.Domains, "domain signal", func(signal v1alpha1.DomainSignal) string {
		return signal.Name
	})
	if err != nil {
		return nil, err
	}

	return &crdValidationContext{
		models:     buildValidationModelMap(pool.Spec.Models),
		keywords:   keywords,
		embeddings: embeddings,
		domains:    domains,
	}, nil
}

func buildValidationModelMap(models []v1alpha1.ModelConfig) map[string]*v1alpha1.ModelConfig {
	result := make(map[string]*v1alpha1.ModelConfig, len(models))
	for i := range models {
		model := &models[i]
		result[model.Name] = model
	}
	return result
}

func collectUniqueNames[T any](
	items []T,
	label string,
	nameFn func(T) string,
) (map[string]struct{}, error) {
	result := make(map[string]struct{}, len(items))
	for _, item := range items {
		name := nameFn(item)
		if _, exists := result[name]; exists {
			return nil, fmt.Errorf("duplicate %s name: %s", label, name)
		}
		result[name] = struct{}{}
	}
	return result, nil
}

func validateDecisionReferences(decisions []v1alpha1.Decision, ctx *crdValidationContext) error {
	for _, decision := range decisions {
		if err := validateDecisionSignalReferences(decision, ctx); err != nil {
			return err
		}
		if err := validateDecisionModelReferences(decision, ctx.models); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionSignalReferences(decision v1alpha1.Decision, ctx *crdValidationContext) error {
	for _, condition := range decision.Signals.Conditions {
		if err := validateSignalReference(decision.Name, condition, ctx); err != nil {
			return err
		}
	}
	return nil
}

func validateSignalReference(
	decisionName string,
	condition v1alpha1.SignalCondition,
	ctx *crdValidationContext,
) error {
	var known map[string]struct{}
	switch condition.Type {
	case "keyword":
		known = ctx.keywords
	case "embedding":
		known = ctx.embeddings
	case "domain":
		known = ctx.domains
	default:
		return nil
	}
	if _, ok := known[condition.Name]; ok {
		return nil
	}
	return fmt.Errorf("decision %s references unknown %s signal: %s", decisionName, condition.Type, condition.Name)
}

func validateDecisionModelReferences(
	decision v1alpha1.Decision,
	models map[string]*v1alpha1.ModelConfig,
) error {
	for _, modelRef := range decision.ModelRefs {
		model, ok := models[modelRef.Model]
		if !ok {
			return fmt.Errorf("decision %s references unknown model: %s", decision.Name, modelRef.Model)
		}
		if err := validateLoRAReference(decision.Name, modelRef, model); err != nil {
			return err
		}
	}
	return nil
}

func validateLoRAReference(
	decisionName string,
	modelRef v1alpha1.ModelRef,
	model *v1alpha1.ModelConfig,
) error {
	if modelRef.LoRAName == "" {
		return nil
	}
	for _, lora := range model.LoRAs {
		if lora.Name == modelRef.LoRAName {
			return nil
		}
	}
	return fmt.Errorf("decision %s references unknown LoRA %s for model %s", decisionName, modelRef.LoRAName, modelRef.Model)
}

func validateReasoningFamilyReferences(
	models []v1alpha1.ModelConfig,
	reasoningFamilies map[string]config.ReasoningFamilyConfig,
) error {
	if len(reasoningFamilies) == 0 {
		return nil
	}
	for _, model := range models {
		if model.ReasoningFamily == "" {
			continue
		}
		if _, ok := reasoningFamilies[model.ReasoningFamily]; ok {
			continue
		}
		return fmt.Errorf("model %s references unknown reasoning family: %s", model.Name, model.ReasoningFamily)
	}
	return nil
}
