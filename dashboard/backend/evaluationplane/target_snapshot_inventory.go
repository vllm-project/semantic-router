package evaluationplane

import (
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type mixtureModelBinding struct {
	EffectiveModel string
	BaseModel      string
}

type mixtureModelInventory struct {
	poolModels     map[string]mixtureModelBinding
	decisionModels [][]string
	supportModels  map[string]mixtureModelBinding
	fallbackModel  string
	valid          bool
}

func collectMixtureModelInventory(
	canonical routerconfig.CanonicalConfig,
	recipe *routerconfig.RoutingRecipe,
) mixtureModelInventory {
	baseByEffective, bindingsValid := effectiveModelBaseIndex(canonical.Routing.ModelCards)
	inventory := mixtureModelInventory{
		poolModels:     make(map[string]mixtureModelBinding),
		decisionModels: make([][]string, len(recipe.Profile.Decisions)),
		supportModels:  make(map[string]mixtureModelBinding),
		valid:          bindingsValid,
	}
	for index, decision := range recipe.Profile.Decisions {
		for _, ref := range decision.ModelRefs {
			if !addMixtureModel(inventory.poolModels, &inventory.decisionModels[index], ref, baseByEffective) {
				inventory.valid = false
			}
		}
		for _, iteration := range decision.CandidateIterations {
			if strings.TrimSpace(iteration.Source) != "models" {
				continue
			}
			for _, ref := range iteration.Models {
				if !addMixtureModel(inventory.poolModels, &inventory.decisionModels[index], ref, baseByEffective) {
					inventory.valid = false
				}
			}
		}
	}
	addMixtureFallback(canonical, &inventory, baseByEffective)
	addMixturePromptSupport(recipe, &inventory, baseByEffective)
	return inventory
}

func addMixtureFallback(
	canonical routerconfig.CanonicalConfig,
	inventory *mixtureModelInventory,
	baseByEffective map[string]string,
) {
	fallback := strings.TrimSpace(canonical.Providers.Defaults.DefaultModel)
	if fallback == "" {
		return
	}
	binding, valid := modelBindingForName(fallback, baseByEffective)
	if !valid || !addModelBinding(inventory.poolModels, binding) {
		inventory.valid = false
		return
	}
	inventory.fallbackModel = binding.EffectiveModel
	for index := range inventory.decisionModels {
		if len(inventory.decisionModels[index]) == 0 {
			inventory.decisionModels[index] = []string{inventory.fallbackModel}
		}
	}
}

func addMixturePromptSupport(
	recipe *routerconfig.RoutingRecipe,
	inventory *mixtureModelInventory,
	baseByEffective map[string]string,
) {
	for _, decision := range recipe.Profile.Decisions {
		if decision.Algorithm == nil || decision.Algorithm.Prompt == nil {
			continue
		}
		binding, valid := modelBindingForName(decision.Algorithm.Prompt.Model, baseByEffective)
		if !valid {
			inventory.valid = false
			continue
		}
		if _, candidate := inventory.poolModels[binding.EffectiveModel]; !candidate &&
			!addModelBinding(inventory.supportModels, binding) {
			inventory.valid = false
		}
	}
}

func effectiveModelBaseIndex(cards []routerconfig.RoutingModel) (map[string]string, bool) {
	result := make(map[string]string, len(cards))
	adapterBases := make(map[string]string)
	valid := true
	for _, card := range cards {
		name := strings.TrimSpace(card.Name)
		if name == "" {
			valid = false
			continue
		}
		result[name] = name
	}
	for _, card := range cards {
		base := strings.TrimSpace(card.Name)
		for _, adapter := range card.LoRAs {
			effective := strings.TrimSpace(adapter.Name)
			if effective == "" || base == "" {
				valid = false
				continue
			}
			if prior, exists := adapterBases[effective]; exists && prior != base {
				valid = false
				continue
			}
			adapterBases[effective] = base
			result[effective] = base
		}
	}
	return result, valid
}

func modelBindingForName(
	raw string,
	baseByEffective map[string]string,
) (mixtureModelBinding, bool) {
	effective := strings.TrimSpace(raw)
	base, exists := baseByEffective[effective]
	if !exists {
		return mixtureModelBinding{}, false
	}
	return mixtureModelBinding{EffectiveModel: effective, BaseModel: base}, true
}

func addMixtureModel(
	pool map[string]mixtureModelBinding,
	decisionModels *[]string,
	ref routerconfig.ModelRef,
	baseByEffective map[string]string,
) bool {
	effective := strings.TrimSpace(ref.Model)
	if adapter := strings.TrimSpace(ref.LoRAName); adapter != "" {
		effective = adapter
	}
	binding, valid := modelBindingForName(effective, baseByEffective)
	if !valid || (strings.TrimSpace(ref.LoRAName) != "" && binding.BaseModel != strings.TrimSpace(ref.Model)) {
		return false
	}
	if !addModelBinding(pool, binding) {
		return false
	}
	if !containsString(*decisionModels, binding.EffectiveModel) {
		*decisionModels = append(*decisionModels, binding.EffectiveModel)
	}
	return true
}

func addModelBinding(pool map[string]mixtureModelBinding, binding mixtureModelBinding) bool {
	prior, exists := pool[binding.EffectiveModel]
	if exists && prior != binding {
		return false
	}
	pool[binding.EffectiveModel] = binding
	return true
}

func baseModelsForBindings(bindings map[string]mixtureModelBinding) map[string]struct{} {
	result := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		result[binding.BaseModel] = struct{}{}
	}
	return result
}
