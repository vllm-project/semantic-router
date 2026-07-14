package config

import (
	"fmt"
	"slices"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

var balanceAMDLocalAliases = []string{
	"anthropic/claude-opus-4.6",
	"google/gemini-2.5-flash-lite",
	"google/gemini-3.1-pro",
	"openai/gpt5.4",
	"qwen/qwen3.5-rocm",
}

func TestBalanceRecipePreservesAMDLocalAliasContract(t *testing.T) {
	const asset = "deploy/recipes/balance.yaml"

	var recipe CanonicalConfig
	if err := yamlv3.Unmarshal(mustReadRepoFile(t, asset), &recipe); err != nil {
		t.Fatalf("failed to decode %s: %v", asset, err)
	}

	if recipe.Providers.Defaults.DefaultModel != "qwen/qwen3.5-rocm" {
		t.Fatalf("expected AMD recipe default model to stay on the local Qwen alias, got %q", recipe.Providers.Defaults.DefaultModel)
	}
	assertBalanceProviderContract(t, recipe)
	assertBalanceModelCardContract(t, recipe)
	assertBalanceDecisionContract(t, recipe)
}

func assertBalanceProviderContract(t *testing.T, recipe CanonicalConfig) {
	t.Helper()
	const localVLLMEndpoint = "http://vllm:8000/v1"

	providerNames := make([]string, 0, len(recipe.Providers.Models))
	for _, model := range recipe.Providers.Models {
		providerNames = append(providerNames, model.Name)
		if model.ProviderModelID != model.Name {
			t.Fatalf("provider alias %q must preserve its name as provider_model_id, got %q", model.Name, model.ProviderModelID)
		}
		if len(model.BackendRefs) != 1 {
			t.Fatalf("provider alias %q must have exactly one local vLLM backend, got %d", model.Name, len(model.BackendRefs))
		}
		backend := model.BackendRefs[0]
		if got := fmt.Sprintf("%s://%s/v1", backend.Protocol, backend.Endpoint); got != localVLLMEndpoint {
			t.Fatalf("provider alias %q must resolve to %s, got %s", model.Name, localVLLMEndpoint, got)
		}
	}
	if len(providerNames) != len(balanceAMDLocalAliases) {
		t.Fatalf("expected exactly %d provider aliases, got %d", len(balanceAMDLocalAliases), len(providerNames))
	}
	assertBalanceAliasSet(t, "providers.models", providerNames)
}

func assertBalanceModelCardContract(t *testing.T, recipe CanonicalConfig) {
	t.Helper()
	modelCardNames := make([]string, 0, len(recipe.Routing.ModelCards))
	for _, modelCard := range recipe.Routing.ModelCards {
		modelCardNames = append(modelCardNames, modelCard.Name)
	}
	if len(modelCardNames) != len(balanceAMDLocalAliases) {
		t.Fatalf("expected exactly %d routing model cards, got %d", len(balanceAMDLocalAliases), len(modelCardNames))
	}
	assertBalanceAliasSet(t, "routing.modelCards", modelCardNames)
}

func assertBalanceDecisionContract(t *testing.T, recipe CanonicalConfig) {
	t.Helper()
	if len(recipe.Routing.Decisions) != 14 {
		t.Fatalf("expected 14 balance decisions (13 calibrated lanes plus one terminal fallback), got %d", len(recipe.Routing.Decisions))
	}

	decisionModelNames := make([]string, 0, len(recipe.Routing.Decisions))
	calibratedLanes := recipe.Routing.Decisions[:13]
	for index, decision := range calibratedLanes {
		if decision.Tier != index+1 {
			t.Fatalf("expected calibrated lane %q to have tier %d, got %d", decision.Name, index+1, decision.Tier)
		}
		if len(decision.Rules.Conditions) == 0 {
			t.Fatalf("expected calibrated lane %q to have explicit matching conditions", decision.Name)
		}
		if index > 0 && calibratedLanes[index-1].Priority <= decision.Priority {
			t.Fatalf("expected calibrated lane priorities to descend, got %d before %d", calibratedLanes[index-1].Priority, decision.Priority)
		}
		if len(decision.ModelRefs) != 1 {
			t.Fatalf("expected calibrated lane %q to select exactly one local alias, got %d model refs", decision.Name, len(decision.ModelRefs))
		}
		for _, modelRef := range decision.ModelRefs {
			decisionModelNames = append(decisionModelNames, modelRef.Model)
		}
	}

	terminal := recipe.Routing.Decisions[13]
	assertBalanceTerminalDecision(t, terminal)
	for _, modelRef := range terminal.ModelRefs {
		decisionModelNames = append(decisionModelNames, modelRef.Model)
	}
	assertBalanceAliasSet(t, "routing.decisions[].modelRefs", decisionModelNames)
}

func assertBalanceTerminalDecision(t *testing.T, terminal Decision) {
	t.Helper()
	if terminal.Name != "casual_chat" || terminal.Tier != 14 || terminal.Priority != 10 {
		t.Fatalf("expected tier-14 casual_chat terminal fallback at priority 10, got name=%q tier=%d priority=%d", terminal.Name, terminal.Tier, terminal.Priority)
	}
	if terminal.Rules.Operator != "AND" || terminal.Rules.Type != "" || terminal.Rules.Name != "" || len(terminal.Rules.Conditions) != 0 {
		t.Fatalf("expected casual_chat to remain an unconditional terminal fallback, got %+v", terminal.Rules)
	}
	if len(terminal.ModelRefs) != 1 || terminal.ModelRefs[0].Model != "qwen/qwen3.5-rocm" {
		t.Fatalf("expected casual_chat to select only the local Qwen alias, got %+v", terminal.ModelRefs)
	}
}

func assertBalanceAliasSet(t *testing.T, surface string, got []string) {
	t.Helper()
	want := slices.Clone(balanceAMDLocalAliases)
	got = slices.Clone(got)
	slices.Sort(want)
	slices.Sort(got)
	got = slices.Compact(got)
	if !slices.Equal(got, want) {
		t.Fatalf("%s alias set mismatch\nwant: %v\ngot:  %v", surface, want, got)
	}
}
