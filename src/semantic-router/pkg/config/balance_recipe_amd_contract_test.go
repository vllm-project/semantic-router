package config

import (
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
	"local/omni",
}

func TestBalanceRecipePreservesAMDLocalAliasContract(t *testing.T) {
	const asset = "config/recipes/balance/config.yaml"

	var manifest CanonicalConfig
	if err := yamlv3.Unmarshal(mustReadRepoFile(t, asset), &manifest); err != nil {
		t.Fatalf("failed to decode %s: %v", asset, err)
	}

	assertBalanceModelContract(t, manifest)
	assertBalanceDecisionContract(t, manifest)
}

func assertBalanceModelContract(t *testing.T, manifest CanonicalConfig) {
	t.Helper()
	modelNames := make([]string, 0, len(manifest.Providers.Models))
	for _, model := range manifest.Providers.Models {
		modelNames = append(modelNames, model.Name)
		if len(model.BackendRefs) != 1 {
			t.Fatalf("model %q must have exactly one backend ref, got %d", model.Name, len(model.BackendRefs))
		}
		connection := model.BackendRefs[0]
		expectedEndpoint := "http://vllm:8000/v1"
		if model.Name == "local/omni" {
			expectedEndpoint = "http://vllm-omni:8000/v1"
		}
		if connection.Provider != "vllm" || connection.Endpoint != expectedEndpoint {
			t.Fatalf("model %q must use the local vLLM integration, got provider=%q endpoint=%q", model.Name, connection.Provider, connection.Endpoint)
		}
		if model.ProviderModelID != model.Name {
			t.Fatalf("model %q must preserve its provider model name, got %q", model.Name, model.ProviderModelID)
		}
	}
	assertBalanceAliasSet(t, "models", modelNames)
}

func assertBalanceDecisionContract(t *testing.T, manifest CanonicalConfig) {
	t.Helper()
	if len(manifest.Recipes) != 1 {
		t.Fatalf("balance manifest must contain one Recipe, got %d", len(manifest.Recipes))
	}
	decisions := manifest.Recipes[0].Routing.Decisions
	if len(decisions) != 15 {
		t.Fatalf("expected one omni lane, 13 calibrated lanes, and one terminal fallback; got %d decisions", len(decisions))
	}
	if len(manifest.Entrypoints) != 1 || manifest.Entrypoints[0].Recipe != manifest.Recipes[0].Name {
		t.Fatalf("balance manifest must contain one common-form Entrypoint: %+v", manifest.Entrypoints)
	}
	assignments := manifest.Entrypoints[0].Assignments
	omni := decisions[0]
	if omni.Name != "omni" || omni.Priority != 300 {
		t.Fatalf("expected the omni lane to remain first, got name=%q priority=%d", omni.Name, omni.Priority)
	}
	omniAssignments := assignments[omni.Name]
	if len(omniAssignments.Models) != 1 || omniAssignments.Models[0].Model != "local/omni" {
		t.Fatalf("expected the omni lane to use local/omni, got %+v", omniAssignments)
	}

	assignedNames := []string{omniAssignments.Models[0].Model}
	for index, decision := range decisions[1:14] {
		if decision.Tier != index+1 {
			t.Fatalf("expected calibrated lane %q to have tier %d, got %d", decision.Name, index+1, decision.Tier)
		}
		if len(decision.Rules.Conditions) == 0 {
			t.Fatalf("expected calibrated lane %q to have explicit matching conditions", decision.Name)
		}
		if index > 0 && decisions[index].Priority <= decision.Priority {
			t.Fatalf("expected calibrated lane priorities to descend, got %d before %d", decisions[index].Priority, decision.Priority)
		}
		set, found := assignments[decision.Name]
		if !found || len(set.Models) < 2 {
			t.Fatalf("expected calibrated lane %q to have at least two Entrypoint assignments, got %+v", decision.Name, set)
		}
		for _, assignment := range set.Models {
			assignedNames = append(assignedNames, assignment.Model)
		}
	}

	terminal := decisions[14]
	if terminal.Name != "casual_chat" || terminal.Tier != 14 || terminal.Priority != 10 {
		t.Fatalf("expected tier-14 casual_chat terminal fallback at priority 10, got name=%q tier=%d priority=%d", terminal.Name, terminal.Tier, terminal.Priority)
	}
	if terminal.Rules.Operator != "AND" || terminal.Rules.Type != "" || terminal.Rules.Name != "" || len(terminal.Rules.Conditions) != 0 {
		t.Fatalf("expected casual_chat to remain an unconditional terminal fallback, got %+v", terminal.Rules)
	}
	terminalAssignments := assignments[terminal.Name]
	if len(terminalAssignments.Models) < 2 || terminalAssignments.Models[0].Model != "qwen/qwen3.5-rocm" {
		t.Fatalf("expected casual_chat to keep local Qwen first and expose a learning candidate, got %+v", terminalAssignments)
	}
	for _, assignment := range terminalAssignments.Models {
		assignedNames = append(assignedNames, assignment.Model)
	}
	assertBalanceAliasSet(t, "entrypoints[].assignments", assignedNames)
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
