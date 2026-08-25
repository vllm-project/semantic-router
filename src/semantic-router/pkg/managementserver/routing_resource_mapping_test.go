package managementserver

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRoutingEntrypointInputPreservesPriorityFallback(t *testing.T) {
	input := routingEntrypointInput(managementapi.RoutingEntrypointWrite{
		Name: "Blend", Aliases: []string{"vllm-sr/blend"},
		Rules: []managementapi.RoutingEntrypointRuleWrite{{
			Name: "Default", RecipeID: "recipe_one",
			Assignments: map[string]managementapi.RoutingAssignmentSetWrite{
				"decision_one": {
					Models: []managementapi.RoutingAssignmentWrite{
						{ModelID: "model_primary", Priority: 0},
						{ModelID: "model_backup", Priority: 1},
					},
					Fallback: &managementapi.RoutingFallbackPolicy{Strategy: "priority", On: []string{"timeout"}},
				},
			},
		}},
	})
	set := input.Rules[0].Assignments["decision_one"]
	if len(set.Models) != 2 || set.Models[1].Priority != 1 || set.Fallback == nil || set.Fallback.On[0] != "timeout" {
		t.Fatalf("mapped assignment = %#v", set)
	}
}

func TestRoutingModelViewRedactsBackendExecutionInternals(t *testing.T) {
	model := routingmanagement.Model{
		ResourceIdentity: routingmanagement.ResourceIdentity{
			ID: "model_safe", Name: "Safe", Status: routingmanagement.StatusDraft,
			Revision: 2, CreatedAt: time.Unix(1, 0), UpdatedAt: time.Unix(2, 0),
		},
		Current: routingsnapshot.Model{
			ID: "model_safe", Revision: 3,
			CatalogRevision: "sha256:" + strings.Repeat("a", 64),
			Execution:       routingsnapshot.ModelExecution{RequestTimeout: "5s", StreamTimeout: "10s"},
			Backends: []routingsnapshot.Backend{{
				ProviderID: "provider", ProviderModelID: "upstream/model",
				WireFormat: "private.adapter.v1", Origin: "https://private.invalid",
				ProviderCredentialID: "11111111-1111-4111-8111-111111111111",
				Connection: routingsnapshot.BackendConnection{
					Path: "/private", Headers: map[string]string{"X-Internal": "value"},
				},
				Weight: "1",
			}},
		},
	}
	wire, err := json.Marshal(routingModelViewDTO(model))
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{
		"private.adapter.v1", "https://private.invalid", "11111111-1111-4111-8111-111111111111",
		"/private", "X-Internal", "providerCredentialId", "credentialId", "origin", "connection",
	} {
		if strings.Contains(string(wire), forbidden) {
			t.Errorf("safe Model view leaked %q: %s", forbidden, wire)
		}
	}
	for _, required := range []string{"provider", "upstream/model", "credentialConfigured"} {
		if !strings.Contains(string(wire), required) {
			t.Errorf("safe Model view omitted %q: %s", required, wire)
		}
	}
}

func TestRoutingModelCardViewContainsOnlySemanticAuthoringData(t *testing.T) {
	model := routingmanagement.Model{
		ResourceIdentity: routingmanagement.ResourceIdentity{
			ID: "model_safe", Name: "Safe", Status: routingmanagement.StatusActive,
			Revision: 2, CreatedAt: time.Unix(1, 0), UpdatedAt: time.Unix(2, 0),
		},
		Current: routingsnapshot.Model{
			ID: "model_safe", Revision: 3, CatalogRevision: "sha256:" + strings.Repeat("a", 64),
			Aliases: []string{"safe-alias"}, ParamSize: "32b", ContextWindowSize: 131072,
			Description: "Semantic metadata", Capabilities: []string{"reasoning"},
			Reasoning: routingsnapshot.ReasoningFamily{Type: "reasoning", Efforts: []string{"high"}},
			LoRAs:     []string{"adapter"}, QualityScore: 0.9, Modality: "text", Tags: []string{"balanced"},
			Execution: routingsnapshot.ModelExecution{RequestTimeout: "5s", StreamTimeout: "10s"},
			Backends: []routingsnapshot.Backend{{
				ProviderID: "private-provider", ProviderModelID: "private/model",
				ProviderCredentialID: "11111111-1111-4111-8111-111111111111",
			}},
		},
	}
	wire, err := json.Marshal(routingModelCardViewDTO(model))
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		`"id":"model_safe"`, `"name":"Safe"`, `"card"`, `"aliases":["safe-alias"]`,
		`"paramSize":"32b"`, `"contextWindowSize":131072`, `"capabilities":["reasoning"]`,
		`"qualityScore":0.9`, `"modality":"text"`,
	} {
		if !strings.Contains(string(wire), required) {
			t.Errorf("Model Card view omitted %q: %s", required, wire)
		}
	}
	for _, forbidden := range []string{
		"status", "revision", "catalogRevision", "control", "execution", "pricing", "backends",
		"private-provider", "private/model", "11111111-1111-4111-8111-111111111111",
		"createdAt", "updatedAt",
	} {
		if strings.Contains(string(wire), forbidden) {
			t.Errorf("Model Card view leaked %q: %s", forbidden, wire)
		}
	}
}

func TestRoutingEntrypointListOmitsTopology(t *testing.T) {
	entrypoint := routingmanagement.Entrypoint{
		ResourceIdentity: routingmanagement.ResourceIdentity{
			ID: "entrypoint_safe", Name: "Safe", Status: routingmanagement.StatusActive,
			Revision: 2, CreatedAt: time.Unix(1, 0), UpdatedAt: time.Unix(2, 0),
		},
		Current: routingsnapshot.Entrypoint{
			ID: "entrypoint_safe", Revision: 3, Aliases: []string{"safe"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule_safe", Name: "Safe", RecipeID: "recipe_safe", RecipeRevision: 4,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision_safe": {
						Models: []routingsnapshot.Assignment{
							{ModelID: "model_safe", ModelRevision: 5, Priority: 0, Weight: "1"},
							{ModelID: "model_backup", ModelRevision: 2, Priority: 1, Weight: "1"},
						},
						Fallback: &routingsnapshot.FallbackPolicy{Strategy: "priority", On: []string{"unavailable"}},
					},
				},
			}},
		},
		RuleCount: 1, AssignedModelCount: 2,
	}
	listWire, err := json.Marshal(routingEntrypointViewDTO(entrypoint, false))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(listWire), "rules") || strings.Contains(string(listWire), "model_safe") {
		t.Fatalf("Entrypoint identity view leaked topology: %s", listWire)
	}
	if !strings.Contains(string(listWire), `"ruleCount":1`) ||
		!strings.Contains(string(listWire), `"assignedModelCount":2`) {
		t.Fatalf("Entrypoint list omitted its bounded summary: %s", listWire)
	}
	detailWire, err := json.Marshal(routingEntrypointViewDTO(entrypoint, true))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(detailWire), "model_safe") || !strings.Contains(string(detailWire), "model_backup") ||
		!strings.Contains(string(detailWire), `"priority":1`) || !strings.Contains(string(detailWire), `"strategy":"priority"`) ||
		!strings.Contains(string(detailWire), "recipe_safe") {
		t.Fatalf("authorized Entrypoint detail omitted topology: %s", detailWire)
	}
}

func TestRoutingResolveResponseOmitsUnselectedRules(t *testing.T) {
	resolution := routingsnapshot.Resolution{
		Outcome: routingsnapshot.ResolveMatched,
		Entrypoint: &routingsnapshot.Entrypoint{
			ID: "entrypoint_safe", Revision: 3, Name: "Safe", Aliases: []string{"safe"},
			Rules: []routingsnapshot.EntrypointRule{{ID: "unselected_rule", Name: "Do not expose"}},
		},
		Rule:   &routingsnapshot.EntrypointRule{ID: "selected_rule", Name: "Selected"},
		Recipe: &routingsnapshot.Recipe{ID: "recipe_safe", Revision: 4, Name: "Recipe"},
	}
	wire, err := json.Marshal(routingResolveResponseDTO(resolution))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(wire), "unselected_rule") || !strings.Contains(string(wire), "selected_rule") {
		t.Fatalf("resolve response topology boundary is wrong: %s", wire)
	}
}
