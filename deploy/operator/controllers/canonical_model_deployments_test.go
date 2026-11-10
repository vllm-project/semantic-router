package controllers

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOperatorModelDeploymentsReachCanonicalConfig(t *testing.T) {
	spec := vllmv1alpha1.ConfigSpec{
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"shared":{"artifact":"vllm-sr/checkpoint","revision":"`+strings.Repeat("a", 40)+`","provider":"model_runtime","device":"cpu","input":{"max_tokens":512,"overflow":"reject"}}}`),
		ModelAdmission:   rawCanonicalRoutingJSON(t, `{"shared":{"max_concurrency":2,"max_queue":3,"queue_timeout_ms":500,"on_overflow":"shed"}}`),
		ModelBindings:    rawCanonicalRoutingJSON(t, `{"domain":"shared","safety":{"deployment":"shared"}}`),
		Recipes:          rawCanonicalRoutingJSON(t, `[{"name": "default", "model_bindings":{"pii":{"deployment":"shared"}}}]`),
	}
	r := &SemanticRouterReconciler{}
	canonical, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: spec}})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip routerconfig.CanonicalConfig
	if unmarshalErr := yaml.Unmarshal(encoded, &roundTrip); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	wantDeployment := routerconfig.ModelDeployment{Artifact: "vllm-sr/checkpoint", Revision: strings.Repeat("a", 40), Provider: routerconfig.ModelRuntimeProvider, Device: "cpu", Input: routerconfig.ModelInputBudget{MaxTokens: 512, Overflow: "reject"}}
	if !reflect.DeepEqual(roundTrip.Global.ModelCatalog.Deployments["shared"], wantDeployment) {
		t.Fatalf("deployment was changed: %#v", roundTrip.Global.ModelCatalog.Deployments)
	}
	wantBudget := routerconfig.AdmissionConfig{MaxConcurrency: 2, MaxQueue: 3, QueueTimeoutMs: 500, OnOverflow: "shed"}
	if roundTrip.Global.ModelCatalog.Admission["shared"] != wantBudget {
		t.Fatalf("deployment admission was lost: %#v", roundTrip.Global.ModelCatalog.Admission)
	}
	wantBinding := routerconfig.ModelBinding{Deployment: "shared"}
	if defaultRecipe(&roundTrip).ModelBindings["pii"] != wantBinding {
		t.Fatalf("task binding was lost: %#v", defaultRecipe(&roundTrip).ModelBindings)
	}
	wantGlobal := map[string]routerconfig.ModelBinding{"domain": {Deployment: "shared"}, "safety": {Deployment: "shared"}}
	if !reflect.DeepEqual(roundTrip.Global.ModelCatalog.Bindings, wantGlobal) {
		t.Fatalf("global bindings were changed: %#v", roundTrip.Global.ModelCatalog.Bindings)
	}
	// Explicit empty override clears the previously supplied binding map.
	if err := applyOperatorRecipes(canonical, configRoutingSpec{recipes: rawCanonicalRoutingJSON(t, `[{"name":"default","model_bindings":{}}]`)}); err != nil {
		t.Fatal(err)
	}
	if len(defaultRecipe(canonical).ModelBindings) != 0 {
		t.Fatal("empty binding override did not clear bindings")
	}
}

func TestOperatorModelDeploymentRejectsUnknownFields(t *testing.T) {
	for _, raw := range []string{`null`, `[]`, `{"model":{"unknown_provider":"candle"}}`, `{"model":{"provider":"candle","input":{"max_token":512}}}`} {
		spec := vllmv1alpha1.ConfigSpec{ModelDeployments: rawCanonicalRoutingJSON(t, raw)}
		if err := applyOperatorModelDeployments(&routerconfig.CanonicalConfig{}, spec); err == nil || !strings.Contains(err.Error(), "config.model_deployments") {
			t.Errorf("invalid declaration accepted: %s, %v", raw, err)
		}
	}
	if err := applyOperatorRecipes(&routerconfig.CanonicalConfig{}, configRoutingSpec{recipes: rawCanonicalRoutingJSON(t, `[{"name":"default","model_bindings":{"pii":{"deploymnt":"typo"}}}]`)}); err == nil {
		t.Fatal("unknown binding field silently discarded")
	}
	for _, raw := range []string{`null`, `[]`, `{"pii":{"deploymnt":"typo"}}`, `{"pii":["local"]}`} {
		spec := vllmv1alpha1.ConfigSpec{ModelBindings: rawCanonicalRoutingJSON(t, raw)}
		if err := applyOperatorModelDeployments(&routerconfig.CanonicalConfig{}, spec); err == nil || !strings.Contains(err.Error(), "config.model_bindings") {
			t.Errorf("invalid global binding accepted: %s, %v", raw, err)
		}
	}
}
