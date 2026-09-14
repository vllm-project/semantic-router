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
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"shared":{"artifact":"models/checkpoint","revision":"pinned","provider":"candle","device":"cpu","precision":"fp32","input":{"max_tokens":512,"overflow":"reject"}}}`),
		ModelAdmission:   rawCanonicalRoutingJSON(t, `{"shared":{"max_concurrency":2,"max_queue":3,"queue_timeout_ms":500,"on_overflow":"shed"}}`),
		Routing:          rawCanonicalRoutingJSON(t, `{"model_bindings":{"pii_classifier":{"deployment":"shared","contract":"token_spans.v1","adapter":"mmbert","head":"pii","mapping_path":"mappings/pii.json"}}}`),
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
	wantDeployment := routerconfig.ModelDeployment{Artifact: "models/checkpoint", Revision: "pinned", Provider: "candle", Device: "cpu", Precision: "fp32", Input: routerconfig.ModelInputBudget{MaxTokens: 512, Overflow: "reject"}}
	if !reflect.DeepEqual(roundTrip.Global.ModelCatalog.Deployments["shared"], wantDeployment) {
		t.Fatalf("deployment was changed: %#v", roundTrip.Global.ModelCatalog.Deployments)
	}
	wantBudget := routerconfig.AdmissionConfig{MaxConcurrency: 2, MaxQueue: 3, QueueTimeoutMs: 500, OnOverflow: "shed"}
	if roundTrip.Global.ModelCatalog.Admission["shared"] != wantBudget {
		t.Fatalf("deployment admission was lost: %#v", roundTrip.Global.ModelCatalog.Admission)
	}
	wantBinding := routerconfig.ModelBinding{Deployment: "shared", Contract: "token_spans.v1", Adapter: "mmbert", Head: "pii", MappingPath: "mappings/pii.json"}
	if roundTrip.Routing.ModelBindings["pii_classifier"] != wantBinding {
		t.Fatalf("task binding was lost: %#v", roundTrip.Routing.ModelBindings)
	}
	// Explicit empty override clears the previously supplied binding map.
	routing, fields, err := canonicalRoutingFromKubernetesJSON(rawCanonicalRoutingJSON(t, `{"model_bindings":{}}`))
	if err != nil {
		t.Fatal(err)
	}
	applyCanonicalRoutingOverrides(canonical, routing, fields)
	if len(canonical.Routing.ModelBindings) != 0 {
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
	if _, _, err := canonicalRoutingFromKubernetesJSON(rawCanonicalRoutingJSON(t, `{"model_bindings":{"pii_classifier":{"deploymnt":"typo"}}}`)); err == nil {
		t.Fatal("unknown binding field silently discarded")
	}
}
