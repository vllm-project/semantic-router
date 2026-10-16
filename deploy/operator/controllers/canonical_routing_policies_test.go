package controllers

import (
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCanonicalRoutingPolicyOverrides(t *testing.T) {
	raw := &apiextensionsv1.JSON{Raw: []byte(`{"candidate_requirements":{"capabilities":"declared","context":"known_limits"}}`)}
	routing, fields, err := canonicalRoutingFromKubernetesJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	global := routerconfig.DefaultCanonicalGlobal()
	canonical := &routerconfig.CanonicalConfig{Global: &global}
	canonical.Global.Services.RouterReplay.Enabled = true
	applyCanonicalRoutingOverrides(canonical, routing, fields)
	if canonical.Routing.CandidateRequirements == nil || canonical.Routing.CandidateRequirements.Context != routerconfig.CandidateContextKnownLimits {
		t.Fatal("operator lost routing policy")
	}
	routing.CandidateRequirements.Context = ""
	if canonical.Routing.CandidateRequirements.Context != routerconfig.CandidateContextKnownLimits {
		t.Fatal("operator policy shares mutable input")
	}
	if !canonical.Global.Services.RouterReplay.Enabled {
		t.Fatal("routing override changed global replay settings")
	}
	for _, invalid := range []string{
		`{"candidate_requirements":{"context":"bounded"}}`,
		`{"candidate_requirements":{"unknown":true}}`,
		`{"data_policy":{"replay":false}}`,
		`{"data_policy":{"replay":true}}`,
	} {
		if _, _, err := canonicalRoutingFromKubernetesJSON(&apiextensionsv1.JSON{Raw: []byte(invalid)}); err == nil {
			t.Fatalf("accepted %s", invalid)
		}
	}
}
