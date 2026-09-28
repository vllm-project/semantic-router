package controllers

import (
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCanonicalRoutingPolicyOverrides(t *testing.T) {
	raw := &apiextensionsv1.JSON{Raw: []byte(`{"candidate_requirements":{"capabilities":"declared","context":"known_limits"},"data_policy":{"replay":false}}`)}
	routing, fields, err := canonicalRoutingFromKubernetesJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	canonical := &routerconfig.CanonicalConfig{}
	applyCanonicalRoutingOverrides(canonical, routing, fields)
	if canonical.Routing.CandidateRequirements == nil || canonical.Routing.CandidateRequirements.Context != routerconfig.CandidateContextKnownLimits || canonical.Routing.DataPolicy.ReplayAllowed() {
		t.Fatal("operator lost routing policy")
	}
	*routing.DataPolicy.Replay = true
	if canonical.Routing.DataPolicy.ReplayAllowed() {
		t.Fatal("operator policy shares mutable input")
	}
	for _, invalid := range []string{
		`{"candidate_requirements":{"context":"bounded"}}`,
		`{"data_policy":{"replay":false,"unknown":true}}`,
	} {
		if _, _, err := canonicalRoutingFromKubernetesJSON(&apiextensionsv1.JSON{Raw: []byte(invalid)}); err == nil {
			t.Fatalf("accepted %s", invalid)
		}
	}
}
