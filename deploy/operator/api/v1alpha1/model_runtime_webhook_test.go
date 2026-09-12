package v1alpha1

import (
	"context"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

func TestModelRuntimeFieldsSurviveAdmissionAndDeepCopy(t *testing.T) {
	sr := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{
		ModelDeployments: &apiextensionsv1.JSON{Raw: []byte(`{"local":{"provider":"candle","artifact":"models/checkpoint","input":{"max_tokens":512,"overflow":"reject"}}}`)},
		ModelAdmission:   &apiextensionsv1.JSON{Raw: []byte(`{"local":{"max_concurrency":2}}`)},
		Routing:          &apiextensionsv1.JSON{Raw: []byte(`{"model_bindings":{"pii_classifier":{"deployment":"local","contract":"token_spans.v1","adapter":"mmbert"}}}`)},
		PromptGuard:      &PromptGuardConfig{Backend: &RemoteClassifierBackendConfig{Protocol: "http_classify", Contract: "label_distribution.v1", Model: "guardrail-service"}},
	}}}
	if _, err := sr.ValidateCreate(context.Background(), sr); err != nil {
		t.Fatal(err)
	}
	clone := sr.DeepCopy()
	if _, err := sr.ValidateUpdate(context.Background(), sr, clone); err != nil {
		t.Fatal(err)
	}
	clone.Spec.Config.ModelDeployments.Raw[0] = '!'
	clone.Spec.Config.ModelAdmission.Raw[0] = '!'
	clone.Spec.Config.PromptGuard.Backend.Model = "changed"
	if sr.Spec.Config.ModelDeployments.Raw[0] != '{' || sr.Spec.Config.ModelAdmission.Raw[0] != '{' || sr.Spec.Config.PromptGuard.Backend.Model != "guardrail-service" {
		t.Fatal("copied model runtime config aliases the original")
	}
}

func TestPromptGuardLegacyProtocolRejectedAtAdmission(t *testing.T) {
	sr := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{PromptGuard: &PromptGuardConfig{Protocol: "http_classify"}}}}
	if _, err := sr.ValidateCreate(context.Background(), sr); err == nil || !strings.Contains(err.Error(), "named backend") {
		t.Fatalf("legacy protocol accepted: %v", err)
	}
	if _, err := sr.ValidateUpdate(context.Background(), &SemanticRouter{}, sr); err == nil {
		t.Fatal("legacy protocol update accepted")
	}
}
