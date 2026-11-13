package v1alpha1

import (
	"context"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

func TestModelRuntimeFieldsSurviveAdmissionAndDeepCopy(t *testing.T) {
	sr := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{
		ModelDeployments: &apiextensionsv1.JSON{Raw: []byte(`{"local":{"provider":"candle","artifact":"models/checkpoint","input":{"max_tokens":512,"overflow":"reject"}}}`)},
		ModelAdmission:   &apiextensionsv1.JSON{Raw: []byte(`{"local":{"max_concurrency":2}}`)},
		ModelBindings:    &apiextensionsv1.JSON{Raw: []byte(`{"pii":"local"}`)},
		Recipes:          &apiextensionsv1.JSON{Raw: []byte(`[{"name": "default", "model_bindings":{"pii":{"deployment":"local"}}}]`)},
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
	clone.Spec.Config.ModelBindings.Raw[0] = '!'
	clone.Spec.Config.Recipes.Raw[0] = '!'
	if sr.Spec.Config.ModelDeployments.Raw[0] != '{' || sr.Spec.Config.ModelAdmission.Raw[0] != '{' || sr.Spec.Config.ModelBindings.Raw[0] != '{' ||
		sr.Spec.Config.Recipes.Raw[0] != '[' {
		t.Fatal("copied model runtime config aliases the original")
	}
}
