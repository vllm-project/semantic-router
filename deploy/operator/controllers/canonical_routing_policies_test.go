package controllers

import (
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCanonicalRecipePolicyOverrides(t *testing.T) {
	raw := &apiextensionsv1.JSON{Raw: []byte(`[{"name":"default","candidate_requirements":{"capabilities":"declared","context":"known_limits"}}]`)}
	global := routerconfig.DefaultCanonicalGlobal()
	canonical := &routerconfig.CanonicalConfig{Global: &global}
	canonical.Global.Plugins = []routerconfig.DecisionPlugin{{Type: routerconfig.DecisionPluginRouterReplay}}
	if err := applyOperatorRecipes(canonical, configRoutingSpec{recipes: raw}); err != nil {
		t.Fatal(err)
	}
	if defaultRecipe(canonical).CandidateRequirements == nil || defaultRecipe(canonical).CandidateRequirements.Context != routerconfig.CandidateContextKnownLimits {
		t.Fatal("operator lost recipe policy")
	}
	raw.Raw[0] = '!'
	if defaultRecipe(canonical).CandidateRequirements.Context != routerconfig.CandidateContextKnownLimits {
		t.Fatal("operator policy shares mutable input")
	}
	if len(canonical.Global.Plugins) != 1 {
		t.Fatal("a recipe override changed global.plugins")
	}
	for _, invalid := range []string{
		`[{"name":"default","candidate_requirements":{"context":"bounded"}}]`,
		`[{"name":"default","candidate_requirements":{"unknown":true}}]`,
		`[{"name":"default","data_policy":{"replay":false}}]`,
		`[{"name":"default","routing":{"decisions":[]}}]`,
	} {
		global := routerconfig.DefaultCanonicalGlobal()
		target := &routerconfig.CanonicalConfig{Global: &global}
		if err := applyOperatorRecipes(target, configRoutingSpec{recipes: &apiextensionsv1.JSON{Raw: []byte(invalid)}}); err == nil {
			t.Fatalf("accepted %s", invalid)
		}
	}
}
