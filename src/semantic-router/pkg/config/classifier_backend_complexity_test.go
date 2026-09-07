package config

import (
	"strings"
	"testing"
)

func complexityBackendConfig(backend *RemoteClassifierBackend) *RouterConfig {
	return &RouterConfig{
		InlineModels: InlineModels{
			ComplexityModel: ComplexityModelConfig{Backend: backend},
		},
		ExternalModels: []ExternalModelConfig{{
			Name:          "difficulty-scorer",
			ModelRole:     ModelRoleClassification,
			ModelName:     "difficulty-scorer-svc",
			ModelEndpoint: ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080, Protocol: "http"},
		}},
	}
}

// score.v1 is the contract this issue adds. Until it is whitelisted, the
// shared Validate rejects it as an unsupported value.
func TestRemoteClassifierBackend_AcceptsScoreContract(t *testing.T) {
	backend := &RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	}

	if err := backend.Validate(); err != nil {
		t.Fatalf("Validate() rejected the score contract: %v", err)
	}
}

// Complexity accepts two contracts, so the resolver can no longer ask for a
// single expected one. Category keeps passing exactly one, unchanged.
func TestResolveRemoteClassifierBackend_AcceptsAnyDeclaredContract(t *testing.T) {
	for _, contract := range []string{RemoteClassifierContractScore, RemoteClassifierContractLabelDistribution} {
		cfg := complexityBackendConfig(&RemoteClassifierBackend{
			Protocol: RemoteClassifierProtocolHTTPClassify,
			Contract: contract,
			Model:    "difficulty-scorer",
		})

		external, err := ResolveRemoteClassifierBackend(
			cfg,
			cfg.ComplexityModel.Backend,
			ModelRoleClassification,
			RemoteClassifierContractScore,
			RemoteClassifierContractLabelDistribution,
		)
		if err != nil {
			t.Fatalf("contract %q was rejected by a consumer that declares both: %v", contract, err)
		}
		if external == nil || external.Name != "difficulty-scorer" {
			t.Fatalf("contract %q resolved to %#v, want the named external model", contract, external)
		}
	}
}

// A consumer that declares one contract must still reject the other, so
// widening the resolver does not turn it into a pass-through.
func TestResolveRemoteClassifierBackend_RejectsUndeclaredContract(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	})

	_, err := ResolveRemoteClassifierBackend(
		cfg,
		cfg.ComplexityModel.Backend,
		ModelRoleClassification,
		RemoteClassifierContractLabelDistribution,
	)
	if err == nil {
		t.Fatal("expected score.v1 to be rejected by a consumer that only declares label_distribution.v1")
	}
}

// With two contracts on offer there is nothing to default to: omitting the
// contract would leave the router guessing whether the response carries a
// number or a label, and guessing wrong fails at request time rather than at
// config load.
func TestValidateComplexityModelBackend_RequiresAnExplicitContract(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Model:    "difficulty-scorer",
	})

	err := ValidateComplexityModelBackend(cfg)
	if err == nil {
		t.Fatal("expected an omitted contract to be rejected for complexity")
	}
	for _, want := range []string{RemoteClassifierContractScore, RemoteClassifierContractLabelDistribution} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name the valid contract %q", err.Error(), want)
		}
	}
}

func TestValidateComplexityModelBackend_AcceptsBothContracts(t *testing.T) {
	for _, contract := range []string{RemoteClassifierContractScore, RemoteClassifierContractLabelDistribution} {
		cfg := complexityBackendConfig(&RemoteClassifierBackend{
			Protocol: RemoteClassifierProtocolHTTPClassify,
			Contract: contract,
			Model:    "difficulty-scorer",
		})

		if err := ValidateComplexityModelBackend(cfg); err != nil {
			t.Errorf("contract %q rejected: %v", contract, err)
		}
	}
}

// http_chat returns prose, which neither a score nor a label distribution can
// be read from without a parser this issue does not define.
func TestValidateComplexityModelBackend_RejectsUnsupportedProtocol(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPChat,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	})

	if err := ValidateComplexityModelBackend(cfg); err == nil {
		t.Fatal("expected http_chat to be rejected by the complexity consumer")
	}
}

// A nil backend is the local prototype-scoring path and must stay valid.
func TestValidateComplexityModelBackend_NilBackendIsLocalPath(t *testing.T) {
	if err := ValidateComplexityModelBackend(complexityBackendConfig(nil)); err != nil {
		t.Fatalf("a config with no complexity backend must remain valid: %v", err)
	}
}

// The local margin is hardScore minus easyScore, so a higher value is harder
// by construction. A rule that declares the opposite direction without a
// remote score would invert every verdict it reaches, and look perfectly
// reasonable while doing it - the user's own hard examples would score as
// easy. The way to express the inverse locally is to swap the candidate
// lists, so this is rejected rather than honoured.
func TestValidateComplexityModelBackend_RejectsLowerIsHarderWithoutABackend(t *testing.T) {
	cfg := complexityBackendConfig(nil)
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		HardBelow: floatPtr(0.40),
		EasyAbove: floatPtr(0.80),
	}}

	err := ValidateComplexityModelBackend(cfg)
	if err == nil {
		t.Fatal("expected a lower-is-harder pair on a local rule to be rejected")
	}
	if !strings.Contains(err.Error(), "needs_reasoning") {
		t.Errorf("error %q should name the offending rule", err.Error())
	}
}

// With a remote score the direction is the model's, not the margin's, so the
// same pair is exactly what those fields exist for.
func TestValidateComplexityModelBackend_AllowsLowerIsHarderWithAScoreBackend(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	})
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		HardBelow: floatPtr(0.40),
		EasyAbove: floatPtr(0.80),
	}}

	if err := ValidateComplexityModelBackend(cfg); err != nil {
		t.Fatalf("a lower-is-harder pair is valid against a score backend: %v", err)
	}
}

// The higher-is-harder pair reads in the direction the local margin already
// runs, so it stays available locally as the asymmetric form of threshold.
func TestValidateComplexityModelBackend_AllowsHigherIsHarderLocally(t *testing.T) {
	cfg := complexityBackendConfig(nil)
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		HardAbove: floatPtr(0.25),
		EasyBelow: floatPtr(-0.05),
	}}

	if err := ValidateComplexityModelBackend(cfg); err != nil {
		t.Fatalf("an asymmetric higher-is-harder pair is valid locally: %v", err)
	}
}

// routing.signals is replaced wholesale per recipe, so a rule that only
// exists inside a recipe must be checked too.
func TestValidateComplexityModelBackend_ChecksRecipeRules(t *testing.T) {
	cfg := complexityBackendConfig(nil)
	cfg.Recipes = []RoutingRecipe{{
		Name: "internal",
		Profile: RoutingProfile{Signals: Signals{
			ComplexityRules: []ComplexityRule{{
				Name:      "recipe_only_rule",
				HardBelow: floatPtr(0.40),
				EasyAbove: floatPtr(0.80),
			}},
		}},
	}}

	err := ValidateComplexityModelBackend(cfg)
	if err == nil {
		t.Fatal("expected a lower-is-harder pair inside a recipe to be rejected")
	}
	if !strings.Contains(err.Error(), "recipe_only_rule") {
		t.Errorf("error %q should name the offending rule", err.Error())
	}
}

// A malformed pair is reported wherever it appears, backend or not.
func TestValidateComplexityModelBackend_RejectsOverlappingBands(t *testing.T) {
	cfg := complexityBackendConfig(nil)
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		HardAbove: floatPtr(0.05),
		EasyBelow: floatPtr(0.25),
	}}

	if err := ValidateComplexityModelBackend(cfg); err == nil {
		t.Fatal("expected an overlapping band to be reported by the signal validator")
	}
}
