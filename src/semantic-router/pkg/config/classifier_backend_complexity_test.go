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
