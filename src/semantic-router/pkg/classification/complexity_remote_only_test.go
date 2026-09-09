package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func remoteComplexityConfig(contract string) *config.RouterConfig {
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{
			Name:          "difficulty-scorer",
			ModelRole:     config.ModelRoleClassification,
			ModelName:     "difficulty-scorer-svc",
			ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080, Protocol: "http"},
		}},
	}
	cfg.ComplexityModel = config.ComplexityModelConfig{
		Backend: &config.RemoteClassifierBackend{
			Protocol: config.RemoteClassifierProtocolHTTPClassify,
			Contract: contract,
			Model:    "difficulty-scorer",
		},
	}
	cfg.ComplexityRules = []config.ComplexityRule{{
		Name:      "needs_reasoning",
		HardAbove: float64Ptr(0.80),
		EasyBelow: float64Ptr(0.30),
		// A config mid-migration legitimately still carries candidates. With a
		// backend they are never read, so their presence must not drag the
		// local embedding path into a remote-only startup.
		Hard: config.ComplexityCandidates{Candidates: []string{"solve this step by step"}},
		Easy: config.ComplexityCandidates{Candidates: []string{"answer briefly"}},
	}}
	return cfg
}

// A remote-only complexity config must not build the local prototype
// classifier. Building it makes remote startup depend on local embedding model
// resources - the work the advisory tells the user is never read.
func TestRemoteComplexityDoesNotBuildTheLocalClassifier(t *testing.T) {
	for _, contract := range []string{
		config.RemoteClassifierContractScore,
		config.RemoteClassifierContractLabelDistribution,
	} {
		builder := newClassifierOptionBuilder(remoteComplexityConfig(contract), nil)

		option, err := builder.buildComplexityClassifierOption()
		if err != nil {
			t.Fatalf("contract %q: buildComplexityClassifierOption: %v", contract, err)
		}
		if option != nil {
			t.Errorf("contract %q: the local prototype classifier was built for a remote-only config", contract)
		}
	}
}

// Readiness must follow whichever path actually produces the signal, or a
// remote-only config reports complexity as unavailable and the dispatcher
// skips it.
func TestComplexitySignalReadinessFollowsTheActivePath(t *testing.T) {
	empty := &config.RouterConfig{}
	cases := map[string]*Classifier{
		"score backend":  {Config: empty, complexityScoreBackend: &closableScorer{}},
		"label backend":  {Config: empty, complexityLabelBackend: &closableSequenceBackend{closableScorer: &closableScorer{}}},
		"local fallback": {Config: empty, complexityClassifier: &ComplexityClassifier{}},
	}

	for name, classifier := range cases {
		if !classifier.signalReadiness()[config.SignalTypeComplexity] {
			t.Errorf("%s: complexity should be ready", name)
		}
	}

	if (&Classifier{Config: empty}).signalReadiness()[config.SignalTypeComplexity] {
		t.Error("complexity should not be ready with no path configured at all")
	}
}
