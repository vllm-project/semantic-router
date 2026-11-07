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

// ruleDeploymentComplexityConfig names the scorer on each rule, the way a
// canonical configuration writes it.
func ruleDeploymentComplexityConfig() *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"difficulty-scorer": {Provider: config.ClassifyProvider, Endpoint: "http://127.0.0.1:8080", Model: "difficulty-service"},
	}
	for _, name := range []string{"needs_reasoning", "extreme"} {
		cfg.ComplexityRules = append(cfg.ComplexityRules, config.ComplexityRule{
			Name:       name,
			Deployment: "difficulty-scorer",
			HardAbove:  float64Ptr(0.80),
			EasyBelow:  float64Ptr(0.30),
		})
	}
	return cfg
}

// A remote-only complexity config must not build the local prototype
// classifier. Building it makes remote startup depend on local embedding model
// resources - the work the advisory tells the user is never read. The Router
// reads a deprecated module backend as a deployment when it loads, so the
// loaded form has no backend left to look for.
func TestRemoteComplexityDoesNotBuildTheLocalClassifier(t *testing.T) {
	configs := map[string]*config.RouterConfig{"rule deployment": ruleDeploymentComplexityConfig()}
	for _, contract := range []string{
		config.RemoteClassifierContractScore,
		config.RemoteClassifierContractLabelDistribution,
	} {
		configs["module backend "+contract] = remoteComplexityConfig(contract)
		configs["loaded module backend "+contract] = config.WithModelResources(remoteComplexityConfig(contract))
	}
	for name, cfg := range configs {
		builder := newClassifierOptionBuilder(cfg, nil)

		classifierOption, err := builder.buildComplexityClassifierOption()
		if err != nil {
			t.Fatalf("%s: buildComplexityClassifierOption: %v", name, err)
		}
		if classifierOption != nil {
			t.Errorf("%s: the local prototype classifier was built for a remote-only config", name)
		}
	}
}

// A remote scorer keeps its bounded input however the configuration names it.
func TestRemoteComplexityKeepsItsBoundedInput(t *testing.T) {
	for name, cfg := range map[string]*config.RouterConfig{
		"rule deployment":       ruleDeploymentComplexityConfig(),
		"loaded module backend": config.WithModelResources(remoteComplexityConfig(config.RemoteClassifierContractScore)),
	} {
		cfg.EmbeddingConfig.FullContext = true
		if (&Classifier{Config: cfg}).hasLongContextClassifier(config.SignalTypeComplexity) {
			t.Errorf("%s: a remote complexity scorer read the full context", name)
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
