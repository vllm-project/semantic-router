package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func remotePIIConfig() *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ExternalModels = []config.ExternalModelConfig{{
		Name:          "pii-svc",
		ModelRole:     config.ModelRoleClassification,
		ModelName:     "pii-spans",
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 18080, Protocol: "http"},
	}}
	cfg.PIIModel.Backend = &config.RemoteClassifierBackend{
		Protocol: config.RemoteClassifierProtocolHTTPClassify,
		Contract: config.RemoteClassifierContractTokenSpans,
		Model:    "pii-svc",
	}
	cfg.PIIMappingPath = "models/pii/pii_type_mapping.json"
	return cfg
}

// A remote PII backend in a configuration where no reachable routing decision
// consumes the PII signal: the mapping loader hands BuildClassifier a nil
// mapping on purpose, and the backend must not be constructed against it
// (Xunzhuo's review on #3498: a dormant configuration failed startup).
func TestBuildClassifierSkipsDormantRemotePIIBackend(t *testing.T) {
	classifier, err := BuildClassifier(remotePIIConfig(), nil, nil, nil)
	if err != nil {
		t.Fatalf("dormant remote PII backend must not fail the build: %v", err)
	}
	if classifier.piiInference != nil {
		t.Fatal("no PII consumer, but a remote PII backend was constructed")
	}
	if classifier.IsPIIEnabled() {
		t.Fatal("PII must read as disabled when no mapping was loaded")
	}
}

// The same configuration with a loaded mapping, i.e. a reachable PII signal,
// builds the remote backend.
func TestBuildClassifierBuildsReachableRemotePIIBackend(t *testing.T) {
	classifier, err := BuildClassifier(remotePIIConfig(), nil, testPIIMapping(), nil)
	if err != nil {
		t.Fatalf("reachable remote PII backend failed to build: %v", err)
	}
	// The admission gate wraps every inference since #3268; the remote
	// backend must be what sits under it.
	inference := classifier.piiInference
	if admitted, ok := inference.(admittedPIIInference); ok {
		inference = admitted.backend
	}
	if _, ok := inference.(*remoteTokenBinding); !ok {
		t.Fatalf("piiInference = %T, want owned remote token binding", inference)
	}
	if !classifier.IsPIIEnabled() {
		t.Fatal("PII must read as enabled with a mapping and a remote backend")
	}
}
