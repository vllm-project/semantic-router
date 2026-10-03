package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Remote (Backend) classifiers have no local model lifecycle to execute during
// initializeJailbreakClassifier; assembly is the whole lifecycle. They must
// still report ready so readiness checks trust the configured classifier.
func TestJailbreakBackendAssemblyMarksModelReady(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
	cfg.PromptGuard.Backend = &config.RemoteClassifierBackend{
		Protocol: config.RemoteClassifierProtocolHTTPClassify,
		Model:    "jailbreak-detector",
	}

	c := &Classifier{
		Config: cfg,
		JailbreakMapping: &JailbreakMapping{
			LabelToIdx: map[string]int{"harmful": 0, "benign": 1},
			IdxToLabel: map[string]string{"0": "harmful", "1": "benign"},
		},
		jailbreakInference: &stubSequenceBackend{},
	}

	if err := c.initializeJailbreakClassifier(); err != nil {
		t.Fatalf("remote jailbreak classifier must not require a local model lifecycle: %v", err)
	}
	if !c.IsJailbreakModelReady() {
		t.Fatal("remote jailbreak classifier must report ready after initialization")
	}
}

// A classifier with an unset Protocol field and no remote Backend or local
// model lifecycle must not report ready; the readiness check then keeps
// returning 503 instead of trusting an uninitialized detector.
func TestJailbreakModelNotReadyWithoutBackendOrLocalInit(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true

	c := &Classifier{Config: cfg}

	if c.IsJailbreakModelReady() {
		t.Fatal("jailbreak detector with no remote backend and no local lifecycle must not report ready")
	}
}

// Remote (Backend) PII classifiers have no local model lifecycle to execute
// during initializePIIClassifier; they must still report ready.
func TestPIIBackendAssemblyMarksModelReady(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PIIModel.Backend = &config.RemoteClassifierBackend{
		Protocol: config.RemoteClassifierProtocolHTTPClassify,
		Model:    "pii-detector",
	}

	c := &Classifier{Config: cfg}

	if err := c.initializePIIClassifier(); err != nil {
		t.Fatalf("remote PII classifier must not require a local model lifecycle: %v", err)
	}
	if !c.IsPIIModelReady() {
		t.Fatal("remote PII classifier must report ready after initialization")
	}
}
