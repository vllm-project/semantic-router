//go:build onnx && !windows && cgo

package classification

import (
	"errors"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestInitializeLegacyUnifiedClassifierRejectsUnsupportedONNXBackend(t *testing.T) {
	_, err := initializeLegacyUnifiedClassifier(&ModelPaths{
		ModernBertBase:     "models/modernbert-base",
		IntentClassifier:   "models/intent",
		PIIClassifier:      "models/pii",
		SecurityClassifier: "models/security",
	})
	if err == nil {
		t.Fatal("expected legacy unified classifier init to fail on onnx backend")
	}

	var unsupported candle_binding.UnsupportedFeatureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected UnsupportedFeatureError, got %T", err)
	}
	if unsupported.Feature != candle_binding.FeatureUnifiedClassification {
		t.Fatalf("expected unified classification feature error, got %q", unsupported.Feature)
	}
}

func TestInitializeLoRAUnifiedClassifierRejectsUnsupportedONNXBackend(t *testing.T) {
	_, err := initializeLoRAUnifiedClassifier(&ModelPaths{
		LoRAIntentClassifier:   "models/lora-intent",
		LoRAPIIClassifier:      "models/lora-pii",
		LoRASecurityClassifier: "models/lora-security",
		LoRAArchitecture:       "bert",
	})
	if err == nil {
		t.Fatal("expected lora unified classifier init to fail on onnx backend")
	}

	var unsupported candle_binding.UnsupportedFeatureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected UnsupportedFeatureError, got %T", err)
	}
	if unsupported.Feature != candle_binding.FeatureLoRABatchInference {
		t.Fatalf("expected LoRA batch feature error, got %q", unsupported.Feature)
	}
}

func TestUnifiedClassifierInitializeRejectsUnsupportedONNXBackend(t *testing.T) {
	classifier := &UnifiedClassifier{}

	err := classifier.Initialize(
		"models/modernbert-base",
		"models/intent",
		"models/pii",
		"models/security",
		testUnifiedIntentLabels,
		testUnifiedPIILabels,
		testUnifiedSecurityLabels,
		true,
	)
	if err == nil {
		t.Fatal("expected direct unified classifier init to fail on onnx backend")
	}

	var unsupported candle_binding.UnsupportedFeatureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected UnsupportedFeatureError, got %T", err)
	}
	if unsupported.Feature != candle_binding.FeatureUnifiedClassification {
		t.Fatalf("expected unified classification feature error, got %q", unsupported.Feature)
	}
}

func TestUnifiedClassifierInitializeLoRABindingsRejectsUnsupportedONNXBackend(t *testing.T) {
	classifier := &UnifiedClassifier{
		useLoRA: true,
		loraModelPaths: &LoRAModelPaths{
			IntentPath:   "models/lora-intent",
			PIIPath:      "models/lora-pii",
			SecurityPath: "models/lora-security",
			Architecture: "bert",
		},
	}

	err := classifier.initializeLoRABindings()
	if err == nil {
		t.Fatal("expected direct LoRA binding init to fail on onnx backend")
	}

	var unsupported candle_binding.UnsupportedFeatureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected UnsupportedFeatureError, got %T", err)
	}
	if unsupported.Feature != candle_binding.FeatureLoRABatchInference {
		t.Fatalf("expected LoRA batch feature error, got %q", unsupported.Feature)
	}
}
