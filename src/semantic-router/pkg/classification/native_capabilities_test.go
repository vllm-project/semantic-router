package classification

import (
	"strings"
	"testing"
)

func TestCurrentNativeBackendCapabilitiesShape(t *testing.T) {
	capabilities := CurrentNativeBackendCapabilities()
	if capabilities.Name == "" {
		t.Fatal("native backend capability name must be set")
	}

	switch capabilities.Name {
	case "candle":
		if !capabilities.UnifiedBatchClassification {
			t.Fatal("candle backend should advertise unified batch classification")
		}
		if !capabilities.LoRABatchClassification {
			t.Fatal("candle backend should advertise LoRA batch classification")
		}
		if !capabilities.MultimodalEmbedding {
			t.Fatal("candle backend should advertise multimodal embedding support")
		}
	case "onnx", "stub":
		if capabilities.LoRABatchClassification {
			t.Fatalf("%s backend should not advertise LoRA batch classification", capabilities.Name)
		}
	default:
		t.Fatalf("unexpected native backend capability name: %s", capabilities.Name)
	}
}

func TestUnifiedClassifierRejectsUnsupportedNativeCapabilities(t *testing.T) {
	original := nativeBackendCapabilities
	defer func() { nativeBackendCapabilities = original }()

	nativeBackendCapabilities = NativeBackendCapabilities{Name: "test-backend"}

	classifier := &UnifiedClassifier{initialized: true}
	_, err := classifier.ClassifyBatch([]string{"hello"})
	if err == nil || !strings.Contains(err.Error(), `native backend "test-backend" does not support unified batch classification`) {
		t.Fatalf("expected unsupported unified batch error, got %v", err)
	}

	classifier = &UnifiedClassifier{initialized: true, useLoRA: true}
	_, err = classifier.ClassifyBatch([]string{"hello"})
	if err == nil || !strings.Contains(err.Error(), `native backend "test-backend" does not support LoRA unified batch classification`) {
		t.Fatalf("expected unsupported LoRA batch error, got %v", err)
	}

	err = (&UnifiedClassifier{}).Initialize(
		"modernbert",
		"intent",
		"pii",
		"security",
		testUnifiedIntentLabels,
		testUnifiedPIILabels,
		testUnifiedSecurityLabels,
		true,
	)
	if err == nil || !strings.Contains(err.Error(), `native backend "test-backend" does not support unified batch classification`) {
		t.Fatalf("expected initialize capability error, got %v", err)
	}
}

func TestUnifiedClassifierStatsIncludeNativeCapabilities(t *testing.T) {
	stats := (&UnifiedClassifier{}).GetStats()
	value, ok := stats["native_backend"].(NativeBackendCapabilities)
	if !ok {
		t.Fatalf("expected native_backend stats to be NativeBackendCapabilities, got %T", stats["native_backend"])
	}
	if value.Name == "" {
		t.Fatal("expected native backend name in stats")
	}
}
