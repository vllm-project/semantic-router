package classification

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupRealPIIClassifier(t *testing.T) *Classifier {
	t.Helper()
	defaults := config.DefaultGlobalConfig()
	modelPath := requireRealModel(t, "VLLM_SR_PII_MODEL", defaults.PIIModel.ModelID)
	mappingPath := filepath.Join(modelPath, filepath.Base(defaults.PIIMappingPath))
	mapping, err := LoadPIIMapping(mappingPath)
	if err != nil {
		t.Fatalf("load PII mapping: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.PIIModel = defaults.PIIModel
	models, err := newClassifierModelRuntime(cfg, nil)
	if err != nil {
		t.Fatalf("prepare PII runtime: %v", err)
	}
	cfg = models.cfg
	cfg.PIIModel.ModelID = modelPath
	cfg.PIIMappingPath = mappingPath
	initializer, backend, err := buildPIIDependencies(cfg, mapping, models)
	if err != nil {
		t.Fatalf("build PII dependencies: %v", err)
	}
	classifier, err := newClassifierWithOptions(cfg, withPII(mapping, initializer, backend))
	if err != nil {
		t.Fatalf("build PII classifier: %v", err)
	}
	t.Cleanup(func() {
		if err := classifier.Close(); err != nil {
			t.Errorf("close PII classifier: %v", err)
		}
	})
	if err := classifier.initializePIIClassifier(); err != nil {
		t.Fatalf("initialize PII classifier: %v", err)
	}
	switch prepared := backend.(type) {
	case *windowedPIIBackend:
		assertRealModelCPU(t, prepared.handle.Capability())
	case *ownedTokenBackend:
		assertRealModelCPU(t, prepared.handle.Capability())
	default:
		t.Fatalf("expected an owned native PII backend, got %T", backend)
	}
	return classifier
}

// coversSpan reports whether any detection overlaps [start, end) in text, and
// that its own offsets index the original text. The real model can also flag
// entities inside the filler, so this asks about the planted span rather than
// demanding an exact detection count.
func coversSpan(t *testing.T, text string, detections []PIIDetection, start, end int) bool {
	t.Helper()
	for _, d := range detections {
		if d.Start < 0 || d.End > len(text) || d.Start >= d.End {
			t.Errorf("detection offsets outside the original text: [%d-%d] for a text of %d bytes",
				d.Start, d.End, len(text))
			continue
		}
		if d.Start < end && start < d.End {
			return true
		}
	}
	return false
}

// Real tokenization and the owned token-window task must preserve detections
// beyond the first 512-token forward and report offsets in the original input.
func TestClassifyPIIWithDetails_RealModelFindsPIIPastTheWindow(t *testing.T) {
	classifier := setupRealPIIClassifier(t)

	const secret = "Contact John Doe at john.doe@example.com or 555-123-4567."

	// Filler deliberately carries no names, numbers or dates: the model flags
	// those, and the assertion below is about the planted span only.
	sentences := []string{
		"Sailors used the stars to navigate before mechanical instruments existed. ",
		"The compass then made direction independent of a clear night sky. ",
		"Radio beacons later fixed a position without any view of the horizon. ",
		"Satellite systems eventually replaced every one of the earlier techniques. ",
	}
	var builder strings.Builder
	for i := 0; i < 96; i++ {
		builder.WriteString(sentences[i%len(sentences)])
	}
	filler := builder.String()

	t.Run("inside the window", func(t *testing.T) {
		text := secret + " " + filler
		detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
		if err != nil {
			t.Fatalf("ClassifyPIIWithDetails: %v", err)
		}
		if !coversSpan(t, text, detections, 0, len(secret)) {
			t.Fatalf("PII at the head of the text must be detected; got %d detections", len(detections))
		}
	})

	t.Run("past the window", func(t *testing.T) {
		text := filler + " " + secret
		start := strings.Index(text, secret)

		detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
		if err != nil {
			t.Fatalf("ClassifyPIIWithDetails: %v", err)
		}
		t.Logf("text = %d bytes, chunks = %d, detections = %d",
			len(text), len(classifier.piiInputSpans(text)), len(detections))
		for _, d := range detections {
			t.Logf("  %-14s [%d-%d] %q %.3f", d.EntityType, d.Start, d.End, d.Text, d.Confidence)
		}

		if !coversSpan(t, text, detections, start, start+len(secret)) {
			t.Fatalf("PII past the model window must be detected; got %d detections", len(detections))
		}

		// Offsets are remapped from a chunk onto the original text. The entity
		// text the model reported has to slice back out of the original at the
		// reported offsets, or masked_text and start_position are wrong.
		for _, d := range detections {
			if d.Start < 0 || d.End > len(text) || d.Start >= d.End {
				t.Errorf("invalid detection offsets [%d:%d]", d.Start, d.End)
				continue
			}
			if got := text[d.Start:d.End]; got != d.Text {
				t.Errorf("offsets must index the original text: text[%d:%d] = %q, entity text %q",
					d.Start, d.End, got, d.Text)
			}
		}
	})
}
