package classification

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// realPIIModelPath resolves the on-disk mmBERT-32K PII model, honoring the
// VLLM_SR_PII_MODEL override and otherwise looking for the repo-root models/
// download. It returns "" when the model is not present.
func realPIIModelPath() string {
	if p := os.Getenv("VLLM_SR_PII_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		return ""
	}
	// go test runs with the package directory as the working directory.
	def := filepath.Join("..", "..", "..", "..", "models", "mmbert32k-pii-detector-merged")
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return ""
}

// setupRealPIIClassifier initializes the real mmBERT-32K PII model and builds a
// Classifier wired to the real inference backend. It skips the test when the
// model is not present (e.g. minimal-model CI).
func setupRealPIIClassifier(t *testing.T) *Classifier {
	t.Helper()

	modelPath := realPIIModelPath()
	if modelPath == "" {
		t.Skip("mmBERT-32K PII model not present; set VLLM_SR_PII_MODEL or run `make download-mmbert-32k-merged`")
	}

	mappingPath := filepath.Join(modelPath, "pii_type_mapping.json")
	mapping, err := LoadPIIMapping(mappingPath)
	if err != nil {
		t.Fatalf("load PII mapping %q: %v", mappingPath, err)
	}

	if err := candle_binding.InitMmBert32KPIIClassifier(modelPath, true); err != nil {
		t.Fatalf("init mmBERT-32K PII classifier: %v", err)
	}

	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = modelPath
	cfg.PIIModel.UseCPU = true
	cfg.PIIModel.UseMmBERT32K = true
	cfg.PIIModel.Threshold = 0.5
	cfg.PIIMappingPath = mappingPath

	classifier, err := newClassifierWithOptions(cfg,
		withPII(mapping, &MmBERT32KPIIInitializerImpl{}, &MmBERT32KPIIInferenceImpl{}),
	)
	if err != nil {
		t.Fatalf("build classifier: %v", err)
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

// The defect this guards is in the tokenizer, so it only reproduces against the
// real model: a mocked backend never truncates. PII inside the model's window is
// found either way; PII past it is what a single call loses.
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
		detections, err := classifier.ClassifyPIIWithDetails(text)
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

		detections, err := classifier.ClassifyPIIWithDetails(text)
		if err != nil {
			t.Fatalf("ClassifyPIIWithDetails: %v", err)
		}
		t.Logf("text = %d bytes, chunks = %d, detections = %d",
			len(text), len(piiSignalChunkSpans(text)), len(detections))
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
			if got := text[d.Start:d.End]; got != d.Text {
				t.Errorf("offsets must index the original text: text[%d:%d] = %q, entity text %q",
					d.Start, d.End, got, d.Text)
			}
		}
	})
}
