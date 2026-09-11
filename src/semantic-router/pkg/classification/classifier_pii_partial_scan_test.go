package classification

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A partial scan must never read as a complete one. Under on_error: allow the
// detection APIs keep the spans the provider did return and say the scan was
// incomplete; under block they refuse outright. The batch API used to omit the
// item nobody could verify, which a caller cannot tell from a clean item.
func TestPIIDetectionAPIsReportAnIncompleteScan(t *testing.T) {
	const text = "write to alice@corp.example about the rest the provider never saw"
	email := piiEntity("EMAIL_ADDRESS", "alice@corp.example", 9, 27, 0.99)

	t.Run("allow surfaces the truncation alongside the results", func(t *testing.T) {
		classifier, _, mockModel := newTestPIIClassifier()
		classifier.Config.PIIModel.OnError = config.OnErrorAllow
		mockModel.setMockResponse(text, []candle_binding.TokenEntity{email}, ErrTokenSpansTruncated)

		types, err := classifier.ClassifyPII(context.Background(), text)
		if !errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("ClassifyPII err = %v, want the truncation sentinel", err)
		}
		if len(types) == 0 {
			t.Fatal("ClassifyPII dropped the types it did find")
		}

		detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
		if !errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("ClassifyPIIWithDetails err = %v, want the truncation sentinel", err)
		}
		if len(detections) == 0 {
			t.Fatal("ClassifyPIIWithDetails dropped the spans before the cut")
		}

		hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{text})
		if !errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("AnalyzeContentForPII err = %v, want the truncation sentinel", err)
		}
		if !hasPII || len(results) != 1 {
			t.Fatalf("AnalyzeContentForPII hasPII=%v results=%d, want the partial detection kept", hasPII, len(results))
		}
	})

	t.Run("block refuses rather than omitting the unverified item", func(t *testing.T) {
		classifier, _, mockModel := newTestPIIClassifier()
		classifier.Config.PIIModel.OnError = config.OnErrorBlock
		const clean = "nothing to see here"
		mockModel.setMockResponse(text, nil, errors.New("connection refused"))
		mockModel.setMockResponse(clean, nil, nil)

		// The failing item comes first and a clean one follows: the old
		// behaviour returned success with only the clean item, which reads as
		// a complete scan of both.
		hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{text, clean})
		if err == nil {
			t.Fatalf("AnalyzeContentForPII returned hasPII=%v results=%d and no error; unverified content must not be omitted", hasPII, len(results))
		}
		if results != nil {
			t.Fatalf("results = %v, want none when the batch is refused", results)
		}
	})
}

// The jailbreak mapping already refuses a label that collides with its
// on_error sentinel, because a genuine detection of that label would be
// indistinguishable from a classify failure. PII decides error-driven matches
// on the same distinction, and a remote backend can only return labels the
// mapping declares, so the PII loader must refuse it too.
func TestLoadPIIMappingRejectsTheSentinelLabel(t *testing.T) {
	for name, body := range map[string]string{
		"in label_to_idx": `{"label_to_idx": {"O": 0, "classification_error": 1}, "idx_to_label": {"0": "O", "1": "OTHER"}}`,
		"in idx_to_label": `{"label_to_idx": {"O": 0, "OTHER": 1}, "idx_to_label": {"0": "O", "1": "classification_error"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pii_type_mapping.json")
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatalf("write mapping: %v", err)
			}
			if _, err := LoadPIIMapping(path); err == nil {
				t.Fatal("mapping with the reserved sentinel label was accepted")
			}
		})
	}

	path := filepath.Join(t.TempDir(), "pii_type_mapping.json")
	if err := os.WriteFile(path, []byte(`{"label_to_idx": {"O": 0, "EMAIL_ADDRESS": 1}, "idx_to_label": {"0": "O", "1": "EMAIL_ADDRESS"}}`), 0o600); err != nil {
		t.Fatalf("write mapping: %v", err)
	}
	if _, err := LoadPIIMapping(path); err != nil {
		t.Fatalf("ordinary mapping was rejected: %v", err)
	}
}
