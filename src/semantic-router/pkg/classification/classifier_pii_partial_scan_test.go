package classification

import (
	"context"
	"errors"
	"fmt"
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
	// The remote decoder strips one prefix before the detection API translates
	// the result again. Reserve stacked aliases too, not just one BIO tag.
	for _, prefix := range []string{"", "B-", "I-", "E-", "B-B-", "B-I-", "B-E-", "I-B-", "I-I-", "I-E-", "E-B-", "E-I-", "E-E-", "B-I-E-"} {
		t.Run(prefix+PIIClassificationErrorType, func(t *testing.T) {
			testPIIMappingRejectsSentinel(t, prefix+PIIClassificationErrorType)
		})
	}
}

func testPIIMappingRejectsSentinel(t *testing.T, label string) {
	t.Helper()
	for name, body := range map[string]string{
		"in label_to_idx":   fmt.Sprintf(`{"label_to_idx": {"O": 0, %q: 1}, "idx_to_label": {"0": "O", "1": "OTHER"}}`, label),
		"in idx_to_label":   fmt.Sprintf(`{"label_to_idx": {"O": 0, "OTHER": 1}, "idx_to_label": {"0": "O", "1": %q}}`, label),
		"label_to_idx only": fmt.Sprintf(`{"label_to_idx": {"O": 0, %q: 1}}`, label),
		"idx_to_label only": fmt.Sprintf(`{"idx_to_label": {"0": "O", "1": %q}}`, label),
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pii_type_mapping.json")
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatalf("write mapping: %v", err)
			}
			if mapping, err := LoadPIIMapping(path); err == nil || mapping != nil {
				t.Fatalf("mapping with reserved label %q: mapping=%v err=%v, want nil mapping and error", label, mapping, err)
			}
		})
	}
}

func TestLoadPIIMappingAcceptsBIOEntityLabels(t *testing.T) {
	for _, prefix := range []string{"", "B-", "I-", "E-"} {
		t.Run(prefix+"EMAIL_ADDRESS", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pii_type_mapping.json")
			label := prefix + "EMAIL_ADDRESS"
			body := fmt.Sprintf(`{"label_to_idx": {"O": 0, %q: 1}, "idx_to_label": {"0": "O", "1": %q}}`, label, label)
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatalf("write mapping: %v", err)
			}
			mapping, err := LoadPIIMapping(path)
			if err != nil {
				t.Fatalf("ordinary mapping was rejected: %v", err)
			}
			if got := mapping.TranslatePIIType("class_1"); got != "EMAIL_ADDRESS" {
				t.Fatalf("native label = %q, want EMAIL_ADDRESS", got)
			}
			known, _ := knownPIILabels(mapping)
			if _, ok := known["EMAIL_ADDRESS"]; !ok {
				t.Fatalf("remote label set = %v, want EMAIL_ADDRESS", known)
			}
			if _, ok := known[PIIClassificationErrorType]; ok {
				t.Fatal("remote label set contains the reserved sentinel")
			}
		})
	}
}
