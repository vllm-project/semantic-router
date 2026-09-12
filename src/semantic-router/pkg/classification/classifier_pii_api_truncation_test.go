package classification

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// A provider-declared truncation carries valid spans for the part it saw. The
// routing signal has always kept them; the detection and masking APIs used to
// discard every detection in the call, including those from chunks that
// classified cleanly. All three now read classifier.pii.on_error: allow keeps
// what was found and returns ErrTokenSpansTruncated alongside it, block
// refuses the scan. See classifier_pii_partial_scan_test.go for the sentinel
// itself; this table fixes the per-API allow/block split.
func TestPIIDetectionAPIsKeepPartialSpansUnderOnErrorAllow(t *testing.T) {
	const text = "write to alice@corp.example about the rest of this text the provider never saw"
	email := piiEntity("EMAIL_ADDRESS", "alice@corp.example", 9, 27, 0.99)

	for _, tc := range []struct {
		name         string
		onError      string
		wantErr      bool
		wantEntities bool
	}{
		{"allow keeps the partial spans", config.OnErrorAllow, false, true},
		{"block refuses the scan", config.OnErrorBlock, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			classifier, _, mockModel := newTestPIIClassifier()
			classifier.Config.PIIModel.OnError = tc.onError
			mockModel.setMockResponse(text, []tasks.TokenEntity{email}, ErrTokenSpansTruncated)

			detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
			assertTruncationOutcome(t, "ClassifyPIIWithDetails", tc.wantErr, err, len(detections) > 0, tc.wantEntities)

			types, err := classifier.ClassifyPII(context.Background(), text)
			assertTruncationOutcome(t, "ClassifyPII", tc.wantErr, err, slices.Contains(types, "EMAIL_ADDRESS"), tc.wantEntities)

			hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{text})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("AnalyzeContentForPII under block: hasPII=%v results=%d, want the batch refused", hasPII, len(results))
				}
				return
			}
			if !errors.Is(err, ErrTokenSpansTruncated) {
				t.Fatalf("AnalyzeContentForPII: err = %v, want the truncation sentinel", err)
			}
			if !hasPII {
				t.Fatal("AnalyzeContentForPII: partial spans were discarded")
			}
		})
	}
}

func assertTruncationOutcome(t *testing.T, api string, wantErr bool, err error, found, wantFound bool) {
	t.Helper()
	if err != nil && !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("%s: unexpected error: %v", api, err)
	}
	if wantErr {
		if err == nil {
			t.Fatalf("%s: want an error under on_error: block", api)
		}
		if found {
			t.Fatalf("%s: block must not report detections from a refused scan", api)
		}
		return
	}
	// Under allow the truncation sentinel travels with the results rather than
	// replacing them, so a caller can tell a partial scan from a clean one.
	if err == nil {
		t.Fatalf("%s: a truncated scan must surface ErrTokenSpansTruncated", api)
	}
	if found != wantFound {
		t.Fatalf("%s: found=%v, want %v; the spans before the cut are valid", api, found, wantFound)
	}
}
