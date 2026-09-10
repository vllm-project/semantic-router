package classification

import (
	"context"
	"slices"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A provider-declared truncation carries valid spans for the part it saw. The
// routing signal has always kept them; the detection and masking APIs used to
// discard every detection in the call, including those from chunks that
// classified cleanly. All three now read classifier.pii.on_error: allow keeps
// what was found, block refuses the scan rather than reporting a partial
// result as complete.
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
			mockModel.setMockResponse(text, []candle_binding.TokenEntity{email}, ErrTokenSpansTruncated)

			detections, err := classifier.ClassifyPIIWithDetails(context.Background(), text)
			assertTruncationOutcome(t, "ClassifyPIIWithDetails", tc.wantErr, err, len(detections) > 0, tc.wantEntities)

			types, err := classifier.ClassifyPII(context.Background(), text)
			assertTruncationOutcome(t, "ClassifyPII", tc.wantErr, err, slices.Contains(types, "EMAIL_ADDRESS"), tc.wantEntities)

			hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{text})
			// AnalyzeContentForPII tolerates per-item failures as long as one
			// item classified, so under block it reports no PII for the only
			// item rather than returning an error.
			if tc.wantErr {
				if hasPII || len(results) != 0 {
					t.Fatalf("AnalyzeContentForPII under block: hasPII=%v results=%d, want the item dropped", hasPII, len(results))
				}
				if err == nil {
					t.Fatal("AnalyzeContentForPII under block: want an error when nothing could be classified")
				}
				return
			}
			if err != nil {
				t.Fatalf("AnalyzeContentForPII: %v", err)
			}
			if !hasPII {
				t.Fatal("AnalyzeContentForPII: partial spans were discarded")
			}
		})
	}
}

func assertTruncationOutcome(t *testing.T, api string, wantErr bool, err error, found, wantFound bool) {
	t.Helper()
	if wantErr {
		if err == nil {
			t.Fatalf("%s: want an error under on_error: block", api)
		}
		if !strings.Contains(err.Error(), "truncated") {
			t.Fatalf("%s: error does not name the truncation: %v", api, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("%s: %v", api, err)
	}
	if found != wantFound {
		t.Fatalf("%s: found=%v, want %v; the spans before the cut are valid", api, found, wantFound)
	}
}
