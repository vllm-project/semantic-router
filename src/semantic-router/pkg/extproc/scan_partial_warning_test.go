package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// A routed request whose guard read only part of a long input, or none of
// it, tells every caller with scan_partial in x-vsr-warnings; the counts stay
// in the trace. A request read whole carries no warning.
func TestAPartlyReadGuardWarnsEveryCallerWithScanPartial(t *testing.T) {
	for coverage, want := range map[string]string{"partial 117/583": "scan_partial", "none": "scan_partial", "full": ""} {
		ctx := &RequestContext{
			VSRSelectedModel: "deepseek",
			VSRScanCoverage:  map[string]string{"jailbreak:prompt_attack": coverage, "safety:unsafe_request": "full"},
		}
		got, _ := headerValue(buildResponseHeaderMutation(ctx).GetSetHeaders(), headers.VSRWarnings)
		if got != want {
			t.Fatalf("coverage %q: %s = %q, want %q", coverage, headers.VSRWarnings, got, want)
		}
	}
}
