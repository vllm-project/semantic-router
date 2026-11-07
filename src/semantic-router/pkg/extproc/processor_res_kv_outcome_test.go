package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestKVTransferOutcomeRecordedOnce(t *testing.T) {
	for _, tc := range []struct{ status, reason, wantStatus, wantReason string }{
		{"applied", "loaded", "applied", "loaded"},
		{"fallback_reprefill", "load_failed", "fallback_reprefill", "load_failed"},
		{"", "", "unsupported", "no_result"},
		{"arbitrary", "tenant-data", "unsupported", "unknown"},
	} {
		t.Run(tc.status+tc.reason, func(t *testing.T) {
			counter := metrics.KVTransferOutcomes.WithLabelValues(tc.wantStatus, tc.wantReason)
			before := testutil.ToFloat64(counter)
			ctx := &RequestContext{KVTransferAttempted: true}
			headerMap := &core.HeaderMap{}
			if tc.status != "" {
				headerMap.Headers = append(headerMap.Headers, &core.HeaderValue{Key: headers.VSRKVTransferStatus, RawValue: []byte(tc.status)}, &core.HeaderValue{Key: headers.VSRKVTransferReason, Value: tc.reason})
			}
			observeKVTransferResponse(ctx, headerMap)
			finishRequestTrace(ctx, nil)
			observeKVTransferResponse(ctx, headerMap)
			if got := testutil.ToFloat64(counter) - before; got != 1 {
				t.Fatalf("outcomes=%g", got)
			}
		})
	}
	ctx := &RequestContext{}
	recordKVTransferOutcome(ctx, "applied", "loaded")
	if ctx.KVTransferOutcomeRecorded {
		t.Fatal("a request without a handoff was counted")
	}
}
