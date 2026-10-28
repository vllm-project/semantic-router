package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func recordKVTransferOutcome(ctx *RequestContext, status, reason string) {
	if ctx == nil || !ctx.KVTransferAttempted || ctx.KVTransferOutcomeRecorded {
		return
	}
	switch status {
	case headers.KVTransferStatusApplied, headers.KVTransferStatusFallbackReprefill, headers.KVTransferStatusUnsupported:
	default:
		status = headers.KVTransferStatusUnsupported
	}
	switch reason {
	case "loaded", "connector_unavailable", "snapshot_unavailable", "snapshot_missing", "snapshot_expired", "snapshot_invalid", "prefix_incompatible", "unknown_mapper", "load_failed", "request_incomplete", "no_result":
	default:
		reason = "unknown"
	}
	metrics.KVTransferOutcomes.WithLabelValues(status, reason).Inc()
	ctx.KVTransferOutcomeRecorded = true
}

func observeKVTransferResponse(ctx *RequestContext, headerMap *core.HeaderMap) {
	status, reason := headers.KVTransferStatusUnsupported, "no_result"
	for _, header := range headerMap.GetHeaders() {
		switch strings.ToLower(header.Key) {
		case headers.VSRKVTransferStatus:
			status = header.Value
			if len(header.RawValue) > 0 {
				status = string(header.RawValue)
			}
		case headers.VSRKVTransferReason:
			reason = header.Value
			if len(header.RawValue) > 0 {
				reason = string(header.RawValue)
			}
		}
	}
	recordKVTransferOutcome(ctx, status, reason)
}
