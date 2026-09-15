package extproc

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestResponseUsageSettlementPreservesCacheFieldPresence(t *testing.T) {
	one := int64(1)
	zero := int64(0)
	unknown := llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
	tests := []struct {
		name        string
		read        llmprotocol.TokenCount
		write       llmprotocol.TokenCount
		readSeen    bool
		writeSeen   bool
		wantState   string
		wantInvalid bool
	}{
		{name: "missing", read: unknown, write: unknown, wantState: "unavailable"},
		{name: "partial", read: llmprotocol.TokenCount{Value: &one, Provenance: llmprotocol.UsageAuthoritative}, write: unknown, readSeen: true, wantState: "partial"},
		{name: "explicit zero", read: llmprotocol.TokenCount{Value: &zero, Provenance: llmprotocol.UsageAuthoritative}, write: llmprotocol.TokenCount{Value: &zero, Provenance: llmprotocol.UsageAuthoritative}, readSeen: true, writeSeen: true, wantState: "authoritative"},
		{name: "derived cache is invalid", read: llmprotocol.TokenCount{Value: &one, Provenance: llmprotocol.UsageDerived}, write: unknown, wantInvalid: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			usage := responseUsageFromSemanticUsage(llmprotocol.Usage{
				State:          llmprotocol.UsageAvailable,
				InputTotal:     terminalTestCount(6),
				OutputTotal:    terminalTestCount(2),
				Total:          terminalTestCount(8),
				InputCacheRead: test.read, InputCacheWrite: test.write,
			})
			if usage.invalid != test.wantInvalid {
				t.Fatalf("invalid = %v, want %v: %+v", usage.invalid, test.wantInvalid, usage)
			}
			if test.wantInvalid {
				return
			}
			if usage.cachedPromptTokensReported != test.readSeen || usage.cacheWriteTokensReported != test.writeSeen {
				t.Fatalf("cache presence = read:%v write:%v, want read:%v write:%v", usage.cachedPromptTokensReported, usage.cacheWriteTokensReported, test.readSeen, test.writeSeen)
			}
			model := "provider-cache-presence-" + test.name
			before := testutil.ToFloat64(metrics.ModelPromptCacheUsageRecords.WithLabelValues(model, test.wantState))
			recordProviderPromptCacheUsage(model, usage)
			after := testutil.ToFloat64(metrics.ModelPromptCacheUsageRecords.WithLabelValues(model, test.wantState))
			if after != before+1 {
				t.Fatalf("provider cache state = %q was recorded %v times, want once", test.wantState, after-before)
			}
		})
	}
}

func TestResponseUsageRecordsOneProviderCacheMetricPerSettlementPath(t *testing.T) {
	model := "provider-cache-settlement-test"
	before := testutil.ToFloat64(metrics.ModelPromptCacheUsageRecords.WithLabelValues(model, "authoritative"))
	usage := responseUsageMetrics{
		promptTokens: 6, completionTokens: 2, totalTokens: 8,
		cachedPromptTokens: 1, cachedPromptTokensReported: true,
		cacheWriteTokens: 2, cacheWriteTokensReported: true,
	}
	router := &OpenAIRouter{}
	router.reportNonStreamingUsage(&RequestContext{RequestModel: model}, time.Second, usage)
	buffered := testutil.ToFloat64(metrics.ModelPromptCacheUsageRecords.WithLabelValues(model, "authoritative"))
	router.reportSemanticStreamingUsage(&RequestContext{RequestModel: model}, time.Second, usage)
	streaming := testutil.ToFloat64(metrics.ModelPromptCacheUsageRecords.WithLabelValues(model, "authoritative"))
	if buffered != before+1 || streaming != buffered+1 {
		t.Fatalf("provider cache records = before:%v buffered:%v streaming:%v, want one per path", before, buffered, streaming)
	}
}

func terminalTestCount(value int64) llmprotocol.TokenCount {
	return llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageAuthoritative}
}
