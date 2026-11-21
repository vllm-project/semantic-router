package sessiontelemetry

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

func TestRouterSessionMemoryKeepsTheLatestCacheReceipt(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	at := time.Unix(1_800_000_000, 0)
	RecordSessionUsage(SessionUsageParams{
		SessionID: "sess-receipt", Model: "premium", PromptTokens: 20000, CachedPromptTokens: 18000,
		CacheWriteTokens: 1000, CacheWrite1hTokens: 1000, CompletionTokens: 500, CacheReported: true, Timestamp: at,
	})
	// An older response that completes late must not replace the newer receipt.
	RecordSessionUsage(SessionUsageParams{
		SessionID: "sess-receipt", Model: "cheap", PromptTokens: 10, CompletionTokens: 1, Timestamp: at.Add(-time.Second),
	})

	snapshot, ok := PeekRouterSessionSnapshot("sess-receipt", at)
	want := modelpricing.Receipt{
		Model: "premium",
		Usage: modelpricing.Usage{
			PromptTokens: 20000, CachedInputTokens: 18000, CacheWriteTokens: 1000,
			CacheWrite1hTokens: 1000, CompletionTokens: 500,
		},
		CacheReported: true,
		At:            at,
	}
	if !ok || snapshot.LastCacheReceipt == nil || *snapshot.LastCacheReceipt != want {
		t.Fatalf("last cache receipt = %+v, want %+v", snapshot.LastCacheReceipt, want)
	}

	payload, err := encodeRedisRouterSessionSnapshot(snapshot)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, found, err := decodeRedisRouterSessionSnapshot(payload, "sess-receipt")
	if err != nil || !found || decoded.LastCacheReceipt == nil || decoded.LastCacheReceipt.Usage != want.Usage ||
		!decoded.LastCacheReceipt.At.Equal(at) {
		t.Fatalf("shared store dropped the receipt: %+v %v %v", decoded.LastCacheReceipt, found, err)
	}

	older := snapshot
	older.LastCacheReceipt = &modelpricing.Receipt{Model: "cheap", At: at.Add(-time.Minute)}
	for _, merged := range []RouterSessionSnapshot{
		mergeRouterSessionSnapshots(older, snapshot),
		mergeRouterSessionSnapshots(snapshot, older),
	} {
		if merged.LastCacheReceipt == nil || merged.LastCacheReceipt.Model != "premium" {
			t.Fatalf("merge kept the older receipt: %+v", merged.LastCacheReceipt)
		}
	}
}
