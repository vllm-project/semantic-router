package testcases

import (
	"encoding/json"
	"testing"
)

func TestProviderCacheCostAssertions(t *testing.T) {
	const fixture = `{
		"actual_cost":0.00057,"cost_savings":0,
		"provider_cache":{
			"state":"complete",
			"uncached_input":{"tokens":10,"provenance":"authoritative","cost":0.00002},
			"cache_read":{"tokens":10,"provenance":"authoritative","cost":0.00001},
			"cache_write":{"tokens":80,"provenance":"authoritative","cost":0.00052},
			"output":{"tokens":2,"provenance":"authoritative","cost":0.00002},
			"cache_write_5m":{"tokens":30},"cache_write_1h":{"tokens":50},
			"no_cache_cost":0.00022,"cache_savings":-0.00035
		}
	}`
	var record providerCacheReplayRecord
	if err := json.Unmarshal([]byte(fixture), &record); err != nil {
		t.Fatal(err)
	}
	if err := assertProviderCacheCosts(record, true); err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []func(*providerCacheReplayRecord){
		func(r *providerCacheReplayRecord) { r.ProviderCache.CacheSavings = nil },
		func(r *providerCacheReplayRecord) { *r.ProviderCache.CacheSavings = 0 },
		func(r *providerCacheReplayRecord) { r.ProviderCache.CacheRead.Tokens = nil },
		func(r *providerCacheReplayRecord) { r.ProviderCache.Write1h.Tokens = nil },
		func(r *providerCacheReplayRecord) { r.ProviderCache.State = "partial" },
	} {
		var broken providerCacheReplayRecord
		if err := json.Unmarshal([]byte(fixture), &broken); err != nil {
			t.Fatal(err)
		}
		mutation(&broken)
		if err := assertProviderCacheCosts(broken, true); err == nil {
			t.Fatal("incorrect cache accounting passed E2E assertions")
		}
	}
}
