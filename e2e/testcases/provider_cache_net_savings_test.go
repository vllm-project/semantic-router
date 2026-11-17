package testcases

import (
	"encoding/json"
	"testing"
)

func TestProviderCacheNetSavingsAssertions(t *testing.T) {
	const fixture = `{
		"currency":"USD","no_cache_spend":0.00044,"actual_spend":0.0007,"net_savings":-0.00026,
		"priced_records":2,"excluded_records":0,
		"by_decision":[{"name":"provider_cache_cost","currency":"USD","no_cache_spend":0.00044,"actual_spend":0.0007,"net_savings":-0.00026,"priced_records":2}],
		"by_selected_model":[{"name":"MoM","currency":"USD","no_cache_spend":0.00044,"actual_spend":0.0007,"net_savings":-0.00026,"priced_records":2}]
	}`
	decode := func() providerCacheNetSummary {
		var summary providerCacheNetSummary
		if err := json.Unmarshal([]byte(fixture), &summary); err != nil {
			t.Fatal(err)
		}
		return summary
	}
	if err := assertProviderCacheNetSummary(decode()); err != nil {
		t.Fatal(err)
	}
	for name, mutation := range map[string]func(*providerCacheNetSummary){
		"clamped sum":        func(s *providerCacheNetSummary) { s.NetSavings = .00009 },
		"excluded record":    func(s *providerCacheNetSummary) { s.ExcludedRecords = 1 },
		"missing decision":   func(s *providerCacheNetSummary) { s.ByDecision = nil },
		"split disagrees":    func(s *providerCacheNetSummary) { s.BySelectedModel[0].NetSavings = 0 },
		"one priced request": func(s *providerCacheNetSummary) { s.PricedRecords = 1 },
	} {
		summary := decode()
		mutation(&summary)
		if err := assertProviderCacheNetSummary(summary); err == nil {
			t.Fatalf("%s passed the E2E assertions", name)
		}
	}
}
