package store

type TokenCostComponent struct {
	Tokens     *int     `json:"tokens,omitempty"`
	Provenance string   `json:"provenance"`
	Cost       *float64 `json:"cost,omitempty"`
}

type ProviderCacheAccounting struct {
	State         string             `json:"state"`
	UncachedInput TokenCostComponent `json:"uncached_input"`
	CacheRead     TokenCostComponent `json:"cache_read"`
	CacheWrite    TokenCostComponent `json:"cache_write"`
	Output        TokenCostComponent `json:"output"`
	CacheWrite5m  TokenCostComponent `json:"cache_write_5m"`
	CacheWrite1h  TokenCostComponent `json:"cache_write_1h"`
	NoCacheCost   *float64           `json:"no_cache_cost,omitempty"`
	CacheSavings  *float64           `json:"cache_savings,omitempty"`
}

func cloneProviderCacheAccounting(value *ProviderCacheAccounting) *ProviderCacheAccounting {
	if value == nil {
		return nil
	}
	result := *value
	for _, component := range []*TokenCostComponent{
		&result.UncachedInput, &result.CacheRead, &result.CacheWrite, &result.Output,
		&result.CacheWrite5m, &result.CacheWrite1h,
	} {
		component.Tokens = cloneIntPtr(component.Tokens)
		component.Cost = cloneFloat64Ptr(component.Cost)
	}
	result.NoCacheCost = cloneFloat64Ptr(value.NoCacheCost)
	result.CacheSavings = cloneFloat64Ptr(value.CacheSavings)
	return &result
}
