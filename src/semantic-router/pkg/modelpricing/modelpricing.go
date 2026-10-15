// Package modelpricing calculates token costs from provider-reported usage.
package modelpricing

const tokensPerMillion = 1_000_000.0

// Rates contains the active per-million-token prices for one model.
type Rates struct {
	Currency          string
	PromptPer1M       float64
	CachedInputPer1M  float64
	CacheWritePer1M   *float64
	CacheWrite1hPer1M *float64
	CompletionPer1M   float64
}

// IsConfigured reports whether the model has an explicit pricing entry.
func (r Rates) IsConfigured() bool {
	return r.Currency != "" ||
		r.PromptPer1M != 0 ||
		r.CachedInputPer1M != 0 ||
		r.CacheWritePer1M != nil ||
		r.CacheWrite1hPer1M != nil ||
		r.CompletionPer1M != 0
}

// EffectiveCacheWritePer1M preserves legacy accounting by using the normal
// prompt rate when no distinct cache-write rate is configured. This covers
// the default 5-minute ephemeral write retention tier.
func (r Rates) EffectiveCacheWritePer1M() float64 {
	if r.CacheWritePer1M == nil {
		return nonNegativeRate(r.PromptPer1M)
	}
	return nonNegativeRate(*r.CacheWritePer1M)
}

// EffectiveCacheWrite1hPer1M falls back to the aggregate cache-write rate
// (or normal prompt rate) when no distinct 1-hour cache-write rate is configured.
func (r Rates) EffectiveCacheWrite1hPer1M() float64 {
	if r.CacheWrite1hPer1M == nil {
		return r.EffectiveCacheWritePer1M()
	}
	return nonNegativeRate(*r.CacheWrite1hPer1M)
}

// Usage contains one response's token counts. PromptTokens includes cached
// reads and cache writes; the detail counts are normalized into disjoint
// buckets before costs are calculated.
type Usage struct {
	PromptTokens       int
	CachedInputTokens  int
	CacheWriteTokens   int
	CacheWrite5mTokens int
	CacheWrite1hTokens int
	CompletionTokens   int
}

// Breakdown is the normalized, mutually exclusive token allocation.
type Breakdown struct {
	PromptTokens        int
	StandardInputTokens int
	CachedInputTokens   int
	CacheWriteTokens    int
	CacheWrite5mTokens  int
	CacheWrite1hTokens  int
	CompletionTokens    int
}

// Normalize clamps malformed detail counts while preserving the prompt total.
// Cached reads take precedence, then cache writes consume the remaining input.
// Tier allocations are partitioned within the normalized cache writes so they
// cannot double count or exceed total cache writes.
func Normalize(usage Usage) Breakdown {
	prompt := maxInt(usage.PromptTokens, 0)
	cached := clampInt(usage.CachedInputTokens, 0, prompt)
	rawWrite := usage.CacheWriteTokens
	if rawWrite == 0 && (usage.CacheWrite5mTokens > 0 || usage.CacheWrite1hTokens > 0) {
		rawWrite = usage.CacheWrite5mTokens + usage.CacheWrite1hTokens
	}
	cacheWrite := clampInt(rawWrite, 0, prompt-cached)
	w5m := clampInt(usage.CacheWrite5mTokens, 0, cacheWrite)
	w1h := clampInt(usage.CacheWrite1hTokens, 0, cacheWrite-w5m)

	return Breakdown{
		PromptTokens:        prompt,
		StandardInputTokens: prompt - cached - cacheWrite,
		CachedInputTokens:   cached,
		CacheWriteTokens:    cacheWrite,
		CacheWrite5mTokens:  w5m,
		CacheWrite1hTokens:  w1h,
		CompletionTokens:    maxInt(usage.CompletionTokens, 0),
	}
}

// Cost returns the total response cost in the rates' configured currency.
func Cost(usage Usage, rates Rates) float64 {
	breakdown := Normalize(usage)
	writeCost := float64(breakdown.CacheWrite1hTokens)*rates.EffectiveCacheWrite1hPer1M() +
		float64(breakdown.CacheWriteTokens-breakdown.CacheWrite1hTokens)*rates.EffectiveCacheWritePer1M()

	return (float64(breakdown.StandardInputTokens)*nonNegativeRate(rates.PromptPer1M) +
		float64(breakdown.CachedInputTokens)*nonNegativeRate(rates.CachedInputPer1M) +
		writeCost +
		float64(breakdown.CompletionTokens)*nonNegativeRate(rates.CompletionPer1M)) / tokensPerMillion
}

// InputCostMultiplier normalizes observed input cost against the highest
// configured input-token rate. The result remains in [0, 1], including when
// cache writes cost more than ordinary input.
func InputCostMultiplier(usage Usage, rates Rates) float64 {
	breakdown := Normalize(usage)
	if breakdown.PromptTokens == 0 {
		return 0
	}

	promptRate := nonNegativeRate(rates.PromptPer1M)
	cachedRate := nonNegativeRate(rates.CachedInputPer1M)
	cacheWriteRate := rates.EffectiveCacheWritePer1M()
	cacheWrite1hRate := rates.EffectiveCacheWrite1hPer1M()
	maxInputRate := maxFloat(promptRate, maxFloat(cachedRate, maxFloat(cacheWriteRate, cacheWrite1hRate)))
	if maxInputRate == 0 {
		return 0
	}

	writeCost := float64(breakdown.CacheWrite1hTokens)*cacheWrite1hRate +
		float64(breakdown.CacheWriteTokens-breakdown.CacheWrite1hTokens)*cacheWriteRate

	actual := float64(breakdown.StandardInputTokens)*promptRate +
		float64(breakdown.CachedInputTokens)*cachedRate +
		writeCost
	return actual / (float64(breakdown.PromptTokens) * maxInputRate)
}

func nonNegativeRate(value float64) float64 {
	if value < 0 {
		return 0
	}
	return value
}

func clampInt(value, minimum, maximum int) int {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func maxFloat(left, right float64) float64 {
	if left > right {
		return left
	}
	return right
}
