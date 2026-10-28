package modelpricing

import (
	"math"
	"testing"
)

func TestNormalizeCreatesDisjointInputBuckets(t *testing.T) {
	got := Normalize(Usage{
		PromptTokens:      100,
		CachedInputTokens: 40,
		CacheWriteTokens:  25,
		CompletionTokens:  10,
	})
	want := Breakdown{
		PromptTokens:        100,
		StandardInputTokens: 35,
		CachedInputTokens:   40,
		CacheWriteTokens:    25,
		CompletionTokens:    10,
	}
	if got != want {
		t.Fatalf("Normalize() = %#v, want %#v", got, want)
	}
}

func TestNormalizeClampsOverreportedDetails(t *testing.T) {
	got := Normalize(Usage{
		PromptTokens:      100,
		CachedInputTokens: 80,
		CacheWriteTokens:  70,
		CompletionTokens:  -1,
	})
	if got.CachedInputTokens != 80 || got.CacheWriteTokens != 20 || got.StandardInputTokens != 0 || got.CompletionTokens != 0 {
		t.Fatalf("Normalize() did not clamp detail counts: %#v", got)
	}
}

func TestCostUsesDistinctInputRates(t *testing.T) {
	cacheWriteRate := 6.25
	got := Cost(
		Usage{PromptTokens: 1_000, CachedInputTokens: 200, CacheWriteTokens: 300, CompletionTokens: 100},
		Rates{
			Currency:         "USD",
			PromptPer1M:      5,
			CachedInputPer1M: 0.5,
			CacheWritePer1M:  &cacheWriteRate,
			CompletionPer1M:  30,
		},
	)
	want := (500*5.0 + 200*0.5 + 300*6.25 + 100*30.0) / 1_000_000.0
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want %.12f", got, want)
	}
}

func TestCostFallsBackToPromptRateForCacheWrites(t *testing.T) {
	got := Cost(
		Usage{PromptTokens: 1_000, CacheWriteTokens: 1_000},
		Rates{PromptPer1M: 5},
	)
	if math.Abs(got-0.005) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want 0.005", got)
	}
}

func TestCostPreservesExplicitFreeCacheWrites(t *testing.T) {
	free := 0.0
	got := Cost(
		Usage{PromptTokens: 1_000, CacheWriteTokens: 1_000},
		Rates{PromptPer1M: 5, CacheWritePer1M: &free},
	)
	if got != 0 {
		t.Fatalf("Cost() = %.12f, want 0", got)
	}
}

func TestInputCostMultiplierAccountsForPremiumCacheWrites(t *testing.T) {
	cacheWriteRate := 6.25
	rates := Rates{PromptPer1M: 5, CachedInputPer1M: 0.5, CacheWritePer1M: &cacheWriteRate}

	if got := InputCostMultiplier(Usage{PromptTokens: 100, CacheWriteTokens: 100}, rates); got != 1 {
		t.Fatalf("all-cache-write multiplier = %v, want 1", got)
	}
	if got := InputCostMultiplier(Usage{PromptTokens: 100, CachedInputTokens: 100}, rates); math.Abs(got-0.08) > 1e-12 {
		t.Fatalf("all-cached multiplier = %v, want 0.08", got)
	}
}

func TestNormalizeTieredCacheWrites(t *testing.T) {
	got := Normalize(Usage{
		PromptTokens:       100,
		CachedInputTokens:  40,
		CacheWriteTokens:   50,
		CacheWrite5mTokens: 20,
		CacheWrite1hTokens: 30,
		CompletionTokens:   10,
	})
	want := Breakdown{
		PromptTokens:        100,
		StandardInputTokens: 10,
		CachedInputTokens:   40,
		CacheWriteTokens:    50,
		CacheWrite5mTokens:  20,
		CacheWrite1hTokens:  30,
		CompletionTokens:    10,
	}
	if got != want {
		t.Fatalf("Normalize() = %#v, want %#v", got, want)
	}

	// Clamping when tiers exceed allocated cache write tokens
	clamped := Normalize(Usage{
		PromptTokens:       100,
		CacheWriteTokens:   50,
		CacheWrite5mTokens: 40,
		CacheWrite1hTokens: 30,
	})
	if clamped.CacheWriteTokens != 50 || clamped.CacheWrite5mTokens != 40 || clamped.CacheWrite1hTokens != 10 {
		t.Fatalf("Normalize() did not clamp overreported tiers: %#v", clamped)
	}
}

func TestCostUsesTieredCacheWriteRates(t *testing.T) {
	rate5m := 3.75
	rate1h := 6.25
	got := Cost(
		Usage{
			PromptTokens:       1_000,
			CacheWriteTokens:   600,
			CacheWrite5mTokens: 200,
			CacheWrite1hTokens: 400,
			CompletionTokens:   100,
		},
		Rates{
			PromptPer1M:       3.0,
			CacheWritePer1M:   &rate5m,
			CacheWrite1hPer1M: &rate1h,
			CompletionPer1M:   15.0,
		},
	)
	// (400 standard * 3.0 + 200 * 3.75 + 400 * 6.25 + 100 * 15.0) / 1_000_000
	want := (400*3.0 + 200*3.75 + 400*6.25 + 100*15.0) / 1_000_000.0
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want %.12f", got, want)
	}
}

func TestCostTierFallbackToAggregateAndPrompt(t *testing.T) {
	rateAggregate := 4.5
	// 1h falls back to aggregate rate (which covers 5m)
	got := Cost(
		Usage{
			PromptTokens:       1_000,
			CacheWriteTokens:   600,
			CacheWrite5mTokens: 200,
			CacheWrite1hTokens: 400,
		},
		Rates{
			PromptPer1M:     2.0,
			CacheWritePer1M: &rateAggregate,
		},
	)
	// (400 standard * 2.0 + 600 * 4.5) / 1_000_000
	want := (400*2.0 + 600*4.5) / 1_000_000.0
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want %.12f", got, want)
	}

	// When neither tier nor aggregate configured, both fall back to PromptPer1M
	gotPromptFallback := Cost(
		Usage{
			PromptTokens:       1_000,
			CacheWriteTokens:   600,
			CacheWrite5mTokens: 200,
			CacheWrite1hTokens: 400,
		},
		Rates{
			PromptPer1M: 5.0,
		},
	)
	wantPromptFallback := (1_000 * 5.0) / 1_000_000.0
	if math.Abs(gotPromptFallback-wantPromptFallback) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want %.12f", gotPromptFallback, wantPromptFallback)
	}
}

func TestCostHandlesPartialTierAllocationWithoutDoubleCounting(t *testing.T) {
	rateAgg := 5.0
	rate1h := 8.0
	// CacheWriteTokens is 500, but only 200 are attributed to 1h (300 remaining 5m/aggregate)
	got := Cost(
		Usage{
			PromptTokens:       1_000,
			CacheWriteTokens:   500,
			CacheWrite1hTokens: 200,
		},
		Rates{
			PromptPer1M:       2.0,
			CacheWritePer1M:   &rateAgg,
			CacheWrite1hPer1M: &rate1h,
		},
	)
	// 500 standard * 2.0 + 200 (1h) * 8.0 + 300 (5m/agg) * 5.0
	want := (500*2.0 + 200*8.0 + 300*5.0) / 1_000_000.0
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Cost() = %.12f, want %.12f", got, want)
	}
}
