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
