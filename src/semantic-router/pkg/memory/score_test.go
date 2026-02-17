package memory

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetentionScore_ZeroTime(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now
	R := RetentionScore(lastAccessed, 0, 30, now)
	assert.InDelta(t, 1.0, R, 1e-9)
}

func TestRetentionScore_DefaultS0(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(-15 * 24 * time.Hour)
	R := RetentionScore(lastAccessed, 0, 0, now)
	expect := math.Exp(-15.0 / 30.0)
	assert.InDelta(t, expect, R, 1e-9)
}

func TestRetentionScore_HalfLife(t *testing.T) {
	S0 := 30
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(-30 * 24 * time.Hour)
	R := RetentionScore(lastAccessed, 0, S0, now)
	assert.InDelta(t, math.Exp(-1), R, 1e-6)
}

func TestRetentionScore_AccessCountIncreasesStrength(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(-30 * 24 * time.Hour)
	R0 := RetentionScore(lastAccessed, 0, 30, now)
	R10 := RetentionScore(lastAccessed, 10, 30, now)
	R100 := RetentionScore(lastAccessed, 100, 30, now)
	assert.Less(t, R0, R10)
	assert.Less(t, R10, R100)
	assert.InDelta(t, math.Exp(-30.0/40.0), R10, 1e-6)
}

func TestRetentionScore_NegativeTimeClampedToZero(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(1 * time.Hour)
	R := RetentionScore(lastAccessed, 0, 30, now)
	assert.InDelta(t, 1.0, R, 1e-9)
}

func TestRetentionScore_FractionalDays(t *testing.T) {
	// 12 hours ago → tDays = 0.5, verifies sub-day precision in the decay formula
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(-12 * time.Hour)
	R := RetentionScore(lastAccessed, 0, 30, now)
	assert.InDelta(t, math.Exp(-0.5/30.0), R, 1e-9)
}

func TestRetentionScore_ZeroStrengthFallback(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	lastAccessed := now.Add(-10 * 24 * time.Hour)
	R := RetentionScore(lastAccessed, -30, 30, now)
	assert.InDelta(t, math.Exp(-10), R, 1e-6)
}

func TestRetentionScore_ZeroValueLastAccessed(t *testing.T) {
	// Pre-existing memories without last_accessed would have time.Time{} (year 0001).
	// Without the ListForPrune fallback, this would give R ≈ 0 and cause immediate pruning.
	// This test documents the raw behavior — ListForPrune guards against this at the Milvus layer.
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	zeroTime := time.Time{}
	R := RetentionScore(zeroTime, 0, 30, now)
	// ~740,000 days since year 0001 → R is effectively 0
	assert.Less(t, R, 1e-100, "Zero-value LastAccessed produces R ≈ 0 (protected by ListForPrune fallback)")
}

func TestRetentionScore_UsesNamedConstants(t *testing.T) {
	// Verify the named constants are the expected values
	assert.Equal(t, 30, DefaultInitialStrengthDays)
	assert.InDelta(t, 0.1, DefaultPruneThreshold, 1e-9)
	assert.InDelta(t, 1.0, MinStrength, 1e-9)
}

func TestPruneCandidates_Empty(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	ids := PruneCandidates(nil, now, 30, 0.1, 0)
	assert.Empty(t, ids)
	ids = PruneCandidates([]MemoryPruneEntry{}, now, 30, 0.1, 0)
	assert.Empty(t, ids)
}

func TestPruneCandidates_BelowThreshold(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	entries := []MemoryPruneEntry{
		{ID: "keep", LastAccessed: now.Add(-60 * 24 * time.Hour), AccessCount: 0},
		{ID: "drop", LastAccessed: now.Add(-100 * 24 * time.Hour), AccessCount: 0},
	}
	ids := PruneCandidates(entries, now, 30, 0.1, 0)
	require.Len(t, ids, 1)
	assert.Equal(t, "drop", ids[0])
}

func TestPruneCandidates_MaxMemoriesPerUser(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	entries := []MemoryPruneEntry{
		{ID: "a", LastAccessed: now.Add(-5 * 24 * time.Hour), AccessCount: 0},
		{ID: "b", LastAccessed: now.Add(-20 * 24 * time.Hour), AccessCount: 0},
		{ID: "c", LastAccessed: now.Add(-40 * 24 * time.Hour), AccessCount: 0},
	}
	ids := PruneCandidates(entries, now, 30, 0.1, 2)
	require.Len(t, ids, 1)
	assert.Equal(t, "c", ids[0])
}

func TestPruneCandidates_ThresholdAndCap(t *testing.T) {
	now := time.Date(2026, 2, 10, 12, 0, 0, 0, time.UTC)
	entries := []MemoryPruneEntry{
		{ID: "drop_weak", LastAccessed: now.Add(-90 * 24 * time.Hour), AccessCount: 0},
		{ID: "keep1", LastAccessed: now.Add(-1 * 24 * time.Hour), AccessCount: 0},
		{ID: "keep2", LastAccessed: now.Add(-2 * 24 * time.Hour), AccessCount: 0},
		{ID: "drop_low_r1", LastAccessed: now.Add(-25 * 24 * time.Hour), AccessCount: 0},
		{ID: "drop_low_r2", LastAccessed: now.Add(-30 * 24 * time.Hour), AccessCount: 0},
	}
	ids := PruneCandidates(entries, now, 30, 0.1, 2)
	require.Len(t, ids, 3)
	assert.Contains(t, ids, "drop_weak")
	assert.Contains(t, ids, "drop_low_r1")
	assert.Contains(t, ids, "drop_low_r2")
}
