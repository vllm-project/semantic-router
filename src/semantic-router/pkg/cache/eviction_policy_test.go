package cache

import (
	"testing"
	"time"
)

func TestFIFOPolicy(t *testing.T) {
	policy := &FIFOPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", Timestamp: now.Add(-3 * time.Second)},
		{Query: "query2", Timestamp: now.Add(-1 * time.Second)},
		{Query: "query3", Timestamp: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim index 0 (oldest), got %d", victim)
	}
}

func TestLRUPolicy(t *testing.T) {
	policy := &LRUPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query2", LastAccessAt: now.Add(-1 * time.Second)},
		{Query: "query3", LastAccessAt: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim index 0 (least recently used), got %d", victim)
	}
}

func TestLFUPolicy(t *testing.T) {
	policy := &LFUPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", HitCount: 5, LastAccessAt: now.Add(-2 * time.Second)},
		{Query: "query2", HitCount: 1, LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query3", HitCount: 3, LastAccessAt: now.Add(-1 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim index 1 (least frequently used), got %d", victim)
	}
}

func TestLFUPolicyTiebreaker(t *testing.T) {
	policy := &LFUPolicy{}

	// Test tiebreaker: same frequency, choose least recently used
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", HitCount: 2, LastAccessAt: now.Add(-1 * time.Second)},
		{Query: "query2", HitCount: 2, LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query3", HitCount: 5, LastAccessAt: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim index 1 (LRU tiebreaker), got %d", victim)
	}
}
