package memory

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGroupBySimilarity(t *testing.T) {
	now := time.Now()
	memories := []*Memory{
		{ID: "m1", Content: "User budget for Hawaii trip is $10,000", CreatedAt: now},
		{ID: "m2", Content: "User budget for Hawaii trip is ten thousand dollars", CreatedAt: now},
		{ID: "m3", Content: "User prefers direct flights to Hawaii", CreatedAt: now},
		{ID: "m4", Content: "User prefers non-stop direct flights to Hawaii islands", CreatedAt: now},
		{ID: "m5", Content: "The weather in Paris is mild in spring", CreatedAt: now},
	}

	groups := groupBySimilarity(memories, 0.50)

	// m1 and m2 share many words -> group
	// m3 and m4 share many words -> group
	// m5 is distinct -> singleton
	assert.True(t, len(groups) >= 2, "should have at least 2 groups (distinct topics)")

	foundBudgetGroup := false
	foundFlightGroup := false
	for _, g := range groups {
		ids := make(map[string]bool)
		for _, m := range g {
			ids[m.ID] = true
		}
		if ids["m1"] && ids["m2"] {
			foundBudgetGroup = true
		}
		if ids["m3"] && ids["m4"] {
			foundFlightGroup = true
		}
	}
	assert.True(t, foundBudgetGroup, "budget memories should be grouped together")
	assert.True(t, foundFlightGroup, "flight memories should be grouped together")
}

func TestMergeGroup(t *testing.T) {
	group := []*Memory{
		{Content: "Q: What is my budget?\nA: Your budget is $10,000"},
		{Content: "Q: What is my budget?\nA: Your Hawaii budget is $10,000"},
	}

	merged := mergeGroup(group)
	assert.Contains(t, merged, "Q: What is my budget?")
	assert.Contains(t, merged, "A: Your budget is $10,000")
	assert.Contains(t, merged, "A: Your Hawaii budget is $10,000")

	// Dedup: "Q: What is my budget?" should appear only once
	count := 0
	for _, line := range splitLines(merged) {
		if line == "Q: What is my budget?" {
			count++
		}
	}
	assert.Equal(t, 1, count, "duplicate lines should be deduplicated")
}

func splitLines(s string) []string {
	var lines []string
	for _, l := range []byte(s) {
		if l == '\n' {
			lines = append(lines, "")
		}
	}
	// Simple split for test
	result := make([]string, 0)
	current := ""
	for _, c := range s {
		if c == '\n' {
			result = append(result, current)
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		result = append(result, current)
	}
	return result
}

func TestEarliestCreatedAt(t *testing.T) {
	t1 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	t2 := time.Date(2025, 6, 1, 0, 0, 0, 0, time.UTC)
	t3 := time.Date(2024, 12, 1, 0, 0, 0, 0, time.UTC)

	group := []*Memory{
		{CreatedAt: t1},
		{CreatedAt: t2},
		{CreatedAt: t3},
	}

	assert.Equal(t, t3, earliestCreatedAt(group))
}

func TestMaxImportance(t *testing.T) {
	group := []*Memory{
		{Importance: 0.3},
		{Importance: 0.9},
		{Importance: 0.5},
	}
	assert.InDelta(t, 0.9, maxImportance(group), 0.01)
}
