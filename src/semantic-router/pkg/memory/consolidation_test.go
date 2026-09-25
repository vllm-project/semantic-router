package memory

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	assert.GreaterOrEqual(t, len(groups), 2, "should have at least 2 groups (distinct topics)")

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

func TestConsolidateUserDoesNotMergeAcrossProjects(t *testing.T) {
	// Exact-head style repro: near-identical content in two projects must stay
	// project-scoped. Before the fix, both originals were deleted and the
	// replacement had ProjectID == "".
	contentA := "User prefers morning standups for the alpha roadmap"
	contentB := "User prefers morning standups for the alpha project plan"
	store := newScriptMemoryStore(
		&Memory{
			ID: "proj-a", UserID: "user-1", ProjectID: "project-a",
			Type: MemoryTypeSemantic, Content: contentA, CreatedAt: time.Now(),
		},
		&Memory{
			ID: "proj-b", UserID: "user-1", ProjectID: "project-b",
			Type: MemoryTypeSemantic, Content: contentB, CreatedAt: time.Now(),
		},
	)

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 0, merged)
	require.Equal(t, 0, deleted)
	require.Equal(t, 0, store.stores)
	require.Equal(t, 0, store.forgets)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 2)
	byID := map[string]*Memory{}
	for _, mem := range store.memories {
		byID[mem.ID] = mem
	}
	require.Equal(t, "project-a", byID["proj-a"].ProjectID)
	require.Equal(t, "project-b", byID["proj-b"].ProjectID)
}

func TestConsolidateUserDoesNotMergeAcrossTypes(t *testing.T) {
	content := "Deploy payment-service with npm build then docker push"
	store := newScriptMemoryStore(
		&Memory{
			ID: "sem", UserID: "user-1", ProjectID: "shared",
			Type: MemoryTypeSemantic, Content: content + " semantic note", CreatedAt: time.Now(),
		},
		&Memory{
			ID: "proc", UserID: "user-1", ProjectID: "shared",
			Type: MemoryTypeProcedural, Content: content + " procedural steps", CreatedAt: time.Now(),
		},
	)

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 0, merged)
	require.Equal(t, 0, deleted)
	require.Len(t, store.memories, 2)
}

func TestConsolidateUserPreservesProjectOnMerge(t *testing.T) {
	store := newScriptMemoryStore(
		&Memory{
			ID: "a1", UserID: "user-1", ProjectID: "project-a",
			Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now(),
		},
		&Memory{
			ID: "a2", UserID: "user-1", ProjectID: "project-a",
			Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now(),
		},
		&Memory{
			ID: "b1", UserID: "user-1", ProjectID: "project-b",
			Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now(),
		},
		&Memory{
			ID: "b2", UserID: "user-1", ProjectID: "project-b",
			Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now(),
		},
	)

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 2, merged)
	require.Equal(t, 4, deleted)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 2)
	projects := map[string]int{}
	for _, mem := range store.memories {
		require.NotEmpty(t, mem.ProjectID)
		require.Equal(t, MemoryTypeSemantic, mem.Type)
		require.Equal(t, "consolidation", mem.Source)
		projects[mem.ProjectID]++
	}
	require.Equal(t, 1, projects["project-a"])
	require.Equal(t, 1, projects["project-b"])
}
