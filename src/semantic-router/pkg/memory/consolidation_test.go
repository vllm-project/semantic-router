package memory

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	_ sourceReplacer = (*InMemoryStore)(nil)
	_ sourceReplacer = (*MilvusStore)(nil)
	_ sourceReplacer = (*QdrantStore)(nil)
	_ sourceReplacer = (*CachingStore)(nil)
	_ sourceReplacer = (*ValkeyStore)(nil)
	_ sourceReplacer = (*scriptMemoryStore)(nil)
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

func TestConsolidateUserSkipsMergeWhenSourceDeletedAfterList(t *testing.T) {
	store := newScriptMemoryStore(
		&Memory{ID: "a", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now()},
		&Memory{ID: "b", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now()},
	)
	store.afterList = func() {
		require.NoError(t, store.Forget(context.Background(), "a"))
	}

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 0, merged)
	require.Equal(t, 0, deleted)
	require.Equal(t, 0, store.stores)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 1)
	require.Equal(t, "b", store.memories[0].ID)
	require.NotContains(t, store.memories[0].Content, "alpha beta gamma\n")
	for _, mem := range store.memories {
		require.NotEqual(t, "consolidation", mem.Source)
	}
}

func TestConsolidateUserSkipsMergeWhenSourceUpdatedAfterList(t *testing.T) {
	updated := "budget is now 20000 dollars after the revision"
	store := newScriptMemoryStore(
		&Memory{ID: "a", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now()},
		&Memory{ID: "b", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now()},
	)
	store.afterList = func() {
		store.mu.Lock()
		defer store.mu.Unlock()
		for _, mem := range store.memories {
			if mem.ID == "a" {
				mem.Content = updated
				mem.UpdatedAt = time.Now()
			}
		}
	}

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 0, merged)
	require.Equal(t, 0, deleted)
	require.Equal(t, 0, store.stores)
	require.Equal(t, 0, store.forgets)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 2)
	for _, mem := range store.memories {
		require.NotEqual(t, "consolidation", mem.Source)
		if mem.ID == "a" {
			require.Equal(t, updated, mem.Content)
		}
	}
}

func TestConsolidateUserRollsBackSummaryWhenSourceChangesBeforeDelete(t *testing.T) {
	updated := "budget is now 20000 dollars after the revision"
	store := newScriptMemoryStore(
		&Memory{ID: "a", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now()},
		&Memory{ID: "b", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now()},
	)
	// The pre-delete Gets still see the listed versions. The update lands inside
	// the version-conditional delete, immediately before that compare.
	store.beforeSourceDelete = func(id string) {
		if id != "a" {
			return
		}
		store.mu.Lock()
		defer store.mu.Unlock()
		for _, mem := range store.memories {
			if mem.ID == "a" {
				mem.Content = updated
				mem.UpdatedAt = time.Now()
			}
		}
	}

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 0, merged)
	require.Equal(t, 0, deleted)
	require.Equal(t, 1, store.stores)
	require.Equal(t, 1, store.forgets)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 2)
	for _, mem := range store.memories {
		require.NotEqual(t, "consolidation", mem.Source)
		if mem.ID == "a" {
			require.Equal(t, updated, mem.Content)
		}
	}
}

func TestConsolidateUserKeepsPartialMergeWhenLaterSourceChangesBeforeDelete(t *testing.T) {
	updated := "budget is now 20000 dollars after the revision"
	store := newScriptMemoryStore(
		&Memory{ID: "a", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma", CreatedAt: time.Now()},
		&Memory{ID: "b", UserID: "user-1", Type: MemoryTypeSemantic, Content: "alpha beta gamma delta", CreatedAt: time.Now()},
	)
	store.beforeSourceDelete = func(id string) {
		if id != "b" {
			return
		}
		store.mu.Lock()
		defer store.mu.Unlock()
		for _, mem := range store.memories {
			if mem.ID == "b" {
				mem.Content = updated
				mem.UpdatedAt = time.Now()
			}
		}
	}

	merged, deleted, err := ConsolidateUser(context.Background(), store, "user-1")
	require.NoError(t, err)
	require.Equal(t, 1, merged)
	require.Equal(t, 1, deleted)

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.memories, 2)
	byID := map[string]*Memory{}
	summaryFound := false
	for _, mem := range store.memories {
		if mem.Source == "consolidation" {
			summaryFound = true
		}
		byID[mem.ID] = mem
	}
	require.NotContains(t, byID, "a")
	require.Equal(t, updated, byID["b"].Content)
	require.True(t, summaryFound)
}

func TestInMemoryForgetIfCurrentLeavesNewerVersion(t *testing.T) {
	store := NewInMemoryStore()
	original := &Memory{
		ID: "a", UserID: "user-1", Type: MemoryTypeSemantic,
		Content: "alpha beta gamma", UpdatedAt: time.Unix(10, 0).UTC(),
	}
	store.memories[original.ID] = original
	want := versionOf(original)

	store.memories[original.ID] = &Memory{
		ID: "a", UserID: "user-1", Type: MemoryTypeSemantic,
		Content: "budget is now 20000 dollars after the revision", UpdatedAt: time.Unix(11, 0).UTC(),
	}

	deleted, err := store.forgetIfCurrent(context.Background(), want)
	require.NoError(t, err)
	require.False(t, deleted)
	require.Equal(t, "budget is now 20000 dollars after the revision", store.memories["a"].Content)

	deleted, err = store.forgetIfCurrent(context.Background(), versionOf(store.memories["a"]))
	require.NoError(t, err)
	require.True(t, deleted)
	_, exists := store.memories["a"]
	require.False(t, exists)
}

func TestMilvusConditionalDeleteExprKeepsContentQuoted(t *testing.T) {
	updatedAt := time.Unix(1_700_000_000, 0).UTC()
	expr := milvusConditionalDeleteExpr(memoryVersion{
		id:         `id" || id != "x`,
		userID:     "user-1",
		projectID:  "",
		typ:        MemoryTypeSemantic,
		content:    "alpha \"beta\"",
		createdAt:  time.Unix(1_600_000_000, 0).UTC(),
		updatedAt:  updatedAt,
		importance: 0.7,
	})
	require.Contains(t, expr, `id == "id\" || id != \"x"`)
	require.Contains(t, expr, `project_id == "default"`)
	require.Contains(t, expr, `content == "alpha \"beta\""`)
	require.Contains(t, expr, "created_at == 1600000000")
	require.Contains(t, expr, "updated_at == 1700000000")
	require.Contains(t, expr, "importance == 0.7")
	require.NotContains(t, expr, `id != "x" &&`)
}
