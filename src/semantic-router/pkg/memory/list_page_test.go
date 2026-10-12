package memory

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSortMemoriesForList_TieBreaksByID(t *testing.T) {
	when := time.Unix(1_700_000_000, 0)
	memories := []*Memory{
		{ID: "a", CreatedAt: when},
		{ID: "c", CreatedAt: when.Add(time.Second)},
		{ID: "b", CreatedAt: when},
	}
	sortMemoriesForList(memories)
	require.Equal(t, []string{"c", "b", "a"}, []string{memories[0].ID, memories[1].ID, memories[2].ID})
}

func TestPageMemories_SecondPageAndTotal(t *testing.T) {
	memories := []*Memory{{ID: "3"}, {ID: "2"}, {ID: "1"}}
	page := pageMemories(memories, 2, 2)
	require.Equal(t, []string{"1"}, idsOf(page))
	require.Greater(t, len(memories), len(pageMemories(memories, 0, 2)))
}

func TestCreatedAtTieGroupPagesDoNotRepeat(t *testing.T) {
	when := time.Unix(1_700_000_000, 0)
	// Scroll order is not id order. Taking only offset+limit and then sorting
	// that prefix repeats b,a on the second page and drops d,c.
	scrolled := []*Memory{
		{ID: "a", CreatedAt: when},
		{ID: "b", CreatedAt: when},
		{ID: "c", CreatedAt: when},
		{ID: "d", CreatedAt: when},
	}
	require.True(t, createdAtPrefixNeedsMore(scrolled[:2], 2))
	require.True(t, createdAtPrefixNeedsMore(scrolled, 2))
	require.False(t, createdAtPrefixNeedsMore(append(scrolled, &Memory{ID: "e", CreatedAt: when.Add(-time.Second)}), 2))

	complete := mergeCreatedAtTieGroup(scrolled[:2], when, scrolled)
	sortMemoriesForList(complete)
	page0 := pageMemories(complete, 0, 2)
	page1 := pageMemories(complete, 2, 2)
	require.Equal(t, []string{"d", "c"}, idsOf(page0))
	require.Equal(t, []string{"b", "a"}, idsOf(page1))
}

func TestCreatedAtTieCapIgnoresOffsetPrefix(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)
	const offset = 1000
	const limit = 20
	window := offset + limit
	distinct := make([]*Memory, window)
	for i := range distinct {
		distinct[i] = &Memory{
			ID:        fmt.Sprintf("id-%04d", i),
			CreatedAt: base.Add(time.Duration(window-i) * time.Second),
		}
	}
	require.False(t, createdAtTieGroupExceeds(distinct, window, maxListTieGroup))

	tied := make([]*Memory, maxListTieGroup)
	when := base
	for i := range tied {
		tied[i] = &Memory{ID: fmt.Sprintf("tie-%04d", i), CreatedAt: when}
	}
	require.False(t, createdAtTieGroupExceeds(tied, limit, maxListTieGroup))
	tied = append(tied, &Memory{ID: "tie-extra", CreatedAt: when})
	require.True(t, createdAtTieGroupExceeds(tied, limit, maxListTieGroup))
}

func TestNormalizeListWindow_RejectsNegativeOffset(t *testing.T) {
	_, _, err := normalizeListWindow(ListOptions{Offset: -1})
	require.Error(t, err)
	limit, offset, err := normalizeListWindow(ListOptions{})
	require.NoError(t, err)
	require.Equal(t, defaultListLimit, limit)
	require.Equal(t, 0, offset)
}

func idsOf(memories []*Memory) []string {
	ids := make([]string, len(memories))
	for i, mem := range memories {
		ids[i] = mem.ID
	}
	return ids
}
