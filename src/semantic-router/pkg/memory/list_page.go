package memory

import (
	"fmt"
	"sort"
	"time"
)

const (
	defaultListLimit = 20
	maxListLimit     = 100
	maxListOffset    = 100_000
	// maxListTieGroup bounds how many equal-timestamp rows a page boundary may load.
	// Backends that cannot sort by (created_at, id) must load the whole tie group
	// that touches the page, or a later page repeats rows from the earlier one.
	maxListTieGroup = 1_000
)

// normalizeListWindow rejects a negative offset and applies the shared page size.
func normalizeListWindow(opts ListOptions) (limit, offset int, err error) {
	if opts.Offset < 0 {
		return 0, 0, fmt.Errorf("offset must be non-negative")
	}
	if opts.Offset > maxListOffset {
		return 0, 0, fmt.Errorf("offset exceeds maximum of %d", maxListOffset)
	}
	limit = opts.Limit
	if limit <= 0 {
		limit = defaultListLimit
	}
	if limit > maxListLimit {
		limit = maxListLimit
	}
	return limit, opts.Offset, nil
}

// sortMemoriesForList orders newest first. Equal timestamps break ties by id descending.
func sortMemoriesForList(memories []*Memory) {
	sort.SliceStable(memories, func(i, j int) bool {
		left, right := memories[i], memories[j]
		if left == nil || right == nil {
			return left != nil
		}
		if !left.CreatedAt.Equal(right.CreatedAt) {
			return left.CreatedAt.After(right.CreatedAt)
		}
		return left.ID > right.ID
	})
}

// createdAtPrefixNeedsMore reports whether a created_at-descending prefix still
// ends inside the timestamp group that touches the requested window. Callers
// that only sorted by created_at must keep reading until this is false, then
// sort by id and slice. An empty or short prefix still needs more rows.
func createdAtPrefixNeedsMore(collected []*Memory, window int) bool {
	if window <= 0 || len(collected) < window {
		return true
	}
	boundary := collected[window-1]
	last := collected[len(collected)-1]
	if boundary == nil || last == nil {
		return false
	}
	return !last.CreatedAt.Before(boundary.CreatedAt)
}

// createdAtTieGroupExceeds reports whether the timestamp that touches the page
// end has more than maxTie rows. Rows newer than that timestamp are the prefix
// required to reach the offset, and they do not count toward the cap.
func createdAtTieGroupExceeds(collected []*Memory, window, maxTie int) bool {
	if window <= 0 || maxTie < 0 || len(collected) < window {
		return false
	}
	boundary := collected[window-1]
	if boundary == nil {
		return false
	}
	count := 0
	for _, mem := range collected {
		if mem != nil && mem.CreatedAt.Equal(boundary.CreatedAt) {
			count++
		}
	}
	return count > maxTie
}

// mergeCreatedAtTieGroup keeps rows strictly newer than boundary and replaces
// the boundary timestamp with the complete tie group.
func mergeCreatedAtTieGroup(prefix []*Memory, boundary time.Time, group []*Memory) []*Memory {
	merged := make([]*Memory, 0, len(prefix)+len(group))
	for _, mem := range prefix {
		if mem != nil && mem.CreatedAt.After(boundary) {
			merged = append(merged, mem)
		}
	}
	for _, mem := range group {
		if mem != nil && mem.CreatedAt.Equal(boundary) {
			merged = append(merged, mem)
		}
	}
	return merged
}

// pageMemories returns one page. An offset past the end yields an empty page, not an error.
func pageMemories(memories []*Memory, offset, limit int) []*Memory {
	if offset >= len(memories) {
		return []*Memory{}
	}
	end := offset + limit
	if end > len(memories) {
		end = len(memories)
	}
	page := make([]*Memory, end-offset)
	copy(page, memories[offset:end])
	return page
}
