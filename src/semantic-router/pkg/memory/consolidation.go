package memory

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	consolidationGroupThreshold float32 = 0.60
	consolidationMaxListLimit   int     = 100
)

// ConsolidateUser scans a user's memories and merges semantically related ones.
// Groups are formed by word-level Jaccard similarity. Within each group, the
// contents are concatenated into a single summary memory and the originals are
// deleted. This reduces redundancy and improves retrieval quality over time.
//
// Designed to be called from a background goroutine on a schedule.
func (m *MilvusStore) ConsolidateUser(ctx context.Context, userID string) (merged int, deleted int, err error) {
	if !m.enabled {
		return 0, 0, fmt.Errorf("milvus store is not enabled")
	}
	if userID == "" {
		return 0, 0, fmt.Errorf("user ID is required")
	}

	result, err := m.List(ctx, ListOptions{
		UserID: userID,
		Limit:  consolidationMaxListLimit,
	})
	if err != nil {
		return 0, 0, fmt.Errorf("listing memories for consolidation: %w", err)
	}

	if result.Total <= 1 {
		return 0, 0, nil
	}

	groups := groupBySimilarity(result.Memories, consolidationGroupThreshold)

	for _, group := range groups {
		if len(group) < 2 {
			continue
		}

		summary := mergeGroup(group)
		summaryMem := &Memory{
			ID:         generateMemoryID(),
			Type:       group[0].Type,
			Content:    summary,
			UserID:     userID,
			Source:     "consolidation",
			CreatedAt:  earliestCreatedAt(group),
			Importance: maxImportance(group),
		}

		if err := m.Store(ctx, summaryMem); err != nil {
			logging.Warnf("ConsolidateUser: failed to store merged memory: %v", err)
			continue
		}

		for _, old := range group {
			if ferr := m.Forget(ctx, old.ID); ferr != nil {
				logging.Warnf("ConsolidateUser: failed to delete original memory id=%s: %v", old.ID, ferr)
			} else {
				deleted++
			}
		}
		merged++
	}

	logging.Infof("ConsolidateUser: user=%s merged=%d groups, deleted=%d originals", userID, merged, deleted)
	return merged, deleted, nil
}

// groupBySimilarity clusters memories by pairwise Jaccard similarity.
// Greedy single-linkage: each memory is assigned to the first existing
// group where it exceeds the threshold with any member.
func groupBySimilarity(memories []*Memory, threshold float32) [][]*Memory {
	var groups [][]*Memory

	for _, mem := range memories {
		placed := false
		for gi := range groups {
			for _, existing := range groups[gi] {
				if wordJaccard(mem.Content, existing.Content) >= threshold {
					groups[gi] = append(groups[gi], mem)
					placed = true
					break
				}
			}
			if placed {
				break
			}
		}
		if !placed {
			groups = append(groups, []*Memory{mem})
		}
	}
	return groups
}

// mergeGroup concatenates group members' content, deduplicating identical lines.
func mergeGroup(group []*Memory) string {
	seen := make(map[string]bool)
	var parts []string
	for _, m := range group {
		lines := strings.Split(strings.TrimSpace(m.Content), "\n")
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			if !seen[trimmed] {
				seen[trimmed] = true
				parts = append(parts, trimmed)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func earliestCreatedAt(group []*Memory) time.Time {
	earliest := group[0].CreatedAt
	for _, m := range group[1:] {
		if m.CreatedAt.Before(earliest) {
			earliest = m.CreatedAt
		}
	}
	return earliest
}

func maxImportance(group []*Memory) float32 {
	var max float32
	for _, m := range group {
		if m.Importance > max {
			max = m.Importance
		}
	}
	return max
}
