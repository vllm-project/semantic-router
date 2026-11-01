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
// Groups are formed by text-unit Jaccard similarity. Within each group, the
// contents are concatenated into a single summary memory and the originals are
// deleted. This reduces redundancy and improves retrieval quality over time.
//
// Designed to be called from a background goroutine on a schedule.
func ConsolidateUser(ctx context.Context, store Store, userID string) (merged int, deleted int, err error) {
	if !store.IsEnabled() {
		return 0, 0, fmt.Errorf("memory store is not enabled")
	}
	if userID == "" {
		return 0, 0, fmt.Errorf("user ID is required")
	}
	replacer, ok := store.(atomicGroupReplacer)
	if !ok {
		return 0, 0, fmt.Errorf("memory store does not support atomic consolidation")
	}

	result, err := store.List(ctx, ListOptions{
		UserID: userID,
		Limit:  consolidationMaxListLimit,
	})
	if err != nil {
		return 0, 0, fmt.Errorf("listing memories for consolidation: %w", err)
	}

	if result.Total <= 1 {
		return 0, 0, nil
	}

	// Copy source versions at list time. The backend atomically compares this
	// snapshot before replacing the group, so a concurrent delete or update is
	// never overwritten by the merge.
	listed := make(map[string]memoryVersion, len(result.Memories))
	for _, mem := range result.Memories {
		if mem == nil || mem.ID == "" {
			continue
		}
		listed[mem.ID] = versionOf(mem)
	}

	// Never merge across project or type: similar text in different scopes must stay separate.
	for _, scoped := range partitionByProjectAndType(result.Memories) {
		if len(scoped) < 2 {
			continue
		}
		groups := groupBySimilarity(scoped, consolidationGroupThreshold)
		for _, group := range groups {
			if len(group) < 2 {
				continue
			}
			versions := make([]memoryVersion, 0, len(group))
			for _, mem := range group {
				version, ok := listed[mem.ID]
				if !ok {
					versions = nil
					break
				}
				versions = append(versions, version)
			}
			if len(versions) < 2 {
				continue
			}

			summary := mergeGroup(group)
			summaryMem := &Memory{
				ID:         generateMemoryID(),
				Type:       group[0].Type,
				Content:    summary,
				UserID:     userID,
				ProjectID:  group[0].ProjectID,
				Source:     "consolidation",
				CreatedAt:  earliestCreatedAt(group),
				Importance: maxImportance(group),
			}

			replaced, removed, replaceErr := replacer.replaceCurrentGroup(ctx, versions, summaryMem)
			if replaceErr != nil {
				return merged, deleted, replaceErr
			}
			if !replaced {
				logging.Warnf("ConsolidateUser: skipped stale group for user=%s", userID)
				continue
			}
			merged++
			deleted += removed
		}
	}

	logging.Infof("ConsolidateUser: user=%s merged=%d groups, deleted=%d originals", userID, merged, deleted)
	return merged, deleted, nil
}

// memoryVersion is the list-time identity of one source record.
// A source is deleted only when the store still matches this version.
type memoryVersion struct {
	id         string
	userID     string
	projectID  string
	typ        MemoryType
	content    string
	createdAt  time.Time
	updatedAt  time.Time
	importance float32
}

func versionOf(mem *Memory) memoryVersion {
	return memoryVersion{
		id:         mem.ID,
		userID:     mem.UserID,
		projectID:  mem.ProjectID,
		typ:        mem.Type,
		content:    mem.Content,
		createdAt:  mem.CreatedAt,
		updatedAt:  mem.UpdatedAt,
		importance: mem.Importance,
	}
}

func sameVersion(want memoryVersion, live *Memory) bool {
	if live == nil {
		return false
	}
	return live.ID == want.id &&
		live.UserID == want.userID &&
		live.ProjectID == want.projectID &&
		live.Type == want.typ &&
		live.Content == want.content &&
		live.CreatedAt.Equal(want.createdAt) &&
		live.UpdatedAt.Equal(want.updatedAt) &&
		live.Importance == want.importance
}

// atomicGroupReplacer replaces a group only if every listed source still
// matches its version. Implementations must create the summary and delete all
// originals in one atomic backend operation; partial consolidation can
// resurrect stale content after a concurrent update or delete.
type atomicGroupReplacer interface {
	replaceCurrentGroup(ctx context.Context, versions []memoryVersion, summary *Memory) (replaced bool, deleted int, err error)
}

// partitionByProjectAndType keeps consolidation inside one (project_id, type) bucket.
func partitionByProjectAndType(memories []*Memory) [][]*Memory {
	order := make([]string, 0)
	buckets := make(map[string][]*Memory)
	for _, mem := range memories {
		if mem == nil {
			continue
		}
		key := string(mem.Type) + "\x00" + mem.ProjectID
		if _, ok := buckets[key]; !ok {
			order = append(order, key)
		}
		buckets[key] = append(buckets[key], mem)
	}
	out := make([][]*Memory, 0, len(order))
	for _, key := range order {
		out = append(out, buckets[key])
	}
	return out
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
