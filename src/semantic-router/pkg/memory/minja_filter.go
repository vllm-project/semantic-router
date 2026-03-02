package memory

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MinjaFilter implements MemoryFilter to defend against Memory Injection Attacks
// (arXiv:2503.03704). It applies system-level controls on the read path:
//
//   - Higher similarity threshold for non-owner (shared) memories
//   - Cap on total shared memories per request
//   - Per-creator diversity cap to prevent one attacker from dominating
//
// Prompt-level detection is handled upstream by the jailbreak classifier;
// this filter focuses on structural isolation that is attack-agnostic.
type MinjaFilter struct {
	requestingUserID string
	cfg              config.MinjaDefenseConfig
}

// NewMinjaFilter creates a MinjaFilter for a specific request.
// The requestingUserID is needed to distinguish owned vs. shared memories.
func NewMinjaFilter(cfg config.MinjaDefenseConfig, requestingUserID string) *MinjaFilter {
	if !cfg.IsEnabled() {
		return nil
	}
	return &MinjaFilter{
		requestingUserID: requestingUserID,
		cfg:              cfg,
	}
}

// Filter applies shared-memory isolation controls. Owned memories pass through
// unchanged; non-owner memories are subject to stricter similarity thresholds,
// total caps, and per-creator diversity limits.
func (f *MinjaFilter) Filter(memories []*RetrieveResult) []*RetrieveResult {
	if f == nil || len(memories) == 0 {
		return memories
	}

	sharedMinSim := f.cfg.GetSharedMemoryMinSimilarity()
	maxShared := f.cfg.GetMaxSharedMemoriesPerRequest()
	maxPerCreator := f.cfg.GetMaxSharedPerCreator()

	passed := make([]*RetrieveResult, 0, len(memories))
	sharedCount := 0
	creatorCounts := make(map[string]int)

	for _, mem := range memories {
		if mem.Memory == nil {
			continue
		}

		if !f.isSharedMemory(mem) {
			RecordMinjaFilter(true)
			passed = append(passed, mem)
			continue
		}

		if mem.Score < sharedMinSim {
			logging.Debugf("MinjaFilter: rejected shared memory id=%s score=%.3f < threshold=%.3f",
				mem.Memory.ID, mem.Score, sharedMinSim)
			RecordSharedMemoryFiltered("low_similarity")
			RecordMinjaFilter(false)
			continue
		}

		if sharedCount >= maxShared {
			logging.Debugf("MinjaFilter: rejected shared memory id=%s (shared cap %d reached)",
				mem.Memory.ID, maxShared)
			RecordSharedMemoryFiltered("shared_cap")
			RecordMinjaFilter(false)
			continue
		}

		creator := mem.Memory.UserID
		if creatorCounts[creator] >= maxPerCreator {
			logging.Debugf("MinjaFilter: rejected shared memory id=%s (creator %s cap %d reached)",
				mem.Memory.ID, creator, maxPerCreator)
			RecordSharedMemoryFiltered("creator_cap")
			RecordMinjaFilter(false)
			continue
		}

		sharedCount++
		creatorCounts[creator]++
		RecordSharedMemoryAccepted()
		RecordMinjaFilter(true)
		passed = append(passed, mem)
	}

	if len(passed) < len(memories) {
		logging.Infof("MinjaFilter: %d→%d memories (filtered %d shared/non-owner)",
			len(memories), len(passed), len(memories)-len(passed))
	}

	return passed
}

// isSharedMemory returns true if the memory belongs to a different user.
func (f *MinjaFilter) isSharedMemory(mem *RetrieveResult) bool {
	if f.requestingUserID == "" || mem.Memory == nil {
		return false
	}
	return mem.Memory.UserID != "" && mem.Memory.UserID != f.requestingUserID
}
