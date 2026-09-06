package evaluationplane

import (
	"log"
	"sort"
	"sync"
)

// runMetadataIndex is a process-local projection rebuilt exclusively from
// canonical run bundles. It owns no durable facts. Stores opened on the same
// root share the coordinator and projection so mutations remain coherent
// across Store instances in the one-node backend. Owner warning counters cover
// every quarantined namespace with a previously established immutable owner,
// including warnings beyond the public cap.
type runMetadataIndex struct {
	coordinator sync.Mutex
	mu          sync.RWMutex
	runs        []Run
	positions   map[string]int
	// ownerDigests is write-once while a canonical run namespace exists. It is
	// retained across quarantine refreshes so repaired evidence cannot rebind an
	// identity, and removed only when the namespace is durably absent/deleted.
	ownerDigests   map[string]string
	ownerRunIDs    map[string][]string
	warnings       map[string]runListWarning
	warningCount   int
	ownerWarnings  map[string]int
	eventSequences map[string]uint64
	pendingChanges map[string]bool
}

func newRunMetadataIndex() *runMetadataIndex {
	return &runMetadataIndex{
		positions: make(map[string]int), ownerDigests: make(map[string]string),
		ownerRunIDs: make(map[string][]string), warnings: make(map[string]runListWarning),
		ownerWarnings:  make(map[string]int),
		eventSequences: make(map[string]uint64), pendingChanges: make(map[string]bool),
	}
}

func (index *runMetadataIndex) replace(
	runs []Run,
	ownerDigests map[string]string,
	presentRunIDs map[string]bool,
	warnings map[string]runListWarning,
	warningCount int,
	ownerWarnings map[string]int,
) {
	sort.Slice(runs, func(left, right int) bool { return runNewer(runs[left], runs[right]) })
	index.mu.Lock()
	defer index.mu.Unlock()
	for evidenceID, warning := range warnings {
		if previous, unchanged := index.warnings[evidenceID]; unchanged && previous == warning {
			continue
		}
		log.Printf(
			"evaluationplane: warning_code=%s evidence_id=%q message=%q",
			warning.Code, warning.EvidenceID, warning.Message,
		)
	}
	index.runs = copyRuns(runs)
	index.positions = make(map[string]int, len(index.runs))
	nextOwnerDigests := make(map[string]string, len(index.ownerDigests)+len(ownerDigests))
	for runID, ownerDigest := range index.ownerDigests {
		if presentRunIDs[runID] {
			nextOwnerDigests[runID] = ownerDigest
		}
	}
	for runID, ownerDigest := range ownerDigests {
		if nextOwnerDigests[runID] == "" {
			nextOwnerDigests[runID] = ownerDigest
		}
	}
	index.ownerDigests = nextOwnerDigests
	index.rebuildPositions(0)
	index.rebuildOwnerRunIDsLocked()
	activeSequences := make(map[string]uint64, len(index.runs))
	for _, run := range index.runs {
		if sequence, exists := index.eventSequences[run.ID]; exists {
			activeSequences[run.ID] = sequence
		}
	}
	index.eventSequences = activeSequences
	index.warnings = warnings
	index.warningCount = warningCount
	index.ownerWarnings = ownerWarnings
	index.pendingChanges = make(map[string]bool)
}

func (index *runMetadataIndex) markPendingChange(identity string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.pendingChanges[identity] = true
}

func (index *runMetadataIndex) clearPendingChange(identity string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	delete(index.pendingChanges, identity)
}

func (index *runMetadataIndex) hasPendingChanges() bool {
	index.mu.RLock()
	defer index.mu.RUnlock()
	return len(index.pendingChanges) != 0
}

func (index *runMetadataIndex) upsert(run Run) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.upsertLocked(run, index.ownerDigests[run.ID])
}

func (index *runMetadataIndex) upsertBatch(runs []Run, eventSequences map[string]uint64) {
	index.mu.Lock()
	defer index.mu.Unlock()
	for _, run := range runs {
		index.upsertLocked(run, index.ownerDigests[run.ID])
	}
	for runID, sequence := range eventSequences {
		index.eventSequences[runID] = sequence
	}
}

func (index *runMetadataIndex) upsertOwned(run Run, ownerDigest string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.upsertLocked(run, ownerDigest)
}

func (index *runMetadataIndex) upsertOwnedBatch(
	runs []Run,
	ownerDigest string,
	eventSequences map[string]uint64,
) {
	index.mu.Lock()
	defer index.mu.Unlock()
	for _, run := range runs {
		index.upsertLocked(run, ownerDigest)
	}
	for runID, sequence := range eventSequences {
		index.eventSequences[runID] = sequence
	}
}

func (index *runMetadataIndex) upsertLocked(run Run, ownerDigest string) {
	run = copyRun(run)
	if position, exists := index.positions[run.ID]; exists {
		ownerAdded := index.ownerDigests[run.ID] == "" && ownerDigest != ""
		if ownerAdded {
			index.ownerDigests[run.ID] = ownerDigest
		}
		if index.runs[position].CreatedAt.Equal(run.CreatedAt) {
			index.runs[position] = run
			if ownerAdded {
				index.rebuildOwnerRunIDsLocked()
			}
			return
		}
		copy(index.runs[position:], index.runs[position+1:])
		index.runs = index.runs[:len(index.runs)-1]
		delete(index.positions, run.ID)
		index.rebuildPositions(position)
	}
	position := sort.Search(len(index.runs), func(candidate int) bool {
		return runNewer(run, index.runs[candidate])
	})
	index.runs = append(index.runs, Run{})
	copy(index.runs[position+1:], index.runs[position:])
	index.runs[position] = run
	if index.ownerDigests[run.ID] == "" && ownerDigest != "" {
		index.ownerDigests[run.ID] = ownerDigest
	}
	index.rebuildPositions(position)
	index.rebuildOwnerRunIDsLocked()
}

func (index *runMetadataIndex) remove(runID string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.removeLocked(runID)
}

func (index *runMetadataIndex) removeBatch(runIDs ...string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	for _, runID := range runIDs {
		index.removeLocked(runID)
	}
}

func (index *runMetadataIndex) removeLocked(runID string) {
	position, exists := index.positions[runID]
	if !exists {
		return
	}
	copy(index.runs[position:], index.runs[position+1:])
	index.runs = index.runs[:len(index.runs)-1]
	delete(index.positions, runID)
	delete(index.ownerDigests, runID)
	delete(index.eventSequences, runID)
	index.rebuildPositions(position)
	index.rebuildOwnerRunIDsLocked()
}

func (index *runMetadataIndex) eventSequence(runID string) (uint64, bool) {
	index.mu.RLock()
	defer index.mu.RUnlock()
	sequence, exists := index.eventSequences[runID]
	return sequence, exists
}

func (index *runMetadataIndex) ownerDigest(runID string) (string, bool) {
	index.mu.RLock()
	defer index.mu.RUnlock()
	ownerDigest, exists := index.ownerDigests[runID]
	return ownerDigest, exists && ownerDigest != ""
}

func (index *runMetadataIndex) setEventSequence(runID string, sequence uint64) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.eventSequences[runID] = sequence
}

func (index *runMetadataIndex) rebuildPositions(start int) {
	if index.positions == nil {
		index.positions = make(map[string]int, len(index.runs))
	}
	for position := start; position < len(index.runs); position++ {
		index.positions[index.runs[position].ID] = position
	}
}

func (index *runMetadataIndex) allRuns() []Run {
	index.mu.RLock()
	defer index.mu.RUnlock()
	return copyRuns(index.runs)
}

func (index *runMetadataIndex) page(cursor *runListCursor, limit int) (runs []Run, total int, warnings []runListWarning, warningCount int) {
	index.mu.RLock()
	defer index.mu.RUnlock()
	return index.pageLocked(cursor, limit)
}

func (index *runMetadataIndex) stablePage(
	actor Actor,
	cursor *runListCursor,
	limit int,
) (runs []Run, total int, warnings []runListWarning, warningCount int, stable bool) {
	index.mu.RLock()
	defer index.mu.RUnlock()
	if len(index.pendingChanges) != 0 {
		return nil, 0, nil, 0, false
	}
	if actor.administrator {
		runs, total, warnings, warningCount = index.pageLocked(cursor, limit)
		return runs, total, warnings, warningCount, true
	}
	runs, total = index.ownerPageLocked(actor.principalDigest, cursor, limit)
	warningCount = index.ownerWarnings[actor.principalDigest]
	return runs, total, nil, warningCount, true
}

func (index *runMetadataIndex) ownerPageLocked(
	ownerDigest string,
	cursor *runListCursor,
	limit int,
) ([]Run, int) {
	ids := index.ownerRunIDs[ownerDigest]
	start := 0
	if cursor != nil {
		start = sort.Search(len(ids), func(position int) bool {
			runPosition, exists := index.positions[ids[position]]
			return exists && runOlderThanCursor(index.runs[runPosition], *cursor)
		})
	}
	end := start + limit + 1
	if end > len(ids) {
		end = len(ids)
	}
	runs := make([]Run, 0, end-start)
	for _, runID := range ids[start:end] {
		if position, exists := index.positions[runID]; exists {
			runs = append(runs, copyRun(index.runs[position]))
		}
	}
	return runs, len(ids)
}

func (index *runMetadataIndex) pageLocked(
	cursor *runListCursor,
	limit int,
) (runs []Run, total int, warnings []runListWarning, warningCount int) {
	start := 0
	if cursor != nil {
		start = sort.Search(len(index.runs), func(position int) bool {
			return runOlderThanCursor(index.runs[position], *cursor)
		})
	}
	end := start + limit + 1
	if end > len(index.runs) {
		end = len(index.runs)
	}
	runs = copyRuns(index.runs[start:end])
	warnings = make([]runListWarning, 0, len(index.warnings))
	for _, warning := range index.warnings {
		warnings = append(warnings, warning)
	}
	return runs, len(index.runs), warnings, index.warningCount
}

func (index *runMetadataIndex) rebuildOwnerRunIDsLocked() {
	index.ownerRunIDs = make(map[string][]string)
	for _, run := range index.runs {
		if ownerDigest := index.ownerDigests[run.ID]; ownerDigest != "" {
			index.ownerRunIDs[ownerDigest] = append(index.ownerRunIDs[ownerDigest], run.ID)
		}
	}
}

func copyRuns(source []Run) []Run {
	result := make([]Run, len(source))
	for index := range source {
		result[index] = copyRun(source[index])
	}
	return result
}

func copyRun(source Run) Run {
	result := source
	result.TrackEvidenceLevels = copyTrackEvidenceLevels(source.TrackEvidenceLevels)
	result.Mixture = copyCatalogMixture(source.Mixture)
	result.SuiteIDs = append([]string(nil), source.SuiteIDs...)
	result.TrackIDs = append([]TrackID(nil), source.TrackIDs...)
	result.CapacitySLO = copyCapacitySLO(source.CapacitySLO)
	result.CapacityLoadProtocol = copyCapacityLoadProtocol(source.CapacityLoadProtocol)
	result.ControlledPair = copyControlledPairRunMembership(source.ControlledPair)
	result.StartedAt = copyTime(source.StartedAt)
	result.CompletedAt = copyTime(source.CompletedAt)
	return result
}
