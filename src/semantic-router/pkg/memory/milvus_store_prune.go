package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MemoryPruneEntry holds minimal fields needed to compute retention score for pruning.
type MemoryPruneEntry struct {
	ID           string
	LastAccessed time.Time
	AccessCount  int
}

// triggerCapEnforcement fires an async prune if user exceeds max_memories_per_user.
// Uses context.Background() intentionally: the goroutine must outlive the request ctx.
// Two layers of protection against Milvus pressure:
//   - pruneInFlight (sync.Map): dedup — at most one goroutine per user at any time
//   - pruneSem (channel): semaphore — at most maxConcurrentPrunes goroutines globally
func (m *MilvusStore) triggerCapEnforcement(userID string) {
	if m.config.QualityScoring.MaxMemoriesPerUser <= 0 {
		return
	}
	if _, alreadyRunning := m.pruneInFlight.LoadOrStore(userID, struct{}{}); alreadyRunning {
		return
	}
	select {
	case m.pruneSem <- struct{}{}:
		go func(uid string) {
			defer func() {
				<-m.pruneSem
				m.pruneInFlight.Delete(uid)
			}()
			m.pruneIfOverCap(context.Background(), uid)
		}(userID)
	default:
		m.pruneInFlight.Delete(userID)
		logging.Debugf("MilvusStore.Store: prune semaphore full, skipping cap check for user_id=%s", userID)
	}
}

// pruneIfOverCap counts the user's memories and calls PruneUser if over MaxMemoriesPerUser.
func (m *MilvusStore) pruneIfOverCap(ctx context.Context, userID string) {
	cap := m.config.QualityScoring.MaxMemoriesPerUser
	if cap <= 0 {
		return
	}

	count, err := m.countUserMemories(ctx, userID)
	if err != nil {
		logging.Warnf("MilvusStore.pruneIfOverCap: count failed for user_id=%s: %v", userID, err)
		return
	}

	if count <= cap {
		return
	}

	PruneCapTriggeredTotal.Inc()
	logging.Infof("MilvusStore.pruneIfOverCap: user_id=%s has %d memories (cap=%d), pruning", userID, count, cap)

	deleted, err := m.PruneUser(ctx, userID)
	if err != nil {
		logging.Warnf("MilvusStore.pruneIfOverCap: PruneUser failed for user_id=%s: %v", userID, err)
		return
	}
	if deleted > 0 {
		PruneDeletedTotal.WithLabelValues("cap").Add(float64(deleted))
		logging.Infof("MilvusStore.pruneIfOverCap: user_id=%s pruned %d memories", userID, deleted)
	}
}

// countUserMemories returns the number of memories stored for a given user.
func (m *MilvusStore) countUserMemories(ctx context.Context, userID string) (int, error) {
	filterExpr := fmt.Sprintf("user_id == \"%s\"", userID)

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			[]string{"id"},
		)
		return retryErr
	})
	if err != nil {
		return 0, fmt.Errorf("milvus query failed: %w", err)
	}

	for _, col := range queryResult {
		if col.Name() == "id" {
			return col.Len(), nil
		}
	}
	return 0, nil
}

// ListForPrune returns all memories for a user with id, last_accessed, access_count for pruning.
func (m *MilvusStore) ListForPrune(ctx context.Context, userID string) ([]MemoryPruneEntry, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}
	if userID == "" {
		return nil, fmt.Errorf("user ID is required")
	}

	queryResult, err := m.queryForPrune(ctx, userID)
	if err != nil {
		return nil, err
	}

	idCol, metadataCol, createdAtCol := extractPruneColumns(queryResult)
	if idCol == nil {
		return []MemoryPruneEntry{}, nil
	}

	return buildPruneEntries(idCol, metadataCol, createdAtCol), nil
}

func (m *MilvusStore) queryForPrune(ctx context.Context, userID string) ([]entity.Column, error) {
	filterExpr := fmt.Sprintf("user_id == \"%s\"", userID)
	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(ctx, m.collectionName, []string{}, filterExpr, []string{"id", "metadata", "created_at"})
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}
	return queryResult, nil
}

func extractPruneColumns(cols []entity.Column) (*entity.ColumnVarChar, *entity.ColumnVarChar, *entity.ColumnInt64) {
	var idCol, metadataCol *entity.ColumnVarChar
	var createdAtCol *entity.ColumnInt64
	for _, col := range cols {
		switch col.Name() {
		case "id":
			idCol, _ = col.(*entity.ColumnVarChar)
		case "metadata":
			metadataCol, _ = col.(*entity.ColumnVarChar)
		case "created_at":
			createdAtCol, _ = col.(*entity.ColumnInt64)
		}
	}
	return idCol, metadataCol, createdAtCol
}

func buildPruneEntries(idCol, metadataCol *entity.ColumnVarChar, createdAtCol *entity.ColumnInt64) []MemoryPruneEntry {
	now := time.Now()
	entries := make([]MemoryPruneEntry, 0, idCol.Len())
	for i := 0; i < idCol.Len(); i++ {
		id, _ := idCol.ValueByIdx(i)
		entry := MemoryPruneEntry{ID: id}
		parseAccessMetadata(&entry, metadataCol, i)
		applyLastAccessedFallback(&entry, createdAtCol, i, now)
		entries = append(entries, entry)
	}
	return entries
}

// parseAccessMetadata extracts last_accessed and access_count from the JSON metadata column.
func parseAccessMetadata(entry *MemoryPruneEntry, metadataCol *entity.ColumnVarChar, idx int) {
	if metadataCol == nil || metadataCol.Len() <= idx {
		return
	}
	metadataStr, _ := metadataCol.ValueByIdx(idx)
	var meta map[string]interface{}
	if json.Unmarshal([]byte(metadataStr), &meta) != nil {
		return
	}
	if la, ok := meta["last_accessed"].(float64); ok {
		entry.LastAccessed = time.Unix(int64(la), 0)
	}
	if ac, ok := meta["access_count"].(float64); ok {
		entry.AccessCount = int(ac)
	}
}

// applyLastAccessedFallback sets LastAccessed from created_at (or now) for pre-existing
// memories that were stored before access tracking was enabled.
func applyLastAccessedFallback(entry *MemoryPruneEntry, createdAtCol *entity.ColumnInt64, idx int, now time.Time) {
	if !entry.LastAccessed.IsZero() {
		return
	}
	if createdAtCol != nil && createdAtCol.Len() > idx {
		if val, _ := createdAtCol.ValueByIdx(idx); val > 0 {
			entry.LastAccessed = time.Unix(val, 0)
			return
		}
	}
	entry.LastAccessed = now
}

// PruneUser deletes memories for userID that have R < PruneThreshold, then if over MaxMemoriesPerUser
// deletes lowest-R memories until at cap. No-op if the store is disabled.
func (m *MilvusStore) PruneUser(ctx context.Context, userID string) (deleted int, err error) {
	if !m.enabled {
		return 0, nil
	}
	cfg := m.config.QualityScoring

	initialStrength := cfg.InitialStrengthDays
	if initialStrength <= 0 {
		initialStrength = DefaultInitialStrengthDays
	}
	delta := cfg.PruneThreshold
	if delta <= 0 {
		delta = DefaultPruneThreshold
	}
	maxPerUser := cfg.MaxMemoriesPerUser

	entries, err := m.ListForPrune(ctx, userID)
	if err != nil {
		return 0, err
	}

	toDelete := PruneCandidates(entries, time.Now(), initialStrength, delta, maxPerUser)
	for _, id := range toDelete {
		if err := m.Forget(ctx, id); err != nil {
			logging.Warnf("MilvusStore.PruneUser: Forget id=%s: %v", id, err)
			continue
		}
		deleted++
	}
	logging.Debugf("MilvusStore.PruneUser: user_id=%s deleted %d memories", userID, deleted)
	return deleted, nil
}

// ListStaleUserIDs queries Milvus for memories with created_at older than cutoffUnix
// and returns the deduplicated set of user_id values.
func (m *MilvusStore) ListStaleUserIDs(ctx context.Context, cutoffUnix int64) ([]string, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	filterExpr := fmt.Sprintf("created_at < %d", cutoffUnix)
	outputFields := []string{"user_id"}

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query for stale users failed: %w", err)
	}

	seen := make(map[string]struct{})
	for _, col := range queryResult {
		if col.Name() == "user_id" {
			vc, ok := col.(*entity.ColumnVarChar)
			if !ok {
				continue
			}
			for i := 0; i < vc.Len(); i++ {
				uid, _ := vc.ValueByIdx(i)
				if uid != "" {
					seen[uid] = struct{}{}
				}
			}
		}
	}

	userIDs := make([]string, 0, len(seen))
	for uid := range seen {
		userIDs = append(userIDs, uid)
	}
	logging.Debugf("MilvusStore.ListStaleUserIDs: found %d users with memories older than %d", len(userIDs), cutoffUnix)
	return userIDs, nil
}
