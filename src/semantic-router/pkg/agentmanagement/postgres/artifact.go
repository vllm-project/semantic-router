package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const artifactSelect = `SELECT id::text,session_id::text,turn_id::text,kind,media_type,content,
       content_digest,safe_preview,expires_at,created_at
  FROM agent_artifacts`

func (store *Store) PutArtifact(
	ctx context.Context, namespaceID string, value agentmanagement.Artifact, accessScope json.RawMessage,
) (agentmanagement.Artifact, error) {
	if len(value.Content) == 0 || len(value.Content) > 16<<20 || value.ExpiresAt.IsZero() {
		return agentmanagement.Artifact{}, agentmanagement.ErrInvalid
	}
	var preview, scope map[string]any
	if json.Unmarshal(value.SafePreview, &preview) != nil || preview == nil ||
		json.Unmarshal(accessScope, &scope) != nil || scope == nil {
		return agentmanagement.Artifact{}, agentmanagement.ErrInvalid
	}
	digest := sha256.Sum256(value.Content)
	if value.Digest != "" {
		expected, err := parseDigest(value.Digest)
		if err != nil || !equalDigest(expected, digest[:]) {
			return agentmanagement.Artifact{}, agentmanagement.ErrInvalid
		}
	}
	_, err := store.db.ExecContext(ctx, `INSERT INTO agent_artifacts
  (id,namespace_id,session_id,turn_id,kind,media_type,content,content_digest,safe_preview,access_scope,expires_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`, value.ID, namespaceID,
		value.SessionID, value.TurnID, value.Kind, value.MediaType, value.Content, digest[:],
		value.SafePreview, accessScope, value.ExpiresAt.UTC())
	if err != nil {
		return agentmanagement.Artifact{}, classifyWriteError(err)
	}
	return scanArtifact(store.db.QueryRowContext(ctx, artifactSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, value.ID))
}

func (store *Store) GetArtifact(
	ctx context.Context, namespaceID, id string,
) (agentmanagement.Artifact, error) {
	return scanArtifact(store.db.QueryRowContext(ctx, artifactSelect+`
 WHERE namespace_id=$1 AND id=$2 AND expires_at>clock_timestamp()`, namespaceID, id))
}

func (store *Store) PutCheckpoint(
	ctx context.Context, namespaceID string, value agentmanagement.Checkpoint,
) (agentmanagement.Checkpoint, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Checkpoint, error) {
		return putCheckpointTx(ctx, tx, namespaceID, value)
	})
}

// CommitCheckpoint atomically persists the bounded context projection and the
// event that makes it visible. A replacement worker can therefore recover
// from either the previous complete checkpoint or this one, never a dangling
// event/reference pair.
func (store *Store) CommitCheckpoint(
	ctx context.Context, lease agentmanagement.TurnLease, value agentmanagement.Checkpoint,
) (agentmanagement.Checkpoint, agentmanagement.Event, error) {
	type result struct {
		checkpoint agentmanagement.Checkpoint
		event      agentmanagement.Event
	}
	committed, err := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		var valid bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5
  AND status='running' AND cancel_requested_at IS NULL
  AND lease_expires_at>clock_timestamp())`,
			lease.NamespaceID, lease.SessionID, lease.TurnID, lease.WorkerID, lease.Fence,
		).Scan(&valid); err != nil {
			return result{}, err
		}
		if !valid {
			return result{}, agentmanagement.ErrLeaseLost
		}
		checkpoint, err := putCheckpointTx(ctx, tx, lease.NamespaceID, value)
		if err != nil {
			return result{}, err
		}
		payload, err := json.Marshal(agentmanagement.ContextCheckpointEvent{
			CheckpointID: checkpoint.ID, ThroughSequence: checkpoint.ThroughSequence,
		})
		if err != nil {
			return result{}, agentmanagement.ErrInvalid
		}
		event, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: lease.NamespaceID,
			SessionID:   lease.SessionID,
			TurnID:      lease.TurnID,
			Origin:      "worker",
			Fence:       &lease.Fence,
			Type:        agentmanagement.EventContextCheckpoint,
			Payload:     payload,
		})
		return result{checkpoint: checkpoint, event: event}, err
	})
	return committed.checkpoint, committed.event, err
}

func putCheckpointTx(
	ctx context.Context, tx *sql.Tx, namespaceID string, value agentmanagement.Checkpoint,
) (agentmanagement.Checkpoint, error) {
	goals, resources, toolResults, decisions, state, canonical, err := encodeCheckpoint(value)
	if err != nil {
		return agentmanagement.Checkpoint{}, err
	}
	digest := sha256.Sum256(canonical)
	if value.Digest != "" {
		expected, parseErr := parseDigest(value.Digest)
		if parseErr != nil || !equalDigest(expected, digest[:]) {
			return agentmanagement.Checkpoint{}, agentmanagement.ErrInvalid
		}
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO agent_context_checkpoints
  (id,namespace_id,session_id,turn_id,through_sequence,summary,unresolved_goals,
   resource_references,tool_result_references,decisions,state,content_digest)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`, value.ID, namespaceID,
		value.SessionID, value.TurnID, value.ThroughSequence, value.Summary, goals,
		resources, toolResults, decisions, state, digest[:])
	if err != nil {
		return agentmanagement.Checkpoint{}, classifyWriteError(err)
	}
	return scanCheckpoint(tx.QueryRowContext(ctx, checkpointSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, value.ID))
}

func (store *Store) LatestCheckpoint(
	ctx context.Context, namespaceID, sessionID string,
) (agentmanagement.Checkpoint, error) {
	return scanCheckpoint(store.db.QueryRowContext(ctx, checkpointSelect+`
 WHERE namespace_id=$1 AND session_id=$2 ORDER BY through_sequence DESC LIMIT 1`, namespaceID, sessionID))
}

const checkpointSelect = `SELECT id::text,session_id::text,turn_id::text,through_sequence,summary,
       unresolved_goals,resource_references,tool_result_references,decisions,state,content_digest,created_at
  FROM agent_context_checkpoints`

func encodeCheckpoint(value agentmanagement.Checkpoint) ([]byte, []byte, []byte, []byte, []byte, []byte, error) {
	goals, err := json.Marshal(value.UnresolvedGoals)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	resources, err := json.Marshal(value.ResourceReferences)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	toolResults, err := json.Marshal(value.ToolResultReferences)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	decisions, err := json.Marshal(value.Decisions)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	var state map[string]any
	if len(value.State) == 0 || len(value.State) > 1<<20 || json.Unmarshal(value.State, &state) != nil || state == nil {
		return nil, nil, nil, nil, nil, nil, agentmanagement.ErrInvalid
	}
	stateBytes, err := json.Marshal(state)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	canonical, err := json.Marshal(struct {
		SessionID       string                              `json:"sessionId"`
		TurnID          string                              `json:"turnId"`
		ThroughSequence int64                               `json:"throughSequence"`
		Summary         string                              `json:"summary"`
		Goals           []string                            `json:"goals"`
		Resources       []agentmanagement.ResourceReference `json:"resources"`
		ToolResults     []string                            `json:"toolResults"`
		Decisions       []string                            `json:"decisions"`
		State           json.RawMessage                     `json:"state"`
	}{
		value.SessionID, value.TurnID, value.ThroughSequence, value.Summary,
		value.UnresolvedGoals, value.ResourceReferences, value.ToolResultReferences, value.Decisions, stateBytes,
	})
	return goals, resources, toolResults, decisions, stateBytes, canonical, err
}

func equalDigest(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	var different byte
	for index := range left {
		different |= left[index] ^ right[index]
	}
	return different == 0
}
