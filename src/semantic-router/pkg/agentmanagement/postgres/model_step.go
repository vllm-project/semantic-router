package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const modelStepSelect = `SELECT id::text,namespace_id::text,session_id::text,turn_id::text,
       ordinal,fence,registry_revision,request_digest,status,COALESCE(stop_reason,''),
       COALESCE(output_digest,'\x'::bytea),started_at,completed_at
  FROM agent_model_steps`

func (store *Store) BeginModelStep(
	ctx context.Context, request agentmanagement.ModelStep,
) (agentmanagement.ModelStep, bool, error) {
	if request.ID == "" || request.Ordinal < 1 || request.WorkerID == "" || request.Fence < 1 ||
		len(request.RequestDigest) != sha256.Size || request.RegistryRevision == "" {
		return agentmanagement.ModelStep{}, false, agentmanagement.ErrInvalid
	}
	type result struct {
		step     agentmanagement.ModelStep
		replayed bool
	}
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		if err := requireActiveModelStepLease(ctx, tx, request); err != nil {
			return result{}, err
		}
		existing, scanModelStepErr := scanModelStep(tx.QueryRowContext(ctx, modelStepSelect+`
 WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND ordinal=$4 FOR UPDATE`,
			request.NamespaceID, request.SessionID, request.TurnID, request.Ordinal))
		if scanModelStepErr == nil {
			if existing.ID != request.ID || existing.RegistryRevision != request.RegistryRevision ||
				subtle.ConstantTimeCompare(existing.RequestDigest, request.RequestDigest) != 1 {
				return result{}, agentmanagement.ErrConflict
			}
			if existing.Status != "started" {
				return result{step: existing, replayed: true}, nil
			}
			if existing.Fence == request.Fence {
				return result{}, agentmanagement.ErrConflict
			}
			if _, err := tx.ExecContext(ctx, `UPDATE agent_model_steps
SET fence=$5,status='unknown',completed_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND ordinal=$4 AND status='started'`,
				request.NamespaceID, request.SessionID, request.TurnID, request.Ordinal, request.Fence); err != nil {
				return result{}, classifyWriteError(err)
			}
			existing.Fence = request.Fence
			existing.Status = "unknown"
			now := time.Now().UTC()
			existing.CompletedAt = &now
			return result{step: existing, replayed: true}, nil
		}
		if !errors.Is(scanModelStepErr, agentmanagement.ErrNotFound) {
			return result{}, scanModelStepErr
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_model_steps
  (id,namespace_id,session_id,turn_id,ordinal,fence,registry_revision,request_digest,status)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'started')`, request.ID, request.NamespaceID,
			request.SessionID, request.TurnID, request.Ordinal, request.Fence,
			request.RegistryRevision, request.RequestDigest); err != nil {
			return result{}, classifyWriteError(err)
		}
		created, scanModelStepErr := scanModelStep(tx.QueryRowContext(ctx, modelStepSelect+`
 WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND ordinal=$4`,
			request.NamespaceID, request.SessionID, request.TurnID, request.Ordinal))
		return result{step: created}, scanModelStepErr
	})
	return value.step, value.replayed, err
}

func (store *Store) CommitModelStep(
	ctx context.Context, request agentmanagement.ModelStepCommit,
) (agentmanagement.ModelStepCommitResult, error) {
	if request.Step.ID == "" || request.Step.Ordinal < 1 || !validModelStopReason(request.Step.StopReason) ||
		len(request.Step.RequestDigest) != sha256.Size || len(request.Events) > 66 ||
		request.Checkpoint.ThroughSequence < 1 {
		return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrInvalid
	}
	canonical, err := canonicalModelStepOutput(request.Step.StopReason, request.Step.ID, request.Events)
	if err != nil {
		return agentmanagement.ModelStepCommitResult{}, err
	}
	digest := sha256.Sum256(canonical)
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ModelStepCommitResult, error) {
		step, commitModelStepErr := scanModelStep(tx.QueryRowContext(ctx, modelStepSelect+`
 WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND id=$4 FOR UPDATE`,
			request.Lease.NamespaceID, request.Lease.SessionID, request.Lease.TurnID, request.Step.ID))
		if commitModelStepErr != nil {
			return agentmanagement.ModelStepCommitResult{}, commitModelStepErr
		}
		if step.Fence != request.Lease.Fence || step.RegistryRevision != request.Lease.RegistryRevision ||
			step.Ordinal != request.Step.Ordinal ||
			subtle.ConstantTimeCompare(step.RequestDigest, request.Step.RequestDigest) != 1 {
			return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrLeaseLost
		}
		if step.Status == "completed" {
			if step.StopReason != request.Step.StopReason || subtle.ConstantTimeCompare(step.OutputDigest, digest[:]) != 1 {
				return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrConflict
			}
			return agentmanagement.ModelStepCommitResult{Step: step}, nil
		}
		if step.Status != "started" {
			return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrConflict
		}
		if err := requireActiveModelStepLease(ctx, tx, agentmanagement.ModelStep{
			NamespaceID: request.Lease.NamespaceID, SessionID: request.Lease.SessionID,
			TurnID: request.Lease.TurnID, WorkerID: request.Lease.WorkerID, Fence: request.Lease.Fence,
		}); err != nil {
			return agentmanagement.ModelStepCommitResult{}, err
		}
		result := agentmanagement.ModelStepCommitResult{Events: make([]agentmanagement.Event, 0, len(request.Events))}
		for _, appendRequest := range request.Events {
			appendRequest.NamespaceID = request.Lease.NamespaceID
			appendRequest.SessionID = request.Lease.SessionID
			appendRequest.TurnID = request.Lease.TurnID
			appendRequest.Origin = "worker"
			appendRequest.Fence = &request.Lease.Fence
			event, appendErr := appendEventTx(ctx, tx, appendRequest)
			if appendErr != nil {
				return agentmanagement.ModelStepCommitResult{}, appendErr
			}
			result.Events = append(result.Events, event)
		}
		if len(result.Events) > 0 {
			request.Checkpoint.ThroughSequence = result.Events[len(result.Events)-1].Sequence
		}
		checkpoint, commitModelStepErr := putCheckpointTx(ctx, tx, request.Lease.NamespaceID, request.Checkpoint)
		if commitModelStepErr != nil {
			return agentmanagement.ModelStepCommitResult{}, commitModelStepErr
		}
		checkpointPayload, commitModelStepErr := json.Marshal(agentmanagement.ContextCheckpointEvent{
			CheckpointID: checkpoint.ID, ThroughSequence: checkpoint.ThroughSequence,
		})
		if commitModelStepErr != nil {
			return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrInvalid
		}
		checkpointEvent, commitModelStepErr := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: request.Lease.NamespaceID, SessionID: request.Lease.SessionID,
			TurnID: request.Lease.TurnID, Origin: "worker", Fence: &request.Lease.Fence,
			Type: agentmanagement.EventContextCheckpoint, Payload: checkpointPayload,
		})
		if commitModelStepErr != nil {
			return agentmanagement.ModelStepCommitResult{}, commitModelStepErr
		}
		updated, commitModelStepErr := tx.ExecContext(ctx, `UPDATE agent_model_steps step
SET status='completed',stop_reason=$6,output_digest=$7,completed_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND id=$4 AND fence=$5 AND status='started'
  AND EXISTS (SELECT 1 FROM agent_turns turn
    WHERE turn.namespace_id=step.namespace_id AND turn.session_id=step.session_id
      AND turn.id=step.turn_id AND turn.fence=step.fence AND turn.status='running'
      AND turn.cancel_requested_at IS NULL AND turn.lease_expires_at>clock_timestamp())`,
			request.Lease.NamespaceID, request.Lease.SessionID, request.Lease.TurnID,
			request.Step.ID, request.Lease.Fence, request.Step.StopReason, digest[:])
		if commitModelStepErr != nil {
			return agentmanagement.ModelStepCommitResult{}, classifyWriteError(commitModelStepErr)
		}
		if err := requireOneRow(updated); err != nil {
			return agentmanagement.ModelStepCommitResult{}, agentmanagement.ErrLeaseLost
		}
		step, commitModelStepErr = scanModelStep(tx.QueryRowContext(ctx, modelStepSelect+`
 WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND id=$4`,
			request.Lease.NamespaceID, request.Lease.SessionID, request.Lease.TurnID, request.Step.ID))
		result.Step, result.Checkpoint, result.CheckpointEvent = step, checkpoint, checkpointEvent
		return result, commitModelStepErr
	})
}

func validModelStopReason(value string) bool {
	switch value {
	case "end_turn", "max_tokens", "stop_sequence", "tool_call", "content_filter":
		return true
	default:
		return false
	}
}

func requireActiveModelStepLease(
	ctx context.Context, tx *sql.Tx, request agentmanagement.ModelStep,
) error {
	var valid bool
	if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5 AND status='running'
  AND cancel_requested_at IS NULL AND lease_expires_at>clock_timestamp())`,
		request.NamespaceID, request.SessionID, request.TurnID, request.WorkerID, request.Fence).Scan(&valid); err != nil {
		return fmt.Errorf("validate Agent model step fence: %w", err)
	}
	if !valid {
		return agentmanagement.ErrLeaseLost
	}
	return nil
}

func canonicalModelStepOutput(
	stopReason string, modelStepID string, events []agentmanagement.EventAppend,
) ([]byte, error) {
	type eventValue struct {
		Type    agentmanagement.EventType `json:"type"`
		Payload json.RawMessage           `json:"payload"`
	}
	values := make([]eventValue, 0, len(events))
	summaryCount := 0
	for _, event := range events {
		if event.Type != agentmanagement.EventAssistantDelta &&
			event.Type != agentmanagement.EventModelStepSummary &&
			event.Type != agentmanagement.EventToolRequest {
			return nil, agentmanagement.ErrInvalid
		}
		normalized, err := agentmanagement.NormalizeEventAppend(agentmanagement.EventAppend{
			NamespaceID: event.NamespaceID, SessionID: event.SessionID, TurnID: event.TurnID,
			Origin: "worker", Fence: pointerFence(1), Type: event.Type, Payload: event.Payload,
		})
		if err != nil {
			return nil, err
		}
		switch normalized.Type {
		case agentmanagement.EventAssistantDelta:
			var payload agentmanagement.AssistantDeltaEvent
			if err := json.Unmarshal(normalized.Payload, &payload); err != nil || payload.ModelStepID != modelStepID {
				return nil, agentmanagement.ErrInvalid
			}
		case agentmanagement.EventModelStepSummary:
			var payload agentmanagement.ModelStepSummaryEvent
			if err := json.Unmarshal(normalized.Payload, &payload); err != nil ||
				payload.ModelStepID != modelStepID || summaryCount != 0 {
				return nil, agentmanagement.ErrInvalid
			}
			summaryCount++
		}
		values = append(values, eventValue{Type: normalized.Type, Payload: normalized.Payload})
	}
	if summaryCount != 1 {
		return nil, agentmanagement.ErrInvalid
	}
	return json.Marshal(struct {
		StopReason string       `json:"stopReason"`
		Events     []eventValue `json:"events"`
	}{StopReason: stopReason, Events: values})
}

func pointerFence(value int64) *int64 { return &value }

func scanModelStep(scanner rowScanner) (agentmanagement.ModelStep, error) {
	var value agentmanagement.ModelStep
	var completedAt sql.NullTime
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.SessionID, &value.TurnID, &value.Ordinal,
		&value.Fence, &value.RegistryRevision, &value.RequestDigest, &value.Status,
		&value.StopReason, &value.OutputDigest, &value.StartedAt, &completedAt,
	)
	if err != nil {
		return agentmanagement.ModelStep{}, mapNotFound(err)
	}
	if completedAt.Valid {
		value.CompletedAt = &completedAt.Time
	}
	return value, nil
}
