package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const invocationSelect = `SELECT id::text,namespace_id::text,session_id::text,turn_id::text,fence,
       registry_revision,tool_name,COALESCE(credential_version_id::text,''),input_digest,input,idempotency,classification,status,
       result,artifact_id::text,error_code,started_at,completed_at
  FROM agent_tool_invocations`

func (store *Store) BeginInvocation(
	ctx context.Context, request agentmanagement.InvocationRecord,
) (agentmanagement.InvocationRecord, bool, error) {
	payload, beginInvocationErr := json.Marshal(agentmanagement.ToolRequestEvent{
		InvocationID: request.ID, ToolName: request.ToolName,
		Arguments: request.Input, Class: request.Class,
	})
	if beginInvocationErr != nil {
		return agentmanagement.InvocationRecord{}, false, agentmanagement.ErrInvalid
	}
	fence := request.Fence
	normalized, beginInvocationErr := agentmanagement.NormalizeEventAppend(agentmanagement.EventAppend{
		NamespaceID: request.NamespaceID, SessionID: request.SessionID, TurnID: request.TurnID,
		Origin: "worker", Fence: &fence, Type: agentmanagement.EventToolRequest, Payload: payload,
	})
	if beginInvocationErr != nil {
		return agentmanagement.InvocationRecord{}, false, beginInvocationErr
	}
	var safeRequest agentmanagement.ToolRequestEvent
	if err := json.Unmarshal(normalized.Payload, &safeRequest); err != nil {
		return agentmanagement.InvocationRecord{}, false, agentmanagement.ErrInvalid
	}
	request.Input = append(json.RawMessage(nil), safeRequest.Arguments...)
	digest := sha256.Sum256(request.Input)
	request.InputDigest = digest[:]
	type result struct {
		invocation agentmanagement.InvocationRecord
		replayed   bool
	}
	value, beginInvocationErr := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		var valid bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND fence=$4 AND status='running'
	AND cancel_requested_at IS NULL AND lease_expires_at>clock_timestamp())`, request.NamespaceID, request.SessionID,
			request.TurnID, request.Fence).Scan(&valid); err != nil {
			return result{}, fmt.Errorf("validate Agent invocation fence: %w", err)
		}
		if !valid {
			return result{}, agentmanagement.ErrLeaseLost
		}
		existing, getErr := scanInvocation(tx.QueryRowContext(ctx, invocationSelect+`
 WHERE namespace_id=$1 AND turn_id=$2 AND id=$3 FOR UPDATE`, request.NamespaceID, request.TurnID, request.ID))
		if getErr == nil {
			if existing.ToolName != request.ToolName || existing.RegistryRevision != request.RegistryRevision ||
				existing.CredentialVersionID != request.CredentialVersionID ||
				subtle.ConstantTimeCompare(existing.InputDigest, request.InputDigest) != 1 ||
				existing.Idempotency != request.Idempotency || existing.Class != request.Class {
				return result{}, agentmanagement.ErrConflict
			}
			if existing.Status != "started" {
				return result{invocation: existing, replayed: true}, nil
			}
			if existing.Idempotency != agentmanagement.ToolInvocationIdempotent {
				if _, err := tx.ExecContext(ctx, `UPDATE agent_tool_invocations
SET fence=$4,status='unknown',error_code='side_effect_outcome_unknown',completed_at=clock_timestamp()
WHERE namespace_id=$1 AND turn_id=$2 AND id=$3 AND status='started'`,
					request.NamespaceID, request.TurnID, request.ID, request.Fence); err != nil {
					return result{}, classifyWriteError(err)
				}
				payload, err := json.Marshal(agentmanagement.ToolResultEvent{
					InvocationID: request.ID,
					ToolName:     request.ToolName,
					Status:       "failed",
					Error: &agentmanagement.Failure{
						Code:      "side_effect_outcome_unknown",
						Message:   "The previous tool attempt ended before its outcome was recorded.",
						Retryable: false,
					},
				})
				if err != nil {
					return result{}, agentmanagement.ErrInvalid
				}
				if _, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
					NamespaceID: request.NamespaceID,
					SessionID:   request.SessionID,
					TurnID:      request.TurnID,
					Origin:      "worker",
					Fence:       &request.Fence,
					Type:        agentmanagement.EventToolResult,
					Payload:     payload,
				}); err != nil {
					return result{}, err
				}
				existing.Fence = request.Fence
				existing.Status, existing.ErrorCode = "unknown", "side_effect_outcome_unknown"
				return result{invocation: existing, replayed: true}, nil
			}
			if _, err := tx.ExecContext(ctx, `UPDATE agent_tool_invocations
SET fence=$4,started_at=clock_timestamp(),completed_at=NULL,error_code=NULL
WHERE namespace_id=$1 AND turn_id=$2 AND id=$3 AND status='started'`,
				request.NamespaceID, request.TurnID, request.ID, request.Fence); err != nil {
				return result{}, classifyWriteError(err)
			}
			existing.Fence = request.Fence
			return result{invocation: existing}, nil
		}
		if !errors.Is(getErr, agentmanagement.ErrNotFound) {
			return result{}, getErr
		}
		if !json.Valid(request.Input) {
			return result{}, agentmanagement.ErrInvalid
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_invocations
  (id,namespace_id,session_id,turn_id,fence,registry_revision,tool_name,credential_version_id,input_digest,input,
   idempotency,classification,status)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'started')`, request.ID,
			request.NamespaceID, request.SessionID, request.TurnID, request.Fence,
			request.RegistryRevision, request.ToolName, nullableString(request.CredentialVersionID), request.InputDigest, request.Input,
			request.Idempotency, request.Class); err != nil {
			return result{}, classifyWriteError(err)
		}
		created, err := scanInvocation(tx.QueryRowContext(ctx, invocationSelect+`
 WHERE namespace_id=$1 AND turn_id=$2 AND id=$3`, request.NamespaceID, request.TurnID, request.ID))
		return result{invocation: created}, err
	})
	return value.invocation, value.replayed, beginInvocationErr
}

func (store *Store) FinishInvocation(
	ctx context.Context, request agentmanagement.InvocationRecord,
) (agentmanagement.Event, error) {
	if request.Status != "completed" && request.Status != "failed" && request.Status != "unknown" {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	if len(request.Result) > 0 && !json.Valid(request.Result) {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	eventStatus := "completed"
	var eventFailure *agentmanagement.Failure
	if request.Status != "completed" {
		eventStatus = "failed"
		eventFailure = &agentmanagement.Failure{
			Code: request.ErrorCode, Message: "Tool execution did not complete.", Retryable: false,
		}
	}
	payload, err := json.Marshal(agentmanagement.ToolResultEvent{
		InvocationID: request.ID, ToolName: request.ToolName, Status: eventStatus,
		Result: request.Result, ArtifactID: request.ArtifactID, Error: eventFailure,
	})
	if err != nil {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	fence := request.Fence
	normalized, err := agentmanagement.NormalizeEventAppend(agentmanagement.EventAppend{
		NamespaceID: request.NamespaceID, SessionID: request.SessionID, TurnID: request.TurnID,
		Origin: "worker", Fence: &fence, Type: agentmanagement.EventToolResult, Payload: payload,
	})
	if err != nil {
		return agentmanagement.Event{}, err
	}
	var safeResult agentmanagement.ToolResultEvent
	if err := json.Unmarshal(normalized.Payload, &safeResult); err != nil {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	request.Result = append(json.RawMessage(nil), safeResult.Result...)
	request.ArtifactID = safeResult.ArtifactID
	var resultValue, artifactID, errorCode any
	if len(request.Result) > 0 {
		resultValue = request.Result
	}
	if request.ArtifactID != "" {
		artifactID = request.ArtifactID
	}
	if request.ErrorCode != "" {
		errorCode = request.ErrorCode
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Event, error) {
		existing, finishInvocationErr := scanInvocation(tx.QueryRowContext(ctx, invocationSelect+`
 WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND id=$4 FOR UPDATE`,
			request.NamespaceID, request.SessionID, request.TurnID, request.ID))
		if finishInvocationErr != nil {
			return agentmanagement.Event{}, finishInvocationErr
		}
		if existing.Fence != request.Fence || existing.RegistryRevision != request.RegistryRevision {
			return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
		}
		if existing.Status != "started" {
			if existing.Status == request.Status && existing.ArtifactID == request.ArtifactID &&
				existing.ErrorCode == request.ErrorCode && string(existing.Result) == string(request.Result) {
				return agentmanagement.Event{}, nil
			}
			return agentmanagement.Event{}, agentmanagement.ErrConflict
		}
		result, finishInvocationErr := tx.ExecContext(ctx, `UPDATE agent_tool_invocations invocation
SET status=$7,result=$8,artifact_id=$9,error_code=$10,completed_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND id=$4 AND fence=$5
  AND registry_revision=$6 AND status='started'
  AND EXISTS (SELECT 1 FROM agent_turns turn
    WHERE turn.namespace_id=invocation.namespace_id AND turn.session_id=invocation.session_id
      AND turn.id=invocation.turn_id AND turn.fence=invocation.fence AND turn.status='running'
      AND turn.lease_expires_at>clock_timestamp())`, request.NamespaceID, request.SessionID,
			request.TurnID, request.ID, request.Fence, request.RegistryRevision, request.Status,
			resultValue, artifactID, errorCode)
		if finishInvocationErr != nil {
			return agentmanagement.Event{}, classifyWriteError(finishInvocationErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
		}
		event, finishInvocationErr := appendEventTx(ctx, tx, normalized)
		if finishInvocationErr != nil {
			return agentmanagement.Event{}, finishInvocationErr
		}
		return event, nil
	})
}

func (store *Store) GetInvocation(
	ctx context.Context, namespaceID, turnID, invocationID, registryRevision string,
) (agentmanagement.InvocationRecord, error) {
	return scanInvocation(store.db.QueryRowContext(ctx, invocationSelect+`
 WHERE namespace_id=$1 AND turn_id=$2 AND id=$3 AND registry_revision=$4`,
		namespaceID, turnID, invocationID, registryRevision))
}
