package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

const agentTurnResourceType = "agent_turn"

func (store *Store) CreateTurn(
	ctx context.Context, request agentmanagement.CreateTurnRequest,
) (agentmanagement.Turn, bool, error) {
	type result struct {
		turn     agentmanagement.Turn
		replayed bool
	}
	created, err := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		stored, replayed, createTurnErr := commandpostgres.Lock(ctx, tx, request.Command)
		if createTurnErr != nil {
			return result{}, createTurnErr
		}
		if replayed {
			if stored.Resource == nil || stored.Resource.ResourceType != agentTurnResourceType {
				return result{}, managementcommand.ErrConflict
			}
			turn, err := scanTurn(tx.QueryRowContext(ctx, turnSelect+`
 WHERE turn.namespace_id=$1 AND turn.session_id=$2 AND turn.id=$3`,
				request.NamespaceID, request.Turn.SessionID, stored.Resource.ResourceID))
			if err != nil {
				return result{}, err
			}
			turn.Revision = int64(stored.Resource.ResourceRevision)
			return result{turn: turn, replayed: true}, nil
		}
		var sessionStatus agentmanagement.SessionStatus
		if err := tx.QueryRowContext(ctx, `SELECT status FROM agent_sessions
WHERE namespace_id=$1 AND id=$2 FOR UPDATE`, request.NamespaceID, request.Turn.SessionID).Scan(&sessionStatus); err != nil {
			return result{}, mapNotFound(err)
		}
		if sessionStatus != agentmanagement.SessionActive {
			return result{}, agentmanagement.ErrDenied
		}
		var nonterminalExists bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(
  SELECT 1 FROM agent_turns
  WHERE namespace_id=$1 AND session_id=$2
    AND status IN ('queued','running','waiting_approval'))`,
			request.NamespaceID, request.Turn.SessionID).Scan(&nonterminalExists); err != nil {
			return result{}, fmt.Errorf("check active Agent Turn: %w", err)
		}
		if nonterminalExists {
			return result{}, agentmanagement.ErrConflict
		}
		var ordinal int64
		if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(ordinal),0)+1 FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2`, request.NamespaceID, request.Turn.SessionID).Scan(&ordinal); err != nil {
			return result{}, fmt.Errorf("allocate Agent Turn ordinal: %w", err)
		}
		input, createTurnErr := json.Marshal(request.Turn.Input)
		if createTurnErr != nil {
			return result{}, agentmanagement.ErrInvalid
		}
		digest := request.Command.ActiveDigest()
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_turns
  (id,namespace_id,session_id,ordinal,actor_principal_id,idempotency_hmac_version,
   idempotency_key_digest,request_digest,input,status,registry_revision,revision)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'queued',$10,1)`, request.Turn.ID,
			request.NamespaceID, request.Turn.SessionID, ordinal, request.ActorPrincipalID,
			digest.HMACVersion, digest.KeyDigest[:], digest.RequestDigest[:], input,
			request.Turn.RegistryRevision); err != nil {
			return result{}, classifyWriteError(err)
		}
		payload, _ := json.Marshal(agentmanagement.UserInputEvent{Content: request.Turn.Input.Content})
		if _, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: request.NamespaceID, SessionID: request.Turn.SessionID,
			TurnID: request.Turn.ID, Origin: "control", Type: agentmanagement.EventUserInput, Payload: payload,
		}); err != nil {
			return result{}, err
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, managementcommand.ResourceResult{
			ResourceType: agentTurnResourceType, ResourceID: request.Turn.ID,
			ResourceRevision: 1, ResponseStatus: 201,
		}); err != nil {
			return result{}, err
		}
		turn, createTurnErr := scanTurn(tx.QueryRowContext(ctx, turnSelect+`
 WHERE turn.namespace_id=$1 AND turn.session_id=$2 AND turn.id=$3`,
			request.NamespaceID, request.Turn.SessionID, request.Turn.ID))
		if createTurnErr != nil {
			return result{}, createTurnErr
		}
		return result{turn: turn}, nil
	})
	return created.turn, created.replayed, err
}

func (store *Store) ListTurns(
	ctx context.Context, namespaceID, sessionID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.Turn], returnErr error) {
	statement := turnSelect + `
 WHERE turn.namespace_id=$1 AND turn.session_id=$2
   AND ($3::timestamptz IS NULL OR (turn.created_at,turn.id)<($3,$4::uuid))
 ORDER BY turn.created_at DESC,turn.id DESC LIMIT $5`
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.Timestamp, query.After.ID
	}
	rows, err := store.db.QueryContext(ctx, statement, namespaceID, sessionID, afterTime, afterID, query.Limit)
	if err != nil {
		return agentmanagement.ListResult[agentmanagement.Turn]{}, fmt.Errorf("list Agent Turns: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Turn, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanTurn(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.Turn]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.Turn]{}, fmt.Errorf("iterate Agent Turns: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.Turn]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetTurn(
	ctx context.Context, namespaceID, sessionID, turnID string,
) (agentmanagement.Turn, error) {
	return scanTurn(store.db.QueryRowContext(ctx, turnSelect+`
 WHERE turn.namespace_id=$1 AND turn.session_id=$2 AND turn.id=$3`, namespaceID, sessionID, turnID))
}

// ClaimNextTurn is the durable queue arbitration point. It claims only queued
// work or an expired running lease, skips rows held by another transaction,
// and increments the PostgreSQL fence before a worker can emit an event.
func (store *Store) ClaimNextTurn(
	ctx context.Context, workerID string, expiresAt time.Time,
) (agentmanagement.TurnLease, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.TurnLease, error) {
		row := tx.QueryRowContext(ctx, `WITH candidate AS (
  SELECT id FROM agent_turns
  WHERE (status='queued' AND cancel_requested_at IS NULL)
     OR (status='running' AND lease_expires_at<=clock_timestamp())
  ORDER BY created_at,id
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE agent_turns turn
SET status='running',fence=turn.fence+1,worker_id=$1,lease_expires_at=$2,
    started_at=COALESCE(turn.started_at,clock_timestamp()),revision=turn.revision+1,
    updated_at=clock_timestamp()
FROM candidate
WHERE turn.id=candidate.id
RETURNING turn.namespace_id::text,turn.session_id::text,turn.id::text,turn.worker_id,
          turn.fence,turn.registry_revision,turn.lease_expires_at`, workerID, expiresAt.UTC())
		var lease agentmanagement.TurnLease
		if err := row.Scan(&lease.NamespaceID, &lease.SessionID, &lease.TurnID, &lease.WorkerID,
			&lease.Fence, &lease.RegistryRevision, &lease.ExpiresAt); err != nil {
			return agentmanagement.TurnLease{}, mapNotFound(err)
		}
		return lease, nil
	})
}

func (store *Store) RenewTurn(
	ctx context.Context, lease agentmanagement.TurnLease, expiresAt time.Time,
) (agentmanagement.TurnLease, error) {
	row := store.db.QueryRowContext(ctx, `UPDATE agent_turns
SET lease_expires_at=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5
  AND status='running' AND lease_expires_at>clock_timestamp()
RETURNING lease_expires_at`, lease.NamespaceID, lease.SessionID, lease.TurnID,
		lease.WorkerID, lease.Fence, expiresAt.UTC())
	if err := row.Scan(&lease.ExpiresAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return agentmanagement.TurnLease{}, agentmanagement.ErrLeaseLost
		}
		return agentmanagement.TurnLease{}, fmt.Errorf("renew Agent Turn lease: %w", err)
	}
	return lease, nil
}

func (store *Store) TransitionTurn(
	ctx context.Context, transition agentmanagement.TurnTransition,
) (agentmanagement.Event, error) {
	lease, status, failure := transition.Lease, transition.Status, transition.Failure
	if status != agentmanagement.TurnCompleted && status != agentmanagement.TurnFailed &&
		status != agentmanagement.TurnCancelled && status != agentmanagement.TurnWaitingApproval {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	if (status == agentmanagement.TurnWaitingApproval) != (transition.Approval != nil) {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	var completion any
	if status != agentmanagement.TurnWaitingApproval {
		completion = transition.CompletedAt.UTC()
	}
	var failureCode, failureMessage any
	if failure != nil {
		failureCode, failureMessage = failure.Code, failure.Message
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Event, error) {
		var cancellationRequested bool
		if err := tx.QueryRowContext(ctx, `SELECT cancel_requested_at IS NOT NULL
FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5 AND status='running'
FOR UPDATE`, lease.NamespaceID, lease.SessionID, lease.TurnID, lease.WorkerID, lease.Fence).Scan(
			&cancellationRequested); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
			}
			return agentmanagement.Event{}, fmt.Errorf("lock Agent Turn transition: %w", err)
		}
		if cancellationRequested {
			status, failure = agentmanagement.TurnCancelled, nil
			transition.Approval = nil
			completion, failureCode, failureMessage = transition.CompletedAt.UTC(), nil, nil
		}
		eventType := agentmanagement.EventTerminal
		var eventPayload any = agentmanagement.TerminalEvent{Status: status, Error: failure}
		if status == agentmanagement.TurnWaitingApproval {
			eventType, eventPayload = agentmanagement.EventApprovalRequest, *transition.Approval
		}
		payload, err := json.Marshal(eventPayload)
		if err != nil {
			return agentmanagement.Event{}, agentmanagement.ErrInvalid
		}
		event, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: lease.NamespaceID, SessionID: lease.SessionID, TurnID: lease.TurnID,
			Origin: "worker", Fence: &lease.Fence, Type: eventType, Payload: payload,
		})
		if err != nil {
			return agentmanagement.Event{}, err
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_turns
SET status=$6,worker_id=NULL,lease_expires_at=NULL,completed_at=$7,
    failure_code=$8,failure_message=$9,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5 AND status='running'`,
			lease.NamespaceID, lease.SessionID, lease.TurnID, lease.WorkerID, lease.Fence,
			status, completion, failureCode, failureMessage)
		if err != nil {
			return agentmanagement.Event{}, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
		}
		return event, nil
	})
}

func (store *Store) RequestCancellation(
	ctx context.Context, namespaceID, sessionID, turnID string, requestedAt time.Time,
) (agentmanagement.Turn, bool, error) {
	type result struct {
		turn     agentmanagement.Turn
		replayed bool
	}
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		turn, requestCancellationErr := scanTurn(tx.QueryRowContext(ctx, turnSelect+`
 WHERE turn.namespace_id=$1 AND turn.session_id=$2 AND turn.id=$3 FOR UPDATE`, namespaceID, sessionID, turnID))
		if requestCancellationErr != nil {
			return result{}, requestCancellationErr
		}
		if turn.CancelRequestedAt != nil || terminalTurn(turn.Status) {
			return result{turn: turn, replayed: true}, nil
		}
		if turn.Status == agentmanagement.TurnWaitingApproval {
			var publishing bool
			if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_publication_plans
WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND status='publishing')`,
				namespaceID, sessionID, turnID).Scan(&publishing); err != nil {
				return result{}, fmt.Errorf("check Agent publication cancellation fence: %w", err)
			}
			if publishing {
				return result{}, agentmanagement.ErrConflict
			}
		}
		cancelImmediately := turn.Status == agentmanagement.TurnQueued || turn.Status == agentmanagement.TurnWaitingApproval
		if _, err := tx.ExecContext(ctx, `UPDATE agent_turns
SET cancel_requested_at=$4,
    status=CASE WHEN $5 THEN 'cancelled' ELSE status END,
    completed_at=CASE WHEN $5 THEN $4 ELSE completed_at END,
    worker_id=CASE WHEN $5 THEN NULL ELSE worker_id END,
    lease_expires_at=CASE WHEN $5 THEN NULL ELSE lease_expires_at END,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND id=$3`, namespaceID, sessionID, turnID,
			requestedAt.UTC(), cancelImmediately); err != nil {
			return result{}, classifyWriteError(err)
		}
		payload, requestCancellationErr := json.Marshal(agentmanagement.CancellationEvent{RequestedAt: requestedAt.UTC()})
		if requestCancellationErr != nil {
			return result{}, agentmanagement.ErrInvalid
		}
		if _, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
			Origin: "control", Type: agentmanagement.EventCancellation, Payload: payload,
		}); err != nil {
			return result{}, err
		}
		if _, err := tx.ExecContext(ctx, `UPDATE agent_publication_plans
SET status='invalidated',revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND turn_id=$3 AND status='ready'`,
			namespaceID, sessionID, turnID); err != nil {
			return result{}, classifyWriteError(err)
		}
		if cancelImmediately {
			terminalPayload, marshalErr := json.Marshal(agentmanagement.TerminalEvent{Status: agentmanagement.TurnCancelled})
			if marshalErr != nil {
				return result{}, agentmanagement.ErrInvalid
			}
			if _, err := appendEventTx(ctx, tx, agentmanagement.EventAppend{
				NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
				Origin: "control", Type: agentmanagement.EventTerminal, Payload: terminalPayload,
			}); err != nil {
				return result{}, err
			}
		}
		turn, requestCancellationErr = scanTurn(tx.QueryRowContext(ctx, turnSelect+`
 WHERE turn.namespace_id=$1 AND turn.session_id=$2 AND turn.id=$3`, namespaceID, sessionID, turnID))
		return result{turn: turn}, requestCancellationErr
	})
	return value.turn, value.replayed, err
}

func (store *Store) CancellationRequested(ctx context.Context, lease agentmanagement.TurnLease) (bool, error) {
	var requested bool
	err := store.db.QueryRowContext(ctx, `SELECT cancel_requested_at IS NOT NULL
FROM agent_turns WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND worker_id=$4 AND fence=$5`,
		lease.NamespaceID, lease.SessionID, lease.TurnID, lease.WorkerID, lease.Fence).Scan(&requested)
	if errors.Is(err, sql.ErrNoRows) {
		return false, agentmanagement.ErrLeaseLost
	}
	if err != nil {
		return false, fmt.Errorf("read Agent cancellation: %w", err)
	}
	return requested, nil
}

func terminalTurn(status agentmanagement.TurnStatus) bool {
	return status == agentmanagement.TurnCompleted || status == agentmanagement.TurnFailed || status == agentmanagement.TurnCancelled
}
