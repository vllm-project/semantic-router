package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func (store *Store) AppendEvent(
	ctx context.Context, appendRequest agentmanagement.EventAppend,
) (agentmanagement.Event, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Event, error) {
		return appendEventTx(ctx, tx, appendRequest)
	})
}

func appendEventTx(
	ctx context.Context, tx *sql.Tx, appendRequest agentmanagement.EventAppend,
) (agentmanagement.Event, error) {
	appendRequest, appendEventTxErr := agentmanagement.NormalizeEventAppend(appendRequest)
	if appendEventTxErr != nil {
		return agentmanagement.Event{}, appendEventTxErr
	}
	if appendRequest.Origin == "worker" {
		if appendRequest.Fence == nil || *appendRequest.Fence < 1 {
			return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
		}
		var valid bool
		err := tx.QueryRowContext(ctx, `SELECT EXISTS(
  SELECT 1 FROM agent_turns
  WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND fence=$4
	AND status='running' AND lease_expires_at>clock_timestamp()
	AND (cancel_requested_at IS NULL OR $5 IN ('tool_result','terminal')))`,
			appendRequest.NamespaceID, appendRequest.SessionID, appendRequest.TurnID,
			*appendRequest.Fence, appendRequest.Type).Scan(&valid)
		if err != nil {
			return agentmanagement.Event{}, fmt.Errorf("validate Agent event fence: %w", err)
		}
		if !valid {
			return agentmanagement.Event{}, agentmanagement.ErrLeaseLost
		}
	} else if appendRequest.Origin != "control" || appendRequest.Fence != nil {
		return agentmanagement.Event{}, agentmanagement.ErrInvalid
	}
	var sequence int64
	if err := tx.QueryRowContext(ctx, `UPDATE agent_sessions
SET next_sequence=next_sequence+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND status<>'deleted'
RETURNING next_sequence-1`, appendRequest.NamespaceID, appendRequest.SessionID).Scan(&sequence); err != nil {
		return agentmanagement.Event{}, mapNotFound(err)
	}
	var createdAt time.Time
	appendEventTxErr = tx.QueryRowContext(ctx, `INSERT INTO agent_events
  (namespace_id,session_id,sequence,turn_id,origin,fence,event_type,payload)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
RETURNING created_at`, appendRequest.NamespaceID, appendRequest.SessionID, sequence,
		appendRequest.TurnID, appendRequest.Origin, nullableFence(appendRequest.Fence),
		appendRequest.Type, appendRequest.Payload).Scan(&createdAt)
	if appendEventTxErr != nil {
		return agentmanagement.Event{}, classifyWriteError(appendEventTxErr)
	}
	return agentmanagement.Event{
		SessionID: appendRequest.SessionID, TurnID: appendRequest.TurnID, Sequence: sequence,
		Type: appendRequest.Type, Payload: append([]byte(nil), appendRequest.Payload...),
		CreatedAt: createdAt.UTC(),
	}, nil
}

func (store *Store) ListEventsAfter(
	ctx context.Context, namespaceID, sessionID string, after int64, limit int,
) (_ []agentmanagement.Event, _ bool, returnErr error) {
	statement := `SELECT session_id::text,turn_id::text,sequence,event_type,payload,created_at
FROM agent_events WHERE namespace_id=$1 AND session_id=$2 AND sequence>$3
ORDER BY sequence ASC LIMIT $4`
	rows, listEventsAfterErr := store.db.QueryContext(ctx, statement, namespaceID, sessionID, after, limit)
	if listEventsAfterErr != nil {
		return nil, false, fmt.Errorf("list Agent events: %w", listEventsAfterErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Event, 0, limit)
	for rows.Next() {
		value, scanErr := scanEvent(rows)
		if scanErr != nil {
			return nil, false, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("iterate Agent events: %w", err)
	}
	if len(items) == 0 {
		return items, false, nil
	}
	var exists bool
	listEventsAfterErr = store.db.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_events
WHERE namespace_id=$1 AND session_id=$2 AND sequence>$3)`, namespaceID, sessionID, items[len(items)-1].Sequence).Scan(&exists)
	if listEventsAfterErr != nil {
		return nil, false, fmt.Errorf("page Agent events: %w", listEventsAfterErr)
	}
	return items, exists, nil
}

func (store *Store) ListEventHistory(
	ctx context.Context,
	namespaceID string,
	sessionID string,
	query agentmanagement.EventHistoryQuery,
) (_ []agentmanagement.Event, _ bool, returnErr error) {
	rows, err := store.db.QueryContext(ctx, `SELECT session_id::text,turn_id::text,sequence,event_type,payload,created_at
FROM (SELECT session_id,turn_id,sequence,event_type,payload,created_at
        FROM agent_events
       WHERE namespace_id=$1 AND session_id=$2
         AND ($3::bigint=0 OR sequence<$3)
       ORDER BY sequence DESC
       LIMIT $4) recent
ORDER BY sequence ASC`, namespaceID, sessionID, query.BeforeSequence, query.Limit)
	if err != nil {
		return nil, false, fmt.Errorf("list Agent event history: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Event, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanEvent(rows)
		if scanErr != nil {
			return nil, false, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("iterate Agent event history: %w", err)
	}
	if len(items) == 0 {
		return items, false, nil
	}
	var hasMore bool
	if err := store.db.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM agent_events
WHERE namespace_id=$1 AND session_id=$2 AND sequence<$3)`,
		namespaceID, sessionID, items[0].Sequence).Scan(&hasMore); err != nil {
		return nil, false, fmt.Errorf("page Agent event history: %w", err)
	}
	return items, hasMore, nil
}

func (store *Store) OldestEventSequence(
	ctx context.Context, namespaceID, sessionID string,
) (int64, error) {
	var sequence int64
	err := store.db.QueryRowContext(ctx, `SELECT COALESCE(MIN(sequence),0) FROM agent_events
WHERE namespace_id=$1 AND session_id=$2`, namespaceID, sessionID).Scan(&sequence)
	if err != nil {
		return 0, fmt.Errorf("read oldest Agent event: %w", err)
	}
	if sequence == 0 {
		return 0, agentmanagement.ErrNotFound
	}
	return sequence, nil
}

func nullableFence(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}
