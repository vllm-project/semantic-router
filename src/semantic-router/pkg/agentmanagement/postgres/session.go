package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func (store *Store) ListSessions(
	ctx context.Context, namespaceID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.Session], returnErr error) {
	sessionIDs := scopedIDs(query.Scope, accesscontrol.ScopeResourceAgentSession)
	teamIDs := typedStrings(query.Scope.TeamIDs)
	userIDs := typedStrings(query.Scope.UserIDs)
	statement := sessionSelect + `
 WHERE session.namespace_id=$1 AND session.status<>'deleted'
   AND ($2 OR session.owner_principal_id=$3 OR session.id=ANY($4::uuid[])
        OR session.effective_team_id=ANY($5::uuid[]) OR session.effective_user_id=ANY($6::uuid[]))
	AND ($7='' OR lower(session.title) LIKE $7 ESCAPE '\')
	AND ($8::timestamptz IS NULL OR (session.updated_at,session.id)<($8,$9::uuid))
 ORDER BY session.updated_at DESC,session.id DESC LIMIT $10`
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.Timestamp, query.After.ID
	}
	owner := nullableString(query.OwnerPrincipalID)
	rows, err := store.db.QueryContext(ctx, statement, namespaceID, query.Scope.All, owner,
		pq.Array(sessionIDs), pq.Array(teamIDs), pq.Array(userIDs),
		managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit)
	if err != nil {
		return agentmanagement.ListResult[agentmanagement.Session]{}, fmt.Errorf("list Agent Sessions: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Session, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanSession(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.Session]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.Session]{}, fmt.Errorf("iterate Agent Sessions: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.Session]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetSession(ctx context.Context, namespaceID, id string) (agentmanagement.Session, error) {
	return scanSession(store.db.QueryRowContext(ctx, sessionSelect+`
 WHERE session.namespace_id=$1 AND session.id=$2 AND session.status<>'deleted'`, namespaceID, id))
}

func (store *Store) PatchSession(
	ctx context.Context, namespaceID, id string, expected int64, patch agentmanagement.SessionPatch,
	_ agentmanagement.MutationContext,
) (agentmanagement.Session, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Session, error) {
		current, err := scanSession(tx.QueryRowContext(ctx, sessionSelect+`
 WHERE session.namespace_id=$1 AND session.id=$2 AND session.status<>'deleted' FOR UPDATE`, namespaceID, id))
		if err != nil {
			return agentmanagement.Session{}, err
		}
		if current.Revision != expected {
			return agentmanagement.Session{}, agentmanagement.ErrConflict
		}
		title, status := current.Title, current.Status
		if patch.Title != nil {
			title = *patch.Title
		}
		if patch.Status != nil {
			status = *patch.Status
		}
		if len(title) > 256 || (status != agentmanagement.SessionActive && status != agentmanagement.SessionClosed) ||
			(current.Status == agentmanagement.SessionClosed && status != agentmanagement.SessionClosed) {
			return agentmanagement.Session{}, agentmanagement.ErrInvalid
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_sessions
SET title=$4,status=$5,closed_at=CASE WHEN $5='closed' THEN COALESCE(closed_at,clock_timestamp()) ELSE NULL END,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`, namespaceID, id, expected, title, status)
		if err != nil {
			return agentmanagement.Session{}, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.Session{}, err
		}
		return scanSession(tx.QueryRowContext(ctx, sessionSelect+` WHERE session.namespace_id=$1 AND session.id=$2`, namespaceID, id))
	})
}

func (store *Store) DeleteSession(
	ctx context.Context, namespaceID, id string, expected int64, _ agentmanagement.MutationContext,
) (int64, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (int64, error) {
		result, err := tx.ExecContext(ctx, `UPDATE agent_sessions
SET status='deleted',revision=revision+1,updated_at=clock_timestamp(),deleted_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='closed'`, namespaceID, id, expected)
		if err != nil {
			return 0, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return 0, err
		}
		return expected + 1, nil
	})
}

func typedStrings[T ~string](values []T) []string {
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = string(value)
	}
	return result
}
