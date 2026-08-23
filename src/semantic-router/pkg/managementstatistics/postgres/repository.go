// Package postgres implements the Management statistics read model with one
// bounded, scope-pushed PostgreSQL statement.
package postgres

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

const statisticsQuery = `SELECT
CASE WHEN $4::boolean THEN (
  SELECT count(*)::text FROM access_users u
  WHERE u.namespace_id=$1 AND u.deleted_at IS NULL
    AND ($5::boolean OR u.id=ANY($6::uuid[]))
) END,
CASE WHEN $7::boolean THEN (
  SELECT count(*)::text FROM access_teams t
  WHERE t.namespace_id=$1 AND t.deleted_at IS NULL
    AND ($8::boolean OR t.id=ANY($9::uuid[]))
) END,
CASE WHEN $10::boolean THEN (
  SELECT count(*)::text FROM access_api_keys k
  WHERE k.namespace_id=$1 AND k.deleted_at IS NULL AND k.status='active'
    AND (k.expires_at IS NULL OR k.expires_at>$2)
    AND ($11::boolean OR k.id=ANY($12::uuid[]) OR k.owner_user_id=ANY($13::uuid[])
         OR k.owner_team_id=ANY($14::uuid[]))
) END,
CASE WHEN $10::boolean THEN (
  SELECT count(*)::text FROM access_api_keys k
  WHERE k.namespace_id=$1 AND k.deleted_at IS NULL AND k.status='active'
    AND k.expires_at>$2 AND k.expires_at<=$3
    AND ($11::boolean OR k.id=ANY($12::uuid[]) OR k.owner_user_id=ANY($13::uuid[])
         OR k.owner_team_id=ANY($14::uuid[]))
) END,
CASE WHEN $15::boolean THEN (
  SELECT count(*)::text FROM access_policies p
  WHERE p.namespace_id=$1 AND ($16::boolean OR p.id=ANY($17::uuid[]))
) END,
CASE WHEN $18::boolean THEN (
  SELECT count(*)::text FROM rate_limit_policies p
  WHERE p.namespace_id=$1 AND p.status='active'
    AND ($19::boolean OR p.id=ANY($20::uuid[]))
) END`

type Repository struct{ db *sql.DB }

func New(db *sql.DB) (*Repository, error) {
	if db == nil {
		return nil, managementstatistics.ErrUnavailable
	}
	return &Repository{db: db}, nil
}

func (repository *Repository) Ready(ctx context.Context) error {
	if repository == nil || repository.db == nil {
		return managementstatistics.ErrUnavailable
	}
	return repository.db.PingContext(ctx)
}

func (repository *Repository) Snapshot(
	ctx context.Context,
	query managementstatistics.Query,
) (managementstatistics.Snapshot, error) {
	if repository == nil || repository.db == nil || validateQuery(query) != nil {
		return managementstatistics.Snapshot{}, managementstatistics.ErrInvalidRequest
	}
	usersEnabled, usersAll, userIDs := userScope(query.Scopes.Users)
	teamsEnabled, teamsAll, teamIDs := teamScope(query.Scopes.Teams)
	keysEnabled, keysAll, keyIDs := apiKeyScope(query.Scopes.APIKeys)
	accessEnabled, accessAll, accessIDs := resourceScope(query.Scopes.AccessPolicies, accesscontrol.ScopeResourceAccessPolicy)
	rateEnabled, rateAll, rateIDs := resourceScope(query.Scopes.RatePolicies, accesscontrol.ScopeResourceRateLimitPolicy)

	var users, teams, activeKeys, expiringKeys, accessPolicies, activeRatePolicies sql.NullString
	err := repository.db.QueryRowContext(ctx, statisticsQuery,
		query.NamespaceID, query.AsOf, query.ExpiringBefore,
		usersEnabled, usersAll, pq.Array(userIDs),
		teamsEnabled, teamsAll, pq.Array(teamIDs),
		keysEnabled, keysAll, pq.Array(keyIDs), pq.Array(scopeUserIDs(query.Scopes.APIKeys)), pq.Array(scopeTeamIDs(query.Scopes.APIKeys)),
		accessEnabled, accessAll, pq.Array(accessIDs),
		rateEnabled, rateAll, pq.Array(rateIDs),
	).Scan(&users, &teams, &activeKeys, &expiringKeys, &accessPolicies, &activeRatePolicies)
	if err != nil {
		return managementstatistics.Snapshot{}, fmt.Errorf("%w: read statistics: %w", managementstatistics.ErrUnavailable, err)
	}
	snapshot := managementstatistics.Snapshot{AsOf: query.AsOf, ExpiringBefore: query.ExpiringBefore}
	destinations := []**managementstatistics.Count{
		&snapshot.Users, &snapshot.Teams, &snapshot.ActiveAPIKeys, &snapshot.ExpiringAPIKeys,
		&snapshot.AccessPolicies, &snapshot.ActiveRatePolicies,
	}
	for index, source := range []sql.NullString{
		users, teams, activeKeys, expiringKeys, accessPolicies, activeRatePolicies,
	} {
		count, countErr := optionalCount(source)
		if countErr != nil {
			return managementstatistics.Snapshot{}, countErr
		}
		*destinations[index] = count
	}
	return snapshot, nil
}

func validateQuery(query managementstatistics.Query) error {
	if _, err := uuid.Parse(query.NamespaceID); err != nil || query.AsOf.IsZero() ||
		!query.AsOf.Before(query.ExpiringBefore) {
		return managementstatistics.ErrInvalidRequest
	}
	for _, scope := range []*accesscontrol.ResultScope{
		query.Scopes.Users, query.Scopes.Teams, query.Scopes.APIKeys,
		query.Scopes.AccessPolicies, query.Scopes.RatePolicies,
	} {
		if scope == nil {
			continue
		}
		canonical, err := scope.Canonical()
		if err != nil || canonical.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
			return managementstatistics.ErrInvalidRequest
		}
	}
	return nil
}

func userScope(scope *accesscontrol.ResultScope) (bool, bool, []string) {
	if scope == nil {
		return false, false, nil
	}
	return true, scope.All, scopeUserIDs(scope)
}

func teamScope(scope *accesscontrol.ResultScope) (bool, bool, []string) {
	if scope == nil {
		return false, false, nil
	}
	return true, scope.All, scopeTeamIDs(scope)
}

func apiKeyScope(scope *accesscontrol.ResultScope) (bool, bool, []string) {
	if scope == nil {
		return false, false, nil
	}
	ids := make([]string, len(scope.APIKeyIDs))
	for index, id := range scope.APIKeyIDs {
		ids[index] = string(id)
	}
	return true, scope.All, ids
}

func resourceScope(
	scope *accesscontrol.ResultScope,
	resourceType accesscontrol.ScopeResourceType,
) (bool, bool, []string) {
	if scope == nil {
		return false, false, nil
	}
	resourceIDs := scope.IDs(resourceType)
	ids := make([]string, len(resourceIDs))
	for index, id := range resourceIDs {
		ids[index] = string(id)
	}
	return true, scope.All, ids
}

func scopeUserIDs(scope *accesscontrol.ResultScope) []string {
	if scope == nil {
		return nil
	}
	result := make([]string, len(scope.UserIDs))
	for index, id := range scope.UserIDs {
		result[index] = string(id)
	}
	return result
}

func scopeTeamIDs(scope *accesscontrol.ResultScope) []string {
	if scope == nil {
		return nil
	}
	result := make([]string, len(scope.TeamIDs))
	for index, id := range scope.TeamIDs {
		result[index] = string(id)
	}
	return result
}

func optionalCount(value sql.NullString) (*managementstatistics.Count, error) {
	if !value.Valid {
		return nil, nil
	}
	count := managementstatistics.Count(value.String)
	if !count.Valid() {
		return nil, fmt.Errorf("%w: PostgreSQL returned an invalid count", managementstatistics.ErrUnavailable)
	}
	return &count, nil
}

var _ managementstatistics.Repository = (*Repository)(nil)
