package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func loadUsers(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[string]accesscontrol.User, _ []Barrier, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, email, display_name, status, created_at, updated_at, deleted_at
FROM access_users WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list access users: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string]accesscontrol.User)
	barriers := make([]Barrier, 0)
	for rows.Next() {
		var user accesscontrol.User
		var deletedAt sql.NullTime
		user.NamespaceID = namespaceID
		if err := rows.Scan(&user.ID, &user.Email, &user.DisplayName, &user.Status, &user.CreatedAt, &user.UpdatedAt, &deletedAt); err != nil {
			return nil, nil, fmt.Errorf("scan access user: %w", err)
		}
		if deletedAt.Valid {
			user.Status = accesscontrol.UserStatusDeleted
		}
		result[string(user.ID)] = user
		if user.Status != accesscontrol.UserStatusActive {
			barriers = append(barriers, Barrier{Kind: "user", ResourceID: string(user.ID), Reason: "user_inactive"})
		}
	}
	return result, barriers, rows.Err()
}

func loadTeams(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[string]accesscontrol.Team, _ []Barrier, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, name, status, created_at, updated_at, deleted_at
FROM access_teams WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list access teams: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string]accesscontrol.Team)
	barriers := make([]Barrier, 0)
	for rows.Next() {
		var team accesscontrol.Team
		var deletedAt sql.NullTime
		team.NamespaceID = namespaceID
		if err := rows.Scan(&team.ID, &team.Name, &team.Status, &team.CreatedAt, &team.UpdatedAt, &deletedAt); err != nil {
			return nil, nil, fmt.Errorf("scan access team: %w", err)
		}
		result[string(team.ID)] = team
		if team.Status != accesscontrol.TeamStatusActive || deletedAt.Valid {
			barriers = append(barriers, Barrier{Kind: "team", ResourceID: string(team.ID), Reason: "team_inactive"})
		}
	}
	return result, barriers, rows.Err()
}

func loadMemberships(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[string]accesscontrol.TeamMembership, _ []Barrier, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT team_id, user_id, role, status, created_at, updated_at
FROM access_team_memberships WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list team memberships: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string]accesscontrol.TeamMembership)
	barriers := make([]Barrier, 0)
	for rows.Next() {
		var membership accesscontrol.TeamMembership
		membership.NamespaceID = namespaceID
		if err := rows.Scan(&membership.TeamID, &membership.UserID, &membership.Role, &membership.Status,
			&membership.CreatedAt, &membership.UpdatedAt); err != nil {
			return nil, nil, fmt.Errorf("scan team membership: %w", err)
		}
		identity := membershipIdentity(membership.TeamID, membership.UserID)
		result[identity] = membership
		if membership.Status != accesscontrol.MembershipStatusActive {
			barriers = append(barriers, Barrier{Kind: "membership", ResourceID: identity, Reason: "membership_inactive"})
		}
	}
	return result, barriers, rows.Err()
}

func loadSubjects(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[string]accesscontrol.SubjectRef, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, kind FROM access_subjects WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, fmt.Errorf("list access subjects: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string]accesscontrol.SubjectRef)
	for rows.Next() {
		var id string
		var kind accesscontrol.SubjectKind
		if err := rows.Scan(&id, &kind); err != nil {
			return nil, fmt.Errorf("scan access subject: %w", err)
		}
		result[id] = accesscontrol.SubjectRef{NamespaceID: namespaceID, ID: accesscontrol.SubjectID(id), Kind: kind}
	}
	return result, rows.Err()
}
