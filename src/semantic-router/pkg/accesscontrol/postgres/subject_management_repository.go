package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

func (s *Store) ReadySubjectManagement(ctx context.Context, codec *managementcommand.Codec) error {
	if s == nil || s.db == nil || codec == nil {
		return subjectmanagement.ErrUnavailable
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, s.db, codec)
}

func (s *Store) ReplaySubjectCommand(
	ctx context.Context,
	command managementcommand.Command,
) (subjectmanagement.MutationResult, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, s.db, command)
	if err != nil || !found {
		return subjectmanagement.MutationResult{}, false, err
	}
	result, err := subjectMutationResult(stored)
	return result, true, err
}

func (s *Store) GetSubjectUser(ctx context.Context, namespaceID, userID string) (subjectmanagement.User, error) {
	if err := validateSubjectIDs(namespaceID, userID); err != nil {
		return subjectmanagement.User{}, err
	}
	user, err := scanSubjectUser(s.db.QueryRowContext(ctx, subjectGetUserQuery, namespaceID, userID))
	return user, mapSubjectReadError(err, "get user")
}

func (s *Store) ListSubjectUsers(
	ctx context.Context,
	query subjectmanagement.UserQuery,
) (subjectmanagement.RepositoryPage[subjectmanagement.User], error) {
	if err := validateSubjectList(query.NamespaceID, query.Status, query.Search, query.After, query.Limit); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.User]{}, err
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return subjectmanagement.RepositoryPage[subjectmanagement.User]{}, subjectmanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.UserIDs) == 0 {
		return subjectmanagement.RepositoryPage[subjectmanagement.User]{Items: []subjectmanagement.User{}}, nil
	}
	afterTime, afterID := subjectCursorArgs(query.After)
	var (
		rows *sql.Rows
		err  error
	)
	if query.Search == "" {
		rows, err = s.db.QueryContext(ctx, subjectListUsersQuery,
			query.NamespaceID, query.Status, query.Scope.All, pq.Array(query.Scope.UserIDs), afterTime, afterID, query.Limit+1)
	} else {
		rows, err = s.db.QueryContext(ctx, subjectSearchUsersQuery,
			query.NamespaceID, query.Status, query.Scope.All, pq.Array(query.Scope.UserIDs),
			managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit+1)
	}
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.User]{}, fmt.Errorf("list users: %w", err)
	}
	defer rows.Close()
	items := make([]subjectmanagement.User, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanSubjectUser(rows)
		if err != nil {
			return subjectmanagement.RepositoryPage[subjectmanagement.User]{}, fmt.Errorf("scan user page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.User]{}, fmt.Errorf("read user page: %w", err)
	}
	return trimSubjectPage(items, query.Limit), nil
}

func (s *Store) CreateSubjectUser(
	ctx context.Context,
	mutation subjectmanagement.CreateUserMutation,
) (subjectmanagement.MutationResult, error) {
	user := mutation.User
	if err := validateNewSubjectUser(user); err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	meta, err := subjectMutationMeta(mutation.Actor, "user.create", "Create User.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		if replay, ok, err := lockSubjectCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		if _, err := tx.ExecContext(ctx, insertSubjectQuery,
			user.NamespaceID, user.ID, accesscontrol.SubjectKindUser, user.CreatedAt); err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCreateError(err, "insert User subject")
		}
		created, err := scanSubjectUser(tx.QueryRowContext(ctx, subjectInsertUserQuery,
			user.ID, user.NamespaceID, user.Email, user.DisplayName, user.CreatedAt))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCreateError(err, "insert User")
		}
		if _, err := appendSubjectMutation(ctx, tx, created.NamespaceID, "user", created.ID, created.Revision, outboxCreated, meta, nil); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return completeSubjectCommand(ctx, tx, mutation.Command, "user", created.ID, created.Revision, 201)
	})
}

func (s *Store) UpdateSubjectUser(
	ctx context.Context,
	user subjectmanagement.User,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectUser(user); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "user.update", "Update User.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		updated, err := scanSubjectUser(tx.QueryRowContext(ctx, subjectUpdateUserQuery,
			user.NamespaceID, user.ID, expected, user.Email, user.DisplayName, user.Status))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCAS(err, "update User")
		}
		if _, err := appendSubjectMutation(ctx, tx, updated.NamespaceID, "user", updated.ID, updated.Revision, outboxUpdated, meta, nil); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return subjectmanagement.MutationResult{Kind: "user", ID: updated.ID, Revision: updated.Revision, HTTPStatus: 200}, nil
	})
}

func (s *Store) DeleteSubjectUser(
	ctx context.Context,
	namespaceID, userID string,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectIDs(namespaceID, userID); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "user.delete", "Delete User.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		deleted, err := scanSubjectUser(tx.QueryRowContext(ctx, subjectDeleteUserQuery, namespaceID, userID, expected))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCAS(err, "delete User")
		}
		if _, err := tx.ExecContext(ctx, subjectDisableUserMembershipsQuery, namespaceID, userID); err != nil {
			return subjectmanagement.MutationResult{}, fmt.Errorf("disable User memberships: %w", err)
		}
		if _, err := appendSubjectMutation(ctx, tx, deleted.NamespaceID, "user", deleted.ID, deleted.Revision, outboxDeleted, meta, nil); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return subjectmanagement.MutationResult{Kind: "user", ID: deleted.ID, Revision: deleted.Revision, HTTPStatus: 204}, nil
	})
}

func (s *Store) ResolveSubjectTeamDefaults(ctx context.Context, namespaceID string) (subjectmanagement.TeamDefaults, error) {
	if err := validateUUID("namespace id", namespaceID); err != nil {
		return subjectmanagement.TeamDefaults{}, subjectmanagement.ErrInvalidRequest
	}
	defaults, err := scanTeamDefaults(s.db.QueryRowContext(ctx, subjectResolveTeamDefaultsQuery, namespaceID), namespaceID)
	if errors.Is(err, sql.ErrNoRows) {
		return subjectmanagement.TeamDefaults{}, subjectmanagement.ErrDefaultsUnavailable
	}
	if err != nil {
		return subjectmanagement.TeamDefaults{}, fmt.Errorf("resolve Team defaults: %w", err)
	}
	return defaults, nil
}

func (s *Store) GetSubjectTeam(ctx context.Context, namespaceID, teamID string) (subjectmanagement.Team, error) {
	if err := validateSubjectIDs(namespaceID, teamID); err != nil {
		return subjectmanagement.Team{}, err
	}
	team, err := scanSubjectTeam(s.db.QueryRowContext(ctx, subjectGetTeamQuery, namespaceID, teamID))
	return team, mapSubjectReadError(err, "get Team")
}

func (s *Store) ListSubjectTeams(
	ctx context.Context,
	query subjectmanagement.TeamQuery,
) (subjectmanagement.RepositoryPage[subjectmanagement.Team], error) {
	if err := validateSubjectList(query.NamespaceID, query.Status, query.Search, query.After, query.Limit); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.Team]{}, err
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return subjectmanagement.RepositoryPage[subjectmanagement.Team]{}, subjectmanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.TeamIDs) == 0 {
		return subjectmanagement.RepositoryPage[subjectmanagement.Team]{Items: []subjectmanagement.Team{}}, nil
	}
	afterTime, afterID := subjectCursorArgs(query.After)
	var (
		rows *sql.Rows
		err  error
	)
	if query.Search == "" {
		rows, err = s.db.QueryContext(ctx, subjectListTeamsQuery,
			query.NamespaceID, query.Status, query.Scope.All, pq.Array(query.Scope.TeamIDs), afterTime, afterID, query.Limit+1)
	} else {
		rows, err = s.db.QueryContext(ctx, subjectSearchTeamsQuery,
			query.NamespaceID, query.Status, query.Scope.All, pq.Array(query.Scope.TeamIDs),
			managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit+1)
	}
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.Team]{}, fmt.Errorf("list Teams: %w", err)
	}
	defer rows.Close()
	items := make([]subjectmanagement.Team, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanSubjectTeam(rows)
		if err != nil {
			return subjectmanagement.RepositoryPage[subjectmanagement.Team]{}, fmt.Errorf("scan Team page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.Team]{}, fmt.Errorf("read Team page: %w", err)
	}
	return trimSubjectPage(items, query.Limit), nil
}

func (s *Store) CreateSubjectTeam(
	ctx context.Context,
	mutation subjectmanagement.CreateTeamMutation,
) (subjectmanagement.MutationResult, error) {
	if err := validateNewSubjectTeam(mutation); err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	meta, err := subjectMutationMeta(mutation.Actor, "team.create", "Create Team with policy allocation.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		if replay, ok, err := lockSubjectCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		if err := lockSubjectTeamPolicySelection(ctx, tx, mutation); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		team := mutation.Team
		if _, err := tx.ExecContext(ctx, insertSubjectQuery,
			team.NamespaceID, team.ID, accesscontrol.SubjectKindTeam, team.CreatedAt); err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCreateError(err, "insert Team subject")
		}
		created, createSubjectTeamErr := scanSubjectTeam(tx.QueryRowContext(ctx, subjectInsertTeamQuery,
			team.ID, team.NamespaceID, team.Name, team.Description, team.CreatedAt))
		if createSubjectTeamErr != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCreateError(createSubjectTeamErr, "insert Team")
		}
		accessPolicyIDs := make([]string, 0, len(mutation.AccessPolicyBindings))
		accessBindingIDs := make([]string, 0, len(mutation.AccessPolicyBindings))
		for _, binding := range mutation.AccessPolicyBindings {
			if _, err := tx.ExecContext(ctx, subjectInsertAccessBindingQuery,
				binding.ID, team.NamespaceID, binding.PolicyID, team.ID, team.CreatedAt); err != nil {
				return subjectmanagement.MutationResult{}, fmt.Errorf("bind Team AccessPolicy: %w", err)
			}
			accessPolicyIDs = append(accessPolicyIDs, binding.PolicyID)
			accessBindingIDs = append(accessBindingIDs, binding.ID)
		}
		result, createSubjectTeamErr := tx.ExecContext(ctx, subjectInsertRateBindingQuery,
			mutation.RateLimitAllocation.ID, team.NamespaceID, mutation.RateLimitAllocation.PolicyID,
			team.ID, team.CreatedAt)
		if createSubjectTeamErr != nil {
			return subjectmanagement.MutationResult{}, fmt.Errorf("bind Team RateLimit allocation: %w", createSubjectTeamErr)
		}
		if err := requireOneRow(result, subjectmanagement.ErrPolicySelectionUnavailable); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		references := map[string]string{
			"accessPolicyIds":        strings.Join(accessPolicyIDs, ","),
			"accessPolicyBindingIds": strings.Join(accessBindingIDs, ","),
			"rateLimitPolicyId":      mutation.RateLimitAllocation.PolicyID,
			"rateLimitBindingId":     mutation.RateLimitAllocation.ID,
		}
		if _, err := appendSubjectMutation(ctx, tx, created.NamespaceID, "team", created.ID, created.Revision,
			outboxCreated, meta, references); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return completeSubjectCommand(ctx, tx, mutation.Command, "team", created.ID, created.Revision, 201)
	})
}

func (s *Store) UpdateSubjectTeam(
	ctx context.Context,
	team subjectmanagement.Team,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectTeam(team); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "team.update", "Update Team.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		updated, err := scanSubjectTeam(tx.QueryRowContext(ctx, subjectUpdateTeamQuery,
			team.NamespaceID, team.ID, expected, team.Name, team.Description, team.Status))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCAS(err, "update Team")
		}
		if _, err := appendSubjectMutation(ctx, tx, updated.NamespaceID, "team", updated.ID, updated.Revision, outboxUpdated, meta, nil); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return subjectmanagement.MutationResult{Kind: "team", ID: updated.ID, Revision: updated.Revision, HTTPStatus: 200}, nil
	})
}

func (s *Store) DeleteSubjectTeam(
	ctx context.Context,
	namespaceID, teamID string,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectIDs(namespaceID, teamID); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "team.delete", "Delete Team.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		deleted, err := scanSubjectTeam(tx.QueryRowContext(ctx, subjectDeleteTeamQuery, namespaceID, teamID, expected))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCAS(err, "delete Team")
		}
		for label, query := range map[string]string{
			"memberships":     subjectDisableTeamMembershipsQuery,
			"access bindings": subjectDisableTeamAccessBindingsQuery,
			"rate bindings":   subjectDisableTeamRateBindingsQuery,
		} {
			if _, err := tx.ExecContext(ctx, query, namespaceID, teamID); err != nil {
				return subjectmanagement.MutationResult{}, fmt.Errorf("disable Team %s: %w", label, err)
			}
		}
		if _, err := appendSubjectMutation(ctx, tx, deleted.NamespaceID, "team", deleted.ID, deleted.Revision, outboxDeleted, meta, nil); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return subjectmanagement.MutationResult{Kind: "team", ID: deleted.ID, Revision: deleted.Revision, HTTPStatus: 204}, nil
	})
}

func (s *Store) GetSubjectMembership(
	ctx context.Context,
	namespaceID, teamID, userID string,
) (subjectmanagement.Membership, error) {
	if err := validateSubjectMembershipIDs(namespaceID, teamID, userID); err != nil {
		return subjectmanagement.Membership{}, err
	}
	membership, err := scanSubjectMembership(s.db.QueryRowContext(ctx,
		subjectGetMembershipQuery, namespaceID, teamID, userID))
	return membership, mapSubjectReadError(err, "get Team membership")
}

func (s *Store) ListSubjectUserMemberships(
	ctx context.Context,
	query subjectmanagement.MembershipQuery,
) (subjectmanagement.RepositoryPage[subjectmanagement.UserMembership], error) {
	if err := validateSubjectMembershipQuery(query, true); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.UserMembership]{}, err
	}
	if !query.Scope.All && len(query.Scope.TeamIDs) == 0 {
		return emptySubjectRelationshipPage[subjectmanagement.UserMembership](query.IncludeTotal), nil
	}
	totalCount, err := s.countSubjectRelationships(ctx, query, subjectCountUserMembershipsQuery)
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.UserMembership]{}, fmt.Errorf("count User memberships: %w", err)
	}
	afterTime, afterID := membershipCursorArgs(query.After)
	rows, err := s.db.QueryContext(ctx, subjectListUserMembershipsQuery,
		query.NamespaceID, query.UserID, query.Status, query.Scope.All, pq.Array(query.Scope.TeamIDs),
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.UserMembership]{}, fmt.Errorf("list User memberships: %w", err)
	}
	defer rows.Close()
	items := make([]subjectmanagement.UserMembership, 0, query.Limit+1)
	for rows.Next() {
		var item subjectmanagement.UserMembership
		membership, err := scanSubjectMembershipWith(rows, &item.TeamName, &item.TeamStatus)
		if err != nil {
			return subjectmanagement.RepositoryPage[subjectmanagement.UserMembership]{}, fmt.Errorf("scan User membership page: %w", err)
		}
		item.Membership = membership
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.UserMembership]{}, fmt.Errorf("read User membership page: %w", err)
	}
	page := trimSubjectPage(items, query.Limit)
	page.TotalCount = totalCount
	return page, nil
}

func (s *Store) ListSubjectTeamMembers(
	ctx context.Context,
	query subjectmanagement.MembershipQuery,
) (subjectmanagement.RepositoryPage[subjectmanagement.TeamMember], error) {
	if err := validateSubjectMembershipQuery(query, false); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.TeamMember]{}, err
	}
	if !query.Scope.All && len(query.Scope.UserIDs) == 0 {
		return emptySubjectRelationshipPage[subjectmanagement.TeamMember](query.IncludeTotal), nil
	}
	totalCount, err := s.countSubjectRelationships(ctx, query, subjectCountTeamMembersQuery)
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.TeamMember]{}, fmt.Errorf("count Team members: %w", err)
	}
	afterTime, afterID := membershipCursorArgs(query.After)
	rows, err := s.db.QueryContext(ctx, subjectListTeamMembersQuery,
		query.NamespaceID, query.TeamID, query.Status, query.Scope.All, pq.Array(query.Scope.UserIDs),
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.TeamMember]{}, fmt.Errorf("list Team members: %w", err)
	}
	defer rows.Close()
	items := make([]subjectmanagement.TeamMember, 0, query.Limit+1)
	for rows.Next() {
		var item subjectmanagement.TeamMember
		membership, err := scanSubjectMembershipWith(rows, &item.DisplayName, &item.Email, &item.UserStatus)
		if err != nil {
			return subjectmanagement.RepositoryPage[subjectmanagement.TeamMember]{}, fmt.Errorf("scan Team member page: %w", err)
		}
		item.Membership = membership
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return subjectmanagement.RepositoryPage[subjectmanagement.TeamMember]{}, fmt.Errorf("read Team member page: %w", err)
	}
	page := trimSubjectPage(items, query.Limit)
	page.TotalCount = totalCount
	return page, nil
}

func emptySubjectRelationshipPage[T any](includeTotal bool) subjectmanagement.RepositoryPage[T] {
	page := subjectmanagement.RepositoryPage[T]{Items: []T{}}
	if includeTotal {
		count := uint64(0)
		page.TotalCount = &count
	}
	return page
}

func (s *Store) countSubjectRelationships(
	ctx context.Context,
	query subjectmanagement.MembershipQuery,
	statement string,
) (*uint64, error) {
	if !query.IncludeTotal {
		return nil, nil
	}
	var ownerID any = query.UserID
	var ids any = query.Scope.TeamIDs
	if query.TeamID != "" {
		ownerID, ids = query.TeamID, query.Scope.UserIDs
	}
	var count int64
	if err := s.db.QueryRowContext(ctx, statement, query.NamespaceID, ownerID, query.Status,
		query.Scope.All, pq.Array(ids)).Scan(&count); err != nil {
		return nil, err
	}
	if count < 0 {
		return nil, errors.New("relationship count is negative")
	}
	result := uint64(count)
	return &result, nil
}

func (s *Store) PutSubjectMembership(
	ctx context.Context,
	mutation subjectmanagement.PutMembershipMutation,
) (subjectmanagement.MutationResult, error) {
	if err := validateNewSubjectMembership(mutation.Membership); err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	meta, err := subjectMutationMeta(mutation.Actor, "team_membership.create", "Add Team member.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		if replay, ok, err := lockSubjectCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		var teamActive, userActive bool
		membership := mutation.Membership
		if err := tx.QueryRowContext(ctx, subjectCheckMembershipParentsQuery,
			membership.NamespaceID, membership.TeamID, membership.UserID).Scan(&teamActive, &userActive); err != nil {
			return subjectmanagement.MutationResult{}, fmt.Errorf("validate Team membership parents: %w", err)
		}
		if !teamActive || !userActive {
			return subjectmanagement.MutationResult{}, subjectmanagement.ErrNotFound
		}
		created, err := scanSubjectMembership(tx.QueryRowContext(ctx, subjectInsertMembershipQuery,
			membership.NamespaceID, membership.TeamID, membership.UserID, membership.Role, membership.CreatedAt))
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCreateError(err, "insert Team membership")
		}
		eventID := membershipEventAggregateID(accesscontrol.TeamMembership{
			NamespaceID: accesscontrol.NamespaceID(created.NamespaceID), TeamID: accesscontrol.TeamID(created.TeamID),
			UserID: accesscontrol.UserID(created.UserID), Role: created.Role, Status: created.Status,
			CreatedAt: created.CreatedAt, UpdatedAt: created.UpdatedAt,
		})
		if _, err := appendSubjectMutation(ctx, tx, created.NamespaceID, "team_membership", eventID, created.Revision,
			outboxCreated, meta, map[string]string{"teamId": created.TeamID, "userId": created.UserID}); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return completeSubjectCommand(ctx, tx, mutation.Command, "team_membership", created.UserID, created.Revision, 200)
	})
}

func (s *Store) UpdateSubjectMembership(
	ctx context.Context,
	membership subjectmanagement.Membership,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectMembership(membership); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "team_membership.update", "Update Team membership.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return s.mutateSubjectMembership(ctx, membership, expected, meta, subjectUpdateMembershipQuery, outboxUpdated, 200)
}

func (s *Store) DeleteSubjectMembership(
	ctx context.Context,
	namespaceID, teamID, userID string,
	expected uint64,
	actor subjectmanagement.Actor,
) (subjectmanagement.MutationResult, error) {
	if err := validateSubjectMembershipIDs(namespaceID, teamID, userID); err != nil || expected == 0 {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrInvalidRequest
	}
	meta, err := subjectMutationMeta(actor, "team_membership.delete", "Remove Team member.", nil)
	if err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	membership := subjectmanagement.Membership{
		NamespaceID: namespaceID, TeamID: teamID, UserID: userID,
		Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusDisabled,
	}
	return s.mutateSubjectMembership(ctx, membership, expected, meta, subjectDeleteMembershipQuery, outboxDeleted, 204)
}

func (s *Store) mutateSubjectMembership(
	ctx context.Context,
	membership subjectmanagement.Membership,
	expected uint64,
	meta MutationMeta,
	query string,
	operation outboxOperation,
	status int,
) (subjectmanagement.MutationResult, error) {
	return inTransaction(ctx, s, func(tx *sql.Tx) (subjectmanagement.MutationResult, error) {
		var updated subjectmanagement.Membership
		var err error
		if query == subjectDeleteMembershipQuery {
			updated, err = scanSubjectMembership(tx.QueryRowContext(ctx, query,
				membership.NamespaceID, membership.TeamID, membership.UserID, expected))
		} else {
			updated, err = scanSubjectMembership(tx.QueryRowContext(ctx, query,
				membership.NamespaceID, membership.TeamID, membership.UserID, expected,
				membership.Role, membership.Status))
		}
		if err != nil {
			return subjectmanagement.MutationResult{}, mapSubjectCAS(err, "mutate Team membership")
		}
		eventID := membershipEventAggregateID(accesscontrol.TeamMembership{
			NamespaceID: accesscontrol.NamespaceID(updated.NamespaceID), TeamID: accesscontrol.TeamID(updated.TeamID),
			UserID: accesscontrol.UserID(updated.UserID), Role: updated.Role, Status: updated.Status,
			CreatedAt: updated.CreatedAt, UpdatedAt: updated.UpdatedAt,
		})
		if _, err := appendSubjectMutation(ctx, tx, updated.NamespaceID, "team_membership", eventID, updated.Revision,
			operation, meta, map[string]string{"teamId": updated.TeamID, "userId": updated.UserID}); err != nil {
			return subjectmanagement.MutationResult{}, err
		}
		return subjectmanagement.MutationResult{
			Kind: "team_membership", ID: updated.UserID,
			Revision: updated.Revision, HTTPStatus: status,
		}, nil
	})
}

func appendSubjectMutation(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	kind, id string,
	revision uint64,
	operation outboxOperation,
	meta MutationMeta,
	references map[string]string,
) (MutationReceipt, error) {
	return appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespaceID), outboxMutation{
		AggregateType: kind, AggregateID: id, AggregateRevision: accesscontrol.Revision(revision),
		Operation: operation, References: references,
	}, meta)
}

func subjectMutationMeta(
	actor subjectmanagement.Actor,
	action, reason string,
	details map[string]string,
) (MutationMeta, error) {
	if !canonicalSubjectActor(actor) {
		return MutationMeta{}, subjectmanagement.ErrInvalidRequest
	}
	if details == nil {
		details = make(map[string]string)
	}
	// Namespace is added by callers before append; it is excluded from public audit details below.
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(actor.ActorChain[index])
	}
	return MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: action, Reason: reason, Details: AuditDetails(details),
	}, nil
}

func lockSubjectCommand(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
) (subjectmanagement.MutationResult, bool, error) {
	stored, replayed, err := commandpostgres.Lock(ctx, tx, command)
	if err != nil || !replayed {
		return subjectmanagement.MutationResult{}, false, err
	}
	result, err := subjectMutationResult(stored)
	return result, true, err
}

func completeSubjectCommand(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	kind, id string,
	revision uint64,
	status int,
) (subjectmanagement.MutationResult, error) {
	if err := commandpostgres.CompleteResource(ctx, tx, command, managementcommand.ResourceResult{
		ResourceType: kind, ResourceID: id, ResourceRevision: revision, ResponseStatus: status,
	}); err != nil {
		return subjectmanagement.MutationResult{}, err
	}
	return subjectmanagement.MutationResult{Kind: kind, ID: id, Revision: revision, Idempotent: true, HTTPStatus: status}, nil
}

func subjectMutationResult(stored managementcommand.StoredResult) (subjectmanagement.MutationResult, error) {
	if stored.Resource == nil {
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrUnavailable
	}
	resource := stored.Resource
	switch resource.ResourceType {
	case "user", "team", "team_membership":
	default:
		return subjectmanagement.MutationResult{}, subjectmanagement.ErrUnavailable
	}
	return subjectmanagement.MutationResult{
		Kind: resource.ResourceType, ID: resource.ResourceID,
		Revision: resource.ResourceRevision, Idempotent: true, Replayed: true, HTTPStatus: resource.ResponseStatus,
	}, nil
}
