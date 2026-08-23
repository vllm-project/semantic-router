package postgres

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

func scanSubjectUser(scanner rowScanner) (subjectmanagement.User, error) {
	var user subjectmanagement.User
	var deletedAt sql.NullTime
	if err := scanner.Scan(&user.ID, &user.NamespaceID, &user.Email, &user.DisplayName, &user.Status,
		&user.Revision, &user.CreatedAt, &user.UpdatedAt, &deletedAt); err != nil {
		return subjectmanagement.User{}, err
	}
	user.CreatedAt, user.UpdatedAt = user.CreatedAt.UTC(), user.UpdatedAt.UTC()
	if deletedAt.Valid {
		value := deletedAt.Time.UTC()
		user.DeletedAt = &value
	}
	return user, nil
}

func scanSubjectTeam(scanner rowScanner) (subjectmanagement.Team, error) {
	var team subjectmanagement.Team
	var deletedAt sql.NullTime
	if err := scanner.Scan(&team.ID, &team.NamespaceID, &team.Name, &team.Description, &team.Status,
		&team.Revision, &team.CreatedAt, &team.UpdatedAt, &deletedAt); err != nil {
		return subjectmanagement.Team{}, err
	}
	team.CreatedAt, team.UpdatedAt = team.CreatedAt.UTC(), team.UpdatedAt.UTC()
	if deletedAt.Valid {
		value := deletedAt.Time.UTC()
		team.DeletedAt = &value
	}
	return team, nil
}

func scanSubjectMembership(scanner rowScanner) (subjectmanagement.Membership, error) {
	return scanSubjectMembershipWith(scanner)
}

func scanSubjectMembershipWith(scanner rowScanner, trailing ...any) (subjectmanagement.Membership, error) {
	var membership subjectmanagement.Membership
	columns := []any{
		&membership.NamespaceID, &membership.TeamID, &membership.UserID,
		&membership.Role, &membership.Status, &membership.Revision,
		&membership.CreatedAt, &membership.UpdatedAt,
	}
	columns = append(columns, trailing...)
	if err := scanner.Scan(columns...); err != nil {
		return subjectmanagement.Membership{}, err
	}
	membership.CreatedAt, membership.UpdatedAt = membership.CreatedAt.UTC(), membership.UpdatedAt.UTC()
	return membership, nil
}

func scanTeamDefaults(scanner rowScanner, namespaceID string) (subjectmanagement.TeamDefaults, error) {
	defaults := subjectmanagement.TeamDefaults{NamespaceID: namespaceID}
	if err := scanner.Scan(&defaults.SelfServiceRevision, &defaults.AccessPolicyID,
		&defaults.AccessPolicyRevision, &defaults.RateLimitPolicyID,
		&defaults.RateLimitPolicyRevision); err != nil {
		return subjectmanagement.TeamDefaults{}, err
	}
	return defaults, nil
}

func validateNewSubjectUser(user subjectmanagement.User) error {
	if err := validateSubjectUser(user); err != nil || user.Revision != 1 ||
		user.Status != accesscontrol.UserStatusActive || user.CreatedAt.IsZero() || !user.CreatedAt.Equal(user.UpdatedAt) {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateSubjectUser(user subjectmanagement.User) error {
	domain := accesscontrol.User{
		NamespaceID: accesscontrol.NamespaceID(user.NamespaceID),
		ID:          accesscontrol.UserID(user.ID), Email: user.Email, DisplayName: user.DisplayName,
		Status: user.Status, CreatedAt: user.CreatedAt, UpdatedAt: user.UpdatedAt,
	}
	if domain.Validate() != nil || validateSubjectIDs(user.NamespaceID, user.ID) != nil || user.Revision == 0 || user.DeletedAt != nil {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateNewSubjectTeam(mutation subjectmanagement.CreateTeamMutation) error {
	team := mutation.Team
	if err := validateSubjectTeam(team); err != nil || team.Revision != 1 ||
		team.Status != accesscontrol.TeamStatusActive || team.CreatedAt.IsZero() || !team.CreatedAt.Equal(team.UpdatedAt) ||
		len(mutation.AccessPolicyBindings) == 0 ||
		validateSubjectIDs(team.NamespaceID, mutation.RateLimitAllocation.ID) != nil ||
		validateUUID("RateLimitPolicy id", mutation.RateLimitAllocation.PolicyID) != nil ||
		!validSubjectTeamDefaultSelection(mutation) {
		return subjectmanagement.ErrInvalidRequest
	}
	ids, policies := map[string]struct{}{}, map[string]struct{}{}
	ids[mutation.RateLimitAllocation.ID] = struct{}{}
	previousPolicyID := ""
	for _, binding := range mutation.AccessPolicyBindings {
		if validateSubjectIDs(team.NamespaceID, binding.ID) != nil ||
			validateUUID("AccessPolicy id", binding.PolicyID) != nil ||
			(previousPolicyID != "" && binding.PolicyID <= previousPolicyID) {
			return subjectmanagement.ErrInvalidRequest
		}
		if _, duplicate := ids[binding.ID]; duplicate {
			return subjectmanagement.ErrInvalidRequest
		}
		if _, duplicate := policies[binding.PolicyID]; duplicate {
			return subjectmanagement.ErrInvalidRequest
		}
		ids[binding.ID], policies[binding.PolicyID] = struct{}{}, struct{}{}
		previousPolicyID = binding.PolicyID
	}
	return nil
}

func validSubjectTeamDefaultSelection(mutation subjectmanagement.CreateTeamMutation) bool {
	usesDefaults := mutation.UseDefaultAccessPolicy || mutation.UseDefaultRateLimitPolicy
	if !usesDefaults {
		return mutation.NamespaceDefaults == nil
	}
	if mutation.NamespaceDefaults == nil || mutation.NamespaceDefaults.NamespaceID != mutation.Team.NamespaceID ||
		mutation.NamespaceDefaults.SelfServiceRevision == 0 || mutation.NamespaceDefaults.AccessPolicyRevision == 0 ||
		mutation.NamespaceDefaults.RateLimitPolicyRevision == 0 ||
		validateUUID("default AccessPolicy id", mutation.NamespaceDefaults.AccessPolicyID) != nil ||
		validateUUID("default RateLimitPolicy id", mutation.NamespaceDefaults.RateLimitPolicyID) != nil {
		return false
	}
	if mutation.UseDefaultAccessPolicy &&
		(len(mutation.AccessPolicyBindings) != 1 ||
			mutation.AccessPolicyBindings[0].PolicyID != mutation.NamespaceDefaults.AccessPolicyID) {
		return false
	}
	return !mutation.UseDefaultRateLimitPolicy ||
		mutation.RateLimitAllocation.PolicyID == mutation.NamespaceDefaults.RateLimitPolicyID
}

func validateSubjectTeam(team subjectmanagement.Team) error {
	domain := accesscontrol.Team{
		NamespaceID: accesscontrol.NamespaceID(team.NamespaceID),
		ID:          accesscontrol.TeamID(team.ID), Name: team.Name, Status: team.Status,
		CreatedAt: team.CreatedAt, UpdatedAt: team.UpdatedAt,
	}
	if domain.Validate() != nil || validateSubjectIDs(team.NamespaceID, team.ID) != nil ||
		team.Revision == 0 || team.DeletedAt != nil || strings.TrimSpace(team.Description) != team.Description {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateNewSubjectMembership(membership subjectmanagement.Membership) error {
	if err := validateSubjectMembership(membership); err != nil || membership.Revision != 1 ||
		membership.Status != accesscontrol.MembershipStatusActive || membership.CreatedAt.IsZero() ||
		!membership.CreatedAt.Equal(membership.UpdatedAt) {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateSubjectMembership(membership subjectmanagement.Membership) error {
	domain := accesscontrol.TeamMembership{
		NamespaceID: accesscontrol.NamespaceID(membership.NamespaceID),
		TeamID:      accesscontrol.TeamID(membership.TeamID), UserID: accesscontrol.UserID(membership.UserID),
		Role: membership.Role, Status: membership.Status,
		CreatedAt: membership.CreatedAt, UpdatedAt: membership.UpdatedAt,
	}
	if domain.Validate() != nil || validateSubjectMembershipIDs(membership.NamespaceID, membership.TeamID, membership.UserID) != nil ||
		membership.Revision == 0 {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateSubjectList(namespaceID, status, search string, after *subjectmanagement.UserCursor, limit int) error {
	if validateUUID("namespace id", namespaceID) != nil || limit < 1 || limit > 200 {
		return subjectmanagement.ErrInvalidRequest
	}
	normalized, err := managementsearch.Normalize(search)
	if err != nil || normalized != search {
		return subjectmanagement.ErrInvalidRequest
	}
	if after != nil {
		if after.CreatedAt.IsZero() || validateUUID("cursor id", after.ID) != nil {
			return subjectmanagement.ErrInvalidRequest
		}
	}
	return nil
}

func validateSubjectMembershipQuery(query subjectmanagement.MembershipQuery, userList bool) error {
	if query.Limit < 1 || query.Limit > 200 || validateUUID("namespace id", query.NamespaceID) != nil ||
		(query.Status != "" && query.Status != "active" && query.Status != "disabled") {
		return subjectmanagement.ErrInvalidRequest
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return subjectmanagement.ErrInvalidRequest
	}
	ownerID := query.TeamID
	if userList {
		ownerID = query.UserID
	}
	if validateUUID("membership owner id", ownerID) != nil {
		return subjectmanagement.ErrInvalidRequest
	}
	if query.After != nil && (query.After.CreatedAt.IsZero() || validateUUID("membership cursor id", query.After.ID) != nil) {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateSubjectIDs(namespaceID, resourceID string) error {
	if validateUUID("namespace id", namespaceID) != nil || validateUUID("resource id", resourceID) != nil {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func validateSubjectMembershipIDs(namespaceID, teamID, userID string) error {
	if validateSubjectIDs(namespaceID, teamID) != nil || validateUUID("User id", userID) != nil {
		return subjectmanagement.ErrInvalidRequest
	}
	return nil
}

func canonicalSubjectActor(actor subjectmanagement.Actor) bool {
	if validateUUID("principal id", actor.PrincipalID) != nil || strings.TrimSpace(actor.RequestID) == "" {
		return false
	}
	for _, principalID := range actor.ActorChain {
		if validateUUID("actor chain principal", principalID) != nil {
			return false
		}
	}
	return !actor.SourceIP.IsValid() || actor.SourceIP == actor.SourceIP.Unmap()
}

func subjectCursorArgs(cursor *subjectmanagement.UserCursor) (any, any) {
	if cursor == nil {
		return nil, nil
	}
	return cursor.CreatedAt, cursor.ID
}

func membershipCursorArgs(cursor *subjectmanagement.MembershipCursor) (any, any) {
	if cursor == nil {
		return nil, nil
	}
	return cursor.CreatedAt, cursor.ID
}

func trimSubjectPage[T any](items []T, limit int) subjectmanagement.RepositoryPage[T] {
	page := subjectmanagement.RepositoryPage[T]{Items: items}
	if len(items) > limit {
		page.Items = items[:limit]
		page.HasMore = true
	}
	return page
}

func mapSubjectReadError(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) {
		return subjectmanagement.ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapSubjectCreateError(err error, action string) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return subjectmanagement.ErrAlreadyExists
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapSubjectCAS(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) {
		return subjectmanagement.ErrRevisionConflict
	}
	return mapSubjectCreateError(err, action)
}
