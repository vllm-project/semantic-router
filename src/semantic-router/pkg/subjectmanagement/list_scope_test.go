package subjectmanagement

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	listScopeNamespace = "11111111-1111-4111-8111-111111111111"
	listScopeUserOne   = "22222222-2222-4222-8222-222222222222"
	listScopeUserTwo   = "33333333-3333-4333-8333-333333333333"
	listScopeTeamOne   = "44444444-4444-4444-8444-444444444444"
	listScopeTeamTwo   = "55555555-5555-4555-8555-555555555555"
)

type listScopeRepository struct {
	Repository
	userCalls       int
	teamCalls       int
	membershipCalls int
	emptyMembership bool
	lastUserQuery   UserQuery
	lastTeamQuery   TeamQuery
}

func (repository *listScopeRepository) ListUsers(_ context.Context, query UserQuery) (RepositoryPage[User], error) {
	repository.userCalls++
	repository.lastUserQuery = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	if query.After == nil {
		return RepositoryPage[User]{Items: []User{{
			ID: listScopeUserOne, NamespaceID: listScopeNamespace,
			CreatedAt: now, UpdatedAt: now,
		}}, HasMore: true}, nil
	}
	return RepositoryPage[User]{Items: []User{{
		ID: listScopeUserTwo, NamespaceID: listScopeNamespace,
		CreatedAt: now.Add(-time.Second), UpdatedAt: now,
	}}}, nil
}

func (repository *listScopeRepository) ListTeams(_ context.Context, query TeamQuery) (RepositoryPage[Team], error) {
	repository.teamCalls++
	repository.lastTeamQuery = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	if query.After == nil {
		return RepositoryPage[Team]{Items: []Team{{
			ID: listScopeTeamOne, NamespaceID: listScopeNamespace,
			CreatedAt: now, UpdatedAt: now,
		}}, HasMore: true}, nil
	}
	return RepositoryPage[Team]{Items: []Team{{
		ID: listScopeTeamTwo, NamespaceID: listScopeNamespace,
		CreatedAt: now.Add(-time.Second), UpdatedAt: now,
	}}}, nil
}

func (repository *listScopeRepository) ListUserMemberships(
	_ context.Context,
	query MembershipQuery,
) (RepositoryPage[UserMembership], error) {
	repository.membershipCalls++
	if repository.emptyMembership {
		return RepositoryPage[UserMembership]{HasMore: true}, nil
	}
	var totalCount *uint64
	if query.IncludeTotal {
		count := uint64(2)
		totalCount = &count
	}
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	teamID, createdAt, hasMore := listScopeTeamOne, now, true
	if query.After != nil {
		teamID, createdAt, hasMore = listScopeTeamTwo, now.Add(-time.Second), false
	}
	return RepositoryPage[UserMembership]{Items: []UserMembership{{Membership: Membership{
		NamespaceID: listScopeNamespace, UserID: query.UserID, TeamID: teamID,
		CreatedAt: createdAt, UpdatedAt: now,
	}}}, HasMore: hasMore, TotalCount: totalCount}, nil
}

func TestListUsersBindsScopeBeforeStablePagination(t *testing.T) {
	repository := &listScopeRepository{}
	service := listScopeService(t, repository)
	scope := accesscontrol.ResultScope{
		NamespaceID: listScopeNamespace,
		UserIDs:     []accesscontrol.UserID{listScopeUserOne, listScopeUserTwo},
	}
	first, listUsersErr := service.ListUsers(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "  ADA  ", PageSize: 1, Scope: scope,
	})
	if listUsersErr != nil || len(first.Items) != 1 || !first.HasMore || first.NextCursor == "" || repository.userCalls != 1 {
		t.Fatalf("first page = %#v, calls = %d, error = %v", first, repository.userCalls, listUsersErr)
	}
	swapped := accesscontrol.ResultScope{
		NamespaceID: listScopeNamespace,
		UserIDs:     []accesscontrol.UserID{listScopeUserOne},
	}
	if _, err := service.ListUsers(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "ada", PageSize: 1, Scope: swapped, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.userCalls != 1 {
		t.Fatalf("scope-swapped cursor error = %v, calls = %d", err, repository.userCalls)
	}
	if _, err := service.ListUsers(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "grace", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.userCalls != 1 {
		t.Fatalf("search-swapped cursor error = %v, calls = %d", err, repository.userCalls)
	}
	second, listUsersErr := service.ListUsers(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "ada", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	})
	if listUsersErr != nil || len(second.Items) != 1 || second.Items[0].ID != listScopeUserTwo ||
		second.HasMore || second.NextCursor != "" || repository.userCalls != 2 {
		t.Fatalf("second page = %#v, calls = %d, error = %v", second, repository.userCalls, listUsersErr)
	}
	if repository.lastUserQuery.Scope.All || len(repository.lastUserQuery.Scope.UserIDs) != 2 ||
		repository.lastUserQuery.Search != "ada" {
		t.Fatalf("repository scope = %#v", repository.lastUserQuery.Scope)
	}
	empty, listUsersErr := service.ListUsers(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, PageSize: 17,
		Scope: accesscontrol.ResultScope{
			NamespaceID: listScopeNamespace,
			TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne},
		},
	})
	if listUsersErr != nil || len(empty.Items) != 0 || empty.PageSize != 17 || repository.userCalls != 2 {
		t.Fatalf("empty page = %#v, calls = %d, error = %v", empty, repository.userCalls, listUsersErr)
	}
}

func TestListTeamsBindsSearchToSecondPage(t *testing.T) {
	repository := &listScopeRepository{}
	service := listScopeService(t, repository)
	scope := accesscontrol.ResultScope{
		NamespaceID: listScopeNamespace,
		TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne, listScopeTeamTwo},
	}
	first, listTeamsErr := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "  Platform  ", PageSize: 1, Scope: scope,
	})
	if listTeamsErr != nil || first.NextCursor == "" || !first.HasMore {
		t.Fatalf("first Team page = %#v, error = %v", first, listTeamsErr)
	}
	if _, err := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "other", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.teamCalls != 1 {
		t.Fatalf("search-swapped Team cursor error = %v, calls = %d", err, repository.teamCalls)
	}
	second, listTeamsErr := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, Search: "platform", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	})
	if listTeamsErr != nil || len(second.Items) != 1 || second.Items[0].ID != listScopeTeamTwo ||
		second.HasMore || repository.lastTeamQuery.Search != "platform" {
		t.Fatalf("second Team page = %#v, query = %#v, error = %v", second, repository.lastTeamQuery, listTeamsErr)
	}
}

func TestListMembershipsBindsTeamScopeToCursor(t *testing.T) {
	repository := &listScopeRepository{}
	service := listScopeService(t, repository)
	firstScope := accesscontrol.ResultScope{
		NamespaceID: listScopeNamespace,
		TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne},
	}
	first, err := service.ListUserMemberships(context.Background(), MembershipListRequest{
		NamespaceID: listScopeNamespace, UserID: listScopeUserOne, PageSize: 1,
		IncludeTotal: true, Scope: firstScope,
	})
	if err != nil || first.NextCursor == "" || !first.HasMore || repository.membershipCalls != 1 ||
		first.TotalCount == nil || *first.TotalCount != 2 {
		t.Fatalf("first membership page = %#v, calls = %d, error = %v", first, repository.membershipCalls, err)
	}
	if _, err := service.ListUserMemberships(context.Background(), MembershipListRequest{
		NamespaceID: listScopeNamespace, UserID: listScopeUserOne, PageSize: 1,
		Scope: accesscontrol.ResultScope{
			NamespaceID: listScopeNamespace,
			TeamIDs:     []accesscontrol.TeamID{listScopeTeamTwo},
		}, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.membershipCalls != 1 {
		t.Fatalf("membership scope-swap error = %v, calls = %d", err, repository.membershipCalls)
	}
	second, err := service.ListUserMemberships(context.Background(), MembershipListRequest{
		NamespaceID: listScopeNamespace, UserID: listScopeUserOne, PageSize: 1,
		IncludeTotal: true, Scope: firstScope, Cursor: first.NextCursor,
	})
	if err != nil || second.HasMore || len(second.Items) != 1 ||
		second.Items[0].TeamID != listScopeTeamTwo || second.TotalCount == nil || *second.TotalCount != 2 {
		t.Fatalf("second membership page = %#v, error = %v", second, err)
	}
}

func TestListMembershipsRejectsEmptyContinuedPage(t *testing.T) {
	repository := &listScopeRepository{emptyMembership: true}
	service := listScopeService(t, repository)
	_, err := service.ListUserMemberships(context.Background(), MembershipListRequest{
		NamespaceID: listScopeNamespace,
		UserID:      listScopeUserOne,
		PageSize:    1,
		Scope: accesscontrol.ResultScope{
			NamespaceID: listScopeNamespace,
			TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne},
		},
	})
	if !errors.Is(err, ErrUnavailable) || repository.membershipCalls != 1 {
		t.Fatalf("empty continued membership page error = %v, calls = %d", err, repository.membershipCalls)
	}
}

func TestListTeamsBindsExactTeamScopeBeforePagination(t *testing.T) {
	repository := &listScopeRepository{}
	service := listScopeService(t, repository)
	scope := accesscontrol.ResultScope{
		NamespaceID: listScopeNamespace,
		TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne, listScopeTeamTwo},
	}
	first, listTeamsErr := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, PageSize: 1, Scope: scope,
	})
	if listTeamsErr != nil || !first.HasMore || first.NextCursor == "" || len(first.Items) != 1 || repository.teamCalls != 1 {
		t.Fatalf("first Team page = %#v, calls = %d, error = %v", first, repository.teamCalls, listTeamsErr)
	}
	if _, err := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, PageSize: 1, Cursor: first.NextCursor,
		Scope: accesscontrol.ResultScope{
			NamespaceID: listScopeNamespace,
			TeamIDs:     []accesscontrol.TeamID{listScopeTeamOne},
		},
	}); !errors.Is(err, ErrInvalidRequest) || repository.teamCalls != 1 {
		t.Fatalf("Team scope-swap error = %v, calls = %d", err, repository.teamCalls)
	}
	second, listTeamsErr := service.ListTeams(context.Background(), ListRequest{
		NamespaceID: listScopeNamespace, PageSize: 1, Cursor: first.NextCursor, Scope: scope,
	})
	if listTeamsErr != nil || second.HasMore || len(second.Items) != 1 || second.Items[0].ID != listScopeTeamTwo ||
		repository.teamCalls != 2 || repository.lastTeamQuery.After == nil ||
		repository.lastTeamQuery.After.ID != listScopeTeamOne {
		t.Fatalf("second Team page = %#v, query = %#v, error = %v", second, repository.lastTeamQuery, listTeamsErr)
	}
}

func listScopeService(t *testing.T, repository Repository) *Service {
	t.Helper()
	cursors, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("s", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cursors.close)
	return &Service{repository: repository, cursors: cursors}
}
