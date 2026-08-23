package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

type SubjectManagementService interface {
	Ready(context.Context) error
	ResolveTeamDefaults(context.Context, string) (subjectmanagement.TeamDefaults, error)

	GetUser(context.Context, string, string) (subjectmanagement.User, error)
	ListUsers(context.Context, subjectmanagement.ListRequest) (subjectmanagement.Page[subjectmanagement.User], error)
	CreateUser(context.Context, subjectmanagement.CreateUserRequest) (subjectmanagement.MutationResult, error)
	UpdateUser(context.Context, subjectmanagement.UpdateUserRequest) (subjectmanagement.MutationResult, error)
	DeleteUser(context.Context, subjectmanagement.DeleteUserRequest) (subjectmanagement.MutationResult, error)

	GetTeam(context.Context, string, string) (subjectmanagement.Team, error)
	ListTeams(context.Context, subjectmanagement.ListRequest) (subjectmanagement.Page[subjectmanagement.Team], error)
	CreateTeam(context.Context, subjectmanagement.CreateTeamRequest) (subjectmanagement.MutationResult, error)
	UpdateTeam(context.Context, subjectmanagement.UpdateTeamRequest) (subjectmanagement.MutationResult, error)
	DeleteTeam(context.Context, subjectmanagement.DeleteTeamRequest) (subjectmanagement.MutationResult, error)

	ListUserMemberships(context.Context, subjectmanagement.MembershipListRequest) (subjectmanagement.Page[subjectmanagement.UserMembership], error)
	ListTeamMembers(context.Context, subjectmanagement.MembershipListRequest) (subjectmanagement.Page[subjectmanagement.TeamMember], error)
	PutMembership(context.Context, subjectmanagement.PutMembershipRequest) (subjectmanagement.MutationResult, error)
	UpdateMembership(context.Context, subjectmanagement.UpdateMembershipRequest) (subjectmanagement.MutationResult, error)
	DeleteMembership(context.Context, subjectmanagement.DeleteMembershipRequest) (subjectmanagement.MutationResult, error)
}

type SubjectRoutesOptions struct {
	Service       SubjectManagementService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}
