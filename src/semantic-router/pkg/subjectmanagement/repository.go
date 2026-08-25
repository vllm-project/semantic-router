package subjectmanagement

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type UserCursor struct {
	CreatedAt time.Time
	ID        string
}

type TeamCursor = UserCursor

type MembershipCursor struct {
	CreatedAt time.Time
	ID        string
}

type UserQuery struct {
	NamespaceID string
	Status      string
	Search      string
	Scope       accesscontrol.ResultScope
	After       *UserCursor
	Limit       int
}

type TeamQuery struct {
	NamespaceID string
	Status      string
	Search      string
	Scope       accesscontrol.ResultScope
	After       *TeamCursor
	Limit       int
}

type MembershipQuery struct {
	NamespaceID  string
	UserID       string
	TeamID       string
	Status       string
	Scope        accesscontrol.ResultScope
	After        *MembershipCursor
	Limit        int
	IncludeTotal bool
}

type RepositoryPage[T any] struct {
	Items      []T
	HasMore    bool
	TotalCount *uint64
}

type Repository interface {
	RepositoryLifecycle
	UserRepository
	TeamDefaultsRepository
	TeamRepository
	MembershipReader
	MembershipMutationRepository
}

type RepositoryLifecycle interface {
	Ready(context.Context, *managementcommand.Codec) error
	Replay(context.Context, managementcommand.Command) (MutationResult, bool, error)
}

type UserRepository interface {
	GetUser(context.Context, string, string) (User, error)
	ListUsers(context.Context, UserQuery) (RepositoryPage[User], error)
	CreateUser(context.Context, CreateUserMutation) (MutationResult, error)
	UpdateUser(context.Context, User, uint64, Actor) (MutationResult, error)
	DeleteUser(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type TeamDefaultsRepository interface {
	ResolveTeamDefaults(context.Context, string) (TeamDefaults, error)
}

type TeamRepository interface {
	GetTeam(context.Context, string, string) (Team, error)
	ListTeams(context.Context, TeamQuery) (RepositoryPage[Team], error)
	CreateTeam(context.Context, CreateTeamMutation) (MutationResult, error)
	UpdateTeam(context.Context, Team, uint64, Actor) (MutationResult, error)
	DeleteTeam(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type MembershipReader interface {
	GetMembership(context.Context, string, string, string) (Membership, error)
	ListUserMemberships(context.Context, MembershipQuery) (RepositoryPage[UserMembership], error)
	ListTeamMembers(context.Context, MembershipQuery) (RepositoryPage[TeamMember], error)
}

type MembershipMutationRepository interface {
	PutMembership(context.Context, PutMembershipMutation) (MutationResult, error)
	UpdateMembership(context.Context, Membership, uint64, Actor) (MutationResult, error)
	DeleteMembership(context.Context, string, string, string, uint64, Actor) (MutationResult, error)
}
