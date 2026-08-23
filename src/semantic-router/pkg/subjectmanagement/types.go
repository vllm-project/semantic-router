package subjectmanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
}

type User struct {
	ID          string
	NamespaceID string
	Email       string
	DisplayName string
	Status      accesscontrol.UserStatus
	Revision    uint64
	CreatedAt   time.Time
	UpdatedAt   time.Time
	DeletedAt   *time.Time
}

type Team struct {
	ID          string
	NamespaceID string
	Name        string
	Description string
	Status      accesscontrol.TeamStatus
	Revision    uint64
	CreatedAt   time.Time
	UpdatedAt   time.Time
	DeletedAt   *time.Time
}

type Membership struct {
	NamespaceID string
	TeamID      string
	UserID      string
	Role        accesscontrol.TeamRole
	Status      accesscontrol.MembershipStatus
	Revision    uint64
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

type TeamDefaults struct {
	NamespaceID             string
	SelfServiceRevision     uint64
	AccessPolicyID          string
	AccessPolicyRevision    uint64
	RateLimitPolicyID       string
	RateLimitPolicyRevision uint64
}

type UserMembership struct {
	Membership
	TeamName   string
	TeamStatus accesscontrol.TeamStatus
}

type TeamMember struct {
	Membership
	DisplayName string
	UserStatus  accesscontrol.UserStatus
}

type MutationResult struct {
	Kind       string
	ID         string
	Revision   uint64
	Idempotent bool
	Replayed   bool
	HTTPStatus int
}

type Page[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
	PageSize   int
}

type ListRequest struct {
	NamespaceID string
	Status      string
	Search      string
	Cursor      string
	PageSize    int
	Scope       accesscontrol.ResultScope
}

type MembershipListRequest struct {
	NamespaceID string
	UserID      string
	TeamID      string
	Status      accesscontrol.MembershipStatus
	Cursor      string
	PageSize    int
	Scope       accesscontrol.ResultScope
}

type CreateUserRequest struct {
	NamespaceID    string
	Email          string
	DisplayName    string
	IdempotencyKey string
	Actor          Actor
}

type UpdateUserRequest struct {
	NamespaceID      string
	UserID           string
	ExpectedRevision uint64
	Email            *string
	DisplayName      *string
	Status           *accesscontrol.UserStatus
	Actor            Actor
}

type DeleteUserRequest struct {
	NamespaceID      string
	UserID           string
	ExpectedRevision uint64
	Actor            Actor
}

type CreateTeamRequest struct {
	NamespaceID               string
	Name                      string
	Description               string
	AccessPolicyIDs           []string
	RateLimitPolicyID         string
	NamespaceDefaults         *TeamDefaults
	UseDefaultAccessPolicy    bool
	UseDefaultRateLimitPolicy bool
	IdempotencyKey            string
	Actor                     Actor
}

type UpdateTeamRequest struct {
	NamespaceID      string
	TeamID           string
	ExpectedRevision uint64
	Name             *string
	Description      *string
	Status           *accesscontrol.TeamStatus
	Actor            Actor
}

type DeleteTeamRequest struct {
	NamespaceID      string
	TeamID           string
	ExpectedRevision uint64
	Actor            Actor
}

type PutMembershipRequest struct {
	NamespaceID    string
	TeamID         string
	UserID         string
	Role           accesscontrol.TeamRole
	IdempotencyKey string
	Actor          Actor
}

type UpdateMembershipRequest struct {
	NamespaceID      string
	TeamID           string
	UserID           string
	ExpectedRevision uint64
	Role             *accesscontrol.TeamRole
	Status           *accesscontrol.MembershipStatus
	Actor            Actor
}

type DeleteMembershipRequest struct {
	NamespaceID      string
	TeamID           string
	UserID           string
	ExpectedRevision uint64
	Actor            Actor
}

// Repository mutations receive pre-bound commands so the command result,
// resource, audit event, and outbox record commit in one PostgreSQL transaction.
type CreateUserMutation struct {
	User    User
	Command managementcommand.Command
	Actor   Actor
}

type CreateTeamMutation struct {
	Team                      Team
	AccessPolicyBindings      []TeamAccessPolicyBinding
	RateLimitAllocation       TeamRateLimitAllocation
	NamespaceDefaults         *TeamDefaults
	UseDefaultAccessPolicy    bool
	UseDefaultRateLimitPolicy bool
	Command                   managementcommand.Command
	Actor                     Actor
}

type TeamAccessPolicyBinding struct {
	ID       string
	PolicyID string
}

type TeamRateLimitAllocation struct {
	ID       string
	PolicyID string
}

type PutMembershipMutation struct {
	Membership Membership
	Command    managementcommand.Command
	Actor      Actor
}
