package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type UserReader interface {
	GetUser(context.Context, accesscontrol.NamespaceID, accesscontrol.UserID) (UserRecord, error)
}

type UserWriter interface {
	CreateUser(context.Context, accesscontrol.User, MutationMeta) (MutationResult[UserRecord], error)
	UpdateUser(context.Context, accesscontrol.User, accesscontrol.Revision, MutationMeta) (MutationResult[UserRecord], error)
	SoftDeleteUser(context.Context, accesscontrol.NamespaceID, accesscontrol.UserID, accesscontrol.Revision, MutationMeta) (MutationResult[UserRecord], error)
}

type UserRepository interface {
	UserReader
	UserWriter
}

type TeamReader interface {
	GetTeam(context.Context, accesscontrol.NamespaceID, accesscontrol.TeamID) (TeamRecord, error)
}

type TeamWriter interface {
	CreateTeam(context.Context, TeamRecord, MutationMeta) (MutationResult[TeamRecord], error)
	UpdateTeam(context.Context, TeamRecord, accesscontrol.Revision, MutationMeta) (MutationResult[TeamRecord], error)
	SoftDeleteTeam(context.Context, accesscontrol.NamespaceID, accesscontrol.TeamID, accesscontrol.Revision, MutationMeta) (MutationResult[TeamRecord], error)
}

type TeamRepository interface {
	TeamReader
	TeamWriter
}

type MembershipReader interface {
	GetMembership(context.Context, accesscontrol.NamespaceID, accesscontrol.TeamID, accesscontrol.UserID) (MembershipRecord, error)
}

type MembershipWriter interface {
	CreateMembership(context.Context, accesscontrol.TeamMembership, MutationMeta) (MutationResult[MembershipRecord], error)
	UpdateMembership(context.Context, accesscontrol.TeamMembership, accesscontrol.Revision, MutationMeta) (MutationResult[MembershipRecord], error)
}

type MembershipRepository interface {
	MembershipReader
	MembershipWriter
}
