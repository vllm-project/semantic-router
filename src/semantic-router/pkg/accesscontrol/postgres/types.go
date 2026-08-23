package postgres

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type MutationMeta struct {
	ActorPrincipalID *accesscontrol.ManagementPrincipalID
	ActorChain       []accesscontrol.ManagementPrincipalID
	RequestID        string
	SourceIP         netip.Addr
	Action           string
	Reason           string
	Details          AuditDetails
}

type AuditDetails map[string]string

type MutationReceipt struct {
	DesiredRevision accesscontrol.Revision
}

type MutationResult[T any] struct {
	Value            T
	Receipt          MutationReceipt
	ResourceID       string
	ResourceRevision accesscontrol.Revision
	Replayed         bool
	ResponseStatus   int
}

type UserRecord struct {
	User      accesscontrol.User
	Revision  accesscontrol.Revision
	DeletedAt *time.Time
}

type TeamRecord struct {
	Team        accesscontrol.Team
	Description string
	Revision    accesscontrol.Revision
	DeletedAt   *time.Time
}

type MembershipRecord struct {
	Membership accesscontrol.TeamMembership
	Revision   accesscontrol.Revision
}

type CredentialRecord struct {
	NamespaceID accesscontrol.NamespaceID
	Credential  accesscontrol.CredentialVersion
}

type CredentialRotation struct {
	Credential         accesscontrol.CredentialVersion
	RetireCredentialID *accesscontrol.CredentialVersionID
	RetireAt           *time.Time
}
