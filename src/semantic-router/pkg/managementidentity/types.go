// Package managementidentity owns durable Management principals, roles,
// bindings, user links, session policy metadata, and authentication bootstrap.
// It is independent from any Dashboard account model.
package managementidentity

import (
	"errors"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

var (
	ErrNotFound                = errors.New("management identity resource not found")
	ErrAlreadyExists           = errors.New("management identity resource already exists")
	ErrRevisionConflict        = errors.New("management identity revision conflict")
	ErrBuiltInImmutable        = errors.New("built-in Management role is immutable")
	ErrRoleInUse               = errors.New("management role has active bindings")
	ErrPrincipalLinkInUse      = errors.New("management principal User link has active User-scoped role bindings")
	ErrDelegationDenied        = errors.New("management role delegation is not authorized")
	ErrBootstrapConsumed       = errors.New("management bootstrap is already consumed")
	ErrBootstrapUnavailable    = errors.New("management bootstrap is unavailable")
	ErrBootstrapResultExpired  = errors.New("management bootstrap result has expired")
	ErrBootstrapConflict       = errors.New("management bootstrap idempotency conflict")
	ErrInvalidBootstrapRequest = errors.New("management bootstrap request is invalid")
	ErrRecoveryConsumed        = errors.New("management recovery credential is already consumed")
	ErrRecoveryUnavailable     = errors.New("management recovery is unavailable")
	ErrRecoveryConflict        = errors.New("management recovery idempotency conflict")
	ErrInvalidRecoveryRequest  = errors.New("management recovery request is invalid")
	ErrInvalidLifecycleRequest = errors.New("management identity lifecycle request is invalid")
	ErrBackchannelReplay       = errors.New("management back-channel logout token was reused with different claims")
)

type BootstrapKind string

const (
	BootstrapExternalPrincipal BootstrapKind = "external_principal"
	BootstrapServiceAccount    BootstrapKind = "service_account"
)

type BootstrapRequest struct {
	Kind             BootstrapKind
	DisplayName      string
	IssuerID         string
	Issuer           string
	Subject          string
	DiscoveryURL     string
	Audience         string
	IdempotencyKey   string
	CanonicalRequest []byte
}

type BootstrapResult struct {
	PrincipalID                string
	RoleBindingID              string
	ServiceAccountID           string
	ServiceCredentialID        string
	ServiceCredential          string
	ServiceCredentialExpiresAt time.Time
	FinalizationRequired       bool
	Replayed                   bool
	ResponseStatus             int
}

// RecoveryRequest restores cluster-administrator authority to an existing
// durable Management principal. Recovery deliberately cannot create a new
// principal or issue a credential.
type RecoveryRequest struct {
	PrincipalID      string
	Reason           string
	RequestID        string
	IdempotencyKey   string
	CanonicalRequest []byte
}

type RecoveryResult struct {
	PrincipalID    string
	RoleBindingID  string
	Replayed       bool
	ResponseStatus int
}

type Principal struct {
	Identity      accesscontrol.ManagementPrincipal
	DisplayName   string
	VerifiedEmail string
	Revision      accesscontrol.Revision
}

type Role struct {
	Role        accesscontrol.ManagementRole
	Description string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

type RoleBinding struct {
	Binding   accesscontrol.ManagementRoleBinding
	CreatedAt time.Time
	UpdatedAt time.Time
}

type PrincipalUserLink struct {
	PrincipalID accesscontrol.ManagementPrincipalID
	NamespaceID accesscontrol.NamespaceID
	UserID      accesscontrol.UserID
	Revision    accesscontrol.Revision
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

type MutationActor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
	Reason      string
}

type MutationResult struct {
	Kind           string
	ID             string
	Revision       uint64
	ResponseStatus int
	Replayed       bool
}

type ListRequest struct {
	AfterID string
	Limit   int
}

type PrincipalPage struct {
	Items      []Principal
	NextCursor string
}

type RolePage struct {
	Items      []Role
	NextCursor string
}

type RoleBindingPage struct {
	Items      []RoleBinding
	NextCursor string
}

type CreatePrincipal struct {
	ID            string
	Issuer        string
	Subject       string
	DisplayName   string
	VerifiedEmail string
	Attributes    map[string]string
	Command       managementcommand.Command
	Actor         MutationActor
}

type UpdatePrincipal struct {
	ID               string
	ExpectedRevision uint64
	DisplayName      *string
	VerifiedEmail    *string
	Status           *accesscontrol.PrincipalStatus
	Actor            MutationActor
}

type CreateRole struct {
	ID          string
	NamespaceID string
	Name        string
	DisplayName string
	Description string
	Permissions accesscontrol.PermissionSet
	Command     managementcommand.Command
	Actor       MutationActor
}

type UpdateRole struct {
	ID               string
	ExpectedRevision uint64
	DisplayName      *string
	Description      *string
	Actor            MutationActor
}

type CreateRoleBinding struct {
	ID                string
	PrincipalID       string
	RoleID            string
	Scope             accesscontrol.Scope
	DelegationCeiling accesscontrol.PermissionSet
	Command           managementcommand.Command
	Actor             MutationActor
}

type UpdateRoleBinding struct {
	ID               string
	ExpectedRevision uint64
	Status           accesscontrol.BindingStatus
	Actor            MutationActor
}

type LinkMutation struct {
	PrincipalID      string
	NamespaceID      string
	UserID           string
	ExpectedRevision *uint64
	Command          managementcommand.Command
	Actor            MutationActor
}
