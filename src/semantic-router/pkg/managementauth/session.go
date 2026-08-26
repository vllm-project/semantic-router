package managementauth

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
)

var (
	ErrSessionNotFound           = errors.New("management session not found")
	ErrSessionConflict           = errors.New("management session changed concurrently")
	ErrSessionInactive           = errors.New("management session is inactive")
	ErrSessionLimitExceeded      = errors.New("management session limit exceeded")
	ErrAuthenticationDenied      = errors.New("management authentication denied")
	ErrAuthenticationUnavailable = errors.New("management authentication state is unavailable")
	ErrChallengeCapacityExceeded = errors.New("management exchange challenge capacity exceeded")
	ErrInvitationExpired         = errors.New("management invitation expired")
	ErrInvitationResultExpired   = errors.New("management invitation result expired")
	ErrInvitationConflict        = errors.New("management invitation conflicts with current identity state")
)

// ChallengeCapacityError reports a temporary exhaustion of outstanding
// exchange challenges. RetryAfter is bounded by the store's challenge TTL and
// is safe to translate into an HTTP Retry-After header.
type ChallengeCapacityError struct {
	RetryAfter time.Duration
}

func (err *ChallengeCapacityError) Error() string {
	return ErrChallengeCapacityExceeded.Error()
}

func (err *ChallengeCapacityError) Unwrap() error {
	return ErrChallengeCapacityExceeded
}

type SessionStatus string

const (
	SessionActive  SessionStatus = "active"
	SessionRevoked SessionStatus = "revoked"
	SessionExpired SessionStatus = "expired"
)

type ResourceStatus string

const (
	ResourceActive   ResourceStatus = "active"
	ResourceDisabled ResourceStatus = "disabled"
)

type AuthSourceKind string

const (
	AuthSourceIssuer            AuthSourceKind = "issuer"
	AuthSourceServiceCredential AuthSourceKind = "service_credential"
	AuthSourceMTLS              AuthSourceKind = "mtls"
)

// Session is the global durable Management-session fact. Namespace authority
// is resolved per target request and is never persisted on this row or copied
// into the token. It deliberately contains no role or permission snapshot.
type Session struct {
	ID              string
	PrincipalID     string
	IssuerSessionID *string
	TokenID         string
	Audience        string
	AuthSourceKind  AuthSourceKind
	AuthSourceID    string
	EvidenceKind    EvidenceKind
	Human           *HumanEvidence
	Workload        *WorkloadEvidence
	AuthenticatedAt time.Time
	ExpiresAt       time.Time
	Status          SessionStatus
	RevokedAt       *time.Time
	CreatedAt       time.Time
}

// SessionDraft is verified bootstrap evidence ready to be committed as one
// durable session. EvidenceExpiresAt is the upper bound from the verified
// assertion, service credential, or client-certificate mapping; the store also
// applies the cluster session TTL.
type SessionDraft struct {
	ID                string
	PrincipalID       string
	IssuerSessionID   *string
	TokenID           string
	Audience          string
	AuthSourceKind    AuthSourceKind
	AuthSourceID      string
	EvidenceKind      EvidenceKind
	Human             *HumanEvidence
	Workload          *WorkloadEvidence
	AuthenticatedAt   time.Time
	EvidenceExpiresAt time.Time
}

// LiveSession joins the durable session with the current principal and
// authentication-source lifecycle. A missing source is represented by an empty
// status and therefore fails closed.
type LiveSession struct {
	Session
	PrincipalStatus     ResourceStatus
	AuthSourceStatus    ResourceStatus
	AuthSourceNotBefore *time.Time
	AuthSourceExpiresAt *time.Time
	AuthSourceAssuredAt *time.Time
}

func (s Session) Validate() error {
	for field, value := range map[string]string{
		"session id":               s.ID,
		"principal id":             s.PrincipalID,
		"authentication source id": s.AuthSourceID,
	} {
		if _, err := uuid.Parse(value); err != nil {
			return fmt.Errorf("%s must be a UUID: %w", field, err)
		}
	}
	if !canonicalText(s.TokenID, 1, 256) {
		return errors.New("management session token id is required and must be canonical")
	}
	if !canonicalText(s.Audience, 1, 256) {
		return errors.New("management session audience is required and must be canonical")
	}
	if s.IssuerSessionID != nil && !canonicalText(*s.IssuerSessionID, 1, 512) {
		return errors.New("issuer session id must be canonical when present")
	}
	if !validAuthSourceKind(s.AuthSourceKind) {
		return errors.New("management session authentication source kind is invalid")
	}
	if s.AuthenticatedAt.IsZero() || s.ExpiresAt.IsZero() || s.CreatedAt.IsZero() || !s.ExpiresAt.After(s.CreatedAt) {
		return errors.New("management session times are invalid")
	}
	if !validSessionStatus(s.Status) {
		return errors.New("management session status is invalid")
	}
	if s.Status == SessionRevoked {
		if s.RevokedAt == nil || s.RevokedAt.IsZero() {
			return errors.New("revoked management session requires revoked_at")
		}
	} else if s.RevokedAt != nil {
		return errors.New("non-revoked management session cannot have revoked_at")
	}

	switch s.EvidenceKind {
	case EvidenceHuman:
		if s.Human == nil || s.Workload != nil || s.Human.AuthenticationTime <= 0 ||
			!time.Unix(s.Human.AuthenticationTime, 0).Equal(s.AuthenticatedAt.Truncate(time.Second)) ||
			!validAAL(s.Human.AAL) || !validAMR(s.Human.AMR) {
			return errors.New("management session human evidence is invalid")
		}
		if s.AuthSourceKind == AuthSourceServiceCredential || s.AuthSourceKind == AuthSourceMTLS {
			return errors.New("service credentials and mTLS require workload evidence")
		}
	case EvidenceWorkload:
		if s.Workload == nil || s.Human != nil || !validWorkloadClass(s.Workload.Class) ||
			s.Workload.SourceAssuredAt <= 0 {
			return errors.New("management session workload evidence is invalid")
		}
		if s.AuthSourceKind == AuthSourceIssuer {
			return errors.New("issuer sessions require human evidence")
		}
	default:
		return errors.New("management session evidence kind is invalid")
	}
	return nil
}

func (s LiveSession) ValidateAt(now time.Time) error {
	if err := s.Validate(); err != nil {
		return err
	}
	if s.Status != SessionActive || now.Before(s.CreatedAt) || !now.Before(s.ExpiresAt) ||
		s.PrincipalStatus != ResourceActive || s.AuthSourceStatus != ResourceActive {
		return ErrSessionInactive
	}
	if s.AuthSourceNotBefore != nil && now.Before(*s.AuthSourceNotBefore) {
		return ErrSessionInactive
	}
	if s.AuthSourceExpiresAt != nil && !now.Before(*s.AuthSourceExpiresAt) {
		return ErrSessionInactive
	}
	if s.AuthSourceKind == AuthSourceServiceCredential &&
		(s.AuthSourceNotBefore == nil || s.AuthSourceExpiresAt == nil || s.AuthSourceAssuredAt == nil) {
		return ErrSessionInactive
	}
	if s.AuthSourceKind == AuthSourceMTLS && s.AuthSourceAssuredAt == nil {
		return ErrSessionInactive
	}
	if s.EvidenceKind == EvidenceWorkload {
		if s.AuthSourceAssuredAt == nil ||
			s.Workload.SourceAssuredAt != s.AuthSourceAssuredAt.Unix() {
			return ErrSessionInactive
		}
	}
	return nil
}

type SessionMutation struct {
	SessionID string
	TokenID   string
	Changed   bool
	ChangedAt time.Time
}

// SessionRepository is the authoritative durable session seam. RotateTokenID
// atomically replaces only the short-lived access-token JTI; it never extends
// the durable session lifetime or changes authentication evidence.
type SessionRepository interface {
	Create(context.Context, SessionDraft) (LiveSession, error)
	Get(context.Context, string) (LiveSession, error)
	RotateTokenID(context.Context, string, string, string) (LiveSession, error)
	Revoke(context.Context, string, string) (SessionMutation, error)
}

type BarrierKind string

const (
	BarrierClusterSessionPolicy    BarrierKind = "cluster_session_policy"
	BarrierNamespaceSecurityPolicy BarrierKind = "namespace_security_policy"
	BarrierManagementSession       BarrierKind = "management_session"
	BarrierManagementPrincipal     BarrierKind = "management_principal"
	BarrierAuthenticationSource    BarrierKind = "authentication_source"
)

type BarrierCheck struct {
	SessionID      string
	PrincipalID    string
	AuthSourceKind AuthSourceKind
	AuthSourceID   string
	NamespaceID    string
}

type BarrierState struct {
	Ready            bool
	ClusterDenied    bool
	NamespaceDenied  bool
	SessionDenied    bool
	PrincipalDenied  bool
	AuthSourceDenied bool
}

func (s BarrierState) Allows() bool {
	return s.Ready && !s.ClusterDenied && !s.NamespaceDenied && !s.SessionDenied &&
		!s.PrincipalDenied && !s.AuthSourceDenied
}

// RevocationBarrierStore is the applied, strongly acknowledged runtime seam.
// InstallDeny must not return success before the selected durability profile has
// acknowledged the barrier. Check must never infer "allow" from a cache miss.
type RevocationBarrierStore interface {
	Check(context.Context, BarrierCheck) (BarrierState, error)
	InstallDeny(context.Context, BarrierKind, string) error
}

// DelegationBarrierCheck is the minimum Management lifecycle identity retained
// by a delegated inference credential. Authentication-source facts remain
// private to the Management session runtime; child delegations only need the
// shared session and principal barriers.
type DelegationBarrierCheck struct {
	SessionID   string
	PrincipalID string
}

type DelegationBarrierState struct {
	Ready           bool
	SessionDenied   bool
	PrincipalDenied bool
}

func (s DelegationBarrierState) Allows() bool {
	return s.Ready && !s.SessionDenied && !s.PrincipalDenied
}

// DelegationRevocationBarrierStore exposes the shared Management lifecycle
// barrier generation to the inference runtime without granting it mutation
// authority or requiring it to load a durable Management session.
type DelegationRevocationBarrierStore interface {
	CheckDelegation(context.Context, DelegationBarrierCheck) (DelegationBarrierState, error)
}

// RevocationBarrier is one durable deny fact reconstructed into the shared
// runtime store. Authentication-source IDs include their typed source kind so
// equal UUIDs from distinct tables cannot collide.
type RevocationBarrier struct {
	Kind BarrierKind
	ID   string
}

type RevocationSnapshotLoader interface {
	LoadRevocationBarriers(context.Context) ([]RevocationBarrier, error)
}

func validAuthSourceKind(kind AuthSourceKind) bool {
	switch kind {
	case AuthSourceIssuer, AuthSourceServiceCredential, AuthSourceMTLS:
		return true
	default:
		return false
	}
}

func validSessionStatus(status SessionStatus) bool {
	return status == SessionActive || status == SessionRevoked || status == SessionExpired
}

func validAAL(aal string) bool {
	return aal == "aal1" || aal == "aal2" || aal == "aal3"
}

func validWorkloadClass(class string) bool {
	return class == "workload_standard" || class == "workload_strong"
}

func validAMR(methods []string) bool {
	if len(methods) == 0 || len(methods) > 16 {
		return false
	}
	copyOfMethods := slices.Clone(methods)
	slices.Sort(copyOfMethods)
	for index, method := range copyOfMethods {
		if !canonicalText(method, 1, 64) || (index > 0 && method == copyOfMethods[index-1]) {
			return false
		}
	}
	return true
}

func canonicalText(value string, minimum, maximum int) bool {
	return len(value) >= minimum && len(value) <= maximum && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n\t")
}
