package managementauth

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
)

type ExchangeChallenge struct {
	ID        string
	Nonce     string
	ExpiresAt time.Time
}

type ExchangeChallengeStore interface {
	Ready(context.Context) error
	Create(context.Context, string, string, time.Time) (ExchangeChallenge, error)
	Consume(context.Context, string, string, string, time.Time) error
}

type SubjectTokenType string

const (
	// #nosec G101 -- these are public token-format identifiers, not credential material.
	SubjectTokenOIDCIDToken     SubjectTokenType = "oidc_id_token"
	SubjectTokenRouterAssertion SubjectTokenType = "router_local_assertion"
)

type VerifiedExternalIdentity struct {
	IssuerID          string
	Issuer            string
	Subject           string
	VerifiedEmail     string
	DisplayName       string
	Nonce             string
	IssuerSessionID   *string
	AAL               string
	AMR               []string
	AuthenticatedAt   time.Time
	EvidenceExpiresAt time.Time
}

type SubjectAssertionVerifier interface {
	ValidateIssuer(context.Context, string) error
	Verify(context.Context, string, SubjectTokenType, string, time.Time) (VerifiedExternalIdentity, error)
}

type PrincipalResolver interface {
	ResolvePrincipal(context.Context, string, string) (string, error)
}

// InvitationOnboarding is the bounded, one-time onboarding result nested in a
// successful invited token exchange. APIKey is never persisted in plaintext.
type InvitationOnboarding struct {
	InvitationID      string
	PrincipalID       string
	UserID            string
	TeamID            string
	APIKeyID          string
	APIKey            string
	DeliveryExpiresAt time.Time
}

// IdentityExchangeRequest is verified issuer evidence plus a session draft.
// An empty InvitationToken selects an existing-principal exchange; a non-empty
// token selects atomic invitation acceptance. The coordinator implements both
// modes so authentication never falls back to an optional invitation hook.
type IdentityExchangeRequest struct {
	Identity        VerifiedExternalIdentity
	InvitationToken string
	Session         SessionDraft
	RequestID       string
}

type IdentityExchangeResult struct {
	Issued     IssuedToken
	Onboarding *InvitationOnboarding
	Replayed   bool
}

// PreparedSessionIssuer signs a session while its creating PostgreSQL
// transaction is still open. Returning an error aborts that transaction.
type PreparedSessionIssuer func(context.Context, LiveSession, time.Time) (IssuedToken, error)

// IdentityExchangeCoordinator is the single transactional seam for issuer
// exchanges. Implementations must create an existing-principal session or
// accept an invitation and create its session in one serializable transaction.
type IdentityExchangeCoordinator interface {
	Ready(context.Context) error
	ExchangeIdentity(context.Context, IdentityExchangeRequest, PreparedSessionIssuer) (IdentityExchangeResult, error)
}

type VerifiedServiceCredential struct {
	PrincipalID       string
	CredentialID      string
	WorkloadClass     string
	SourceAssuredAt   time.Time
	EvidenceExpiresAt time.Time
}

type ServiceCredentialVerifier interface {
	VerifyServiceCredential(context.Context, string, time.Time) (VerifiedServiceCredential, error)
}

type AuthServiceOptions struct {
	Challenges         ExchangeChallengeStore
	Assertions         SubjectAssertionVerifier
	Exchanges          IdentityExchangeCoordinator
	ServiceCredentials ServiceCredentialVerifier
	MTLSIdentities     MTLSIdentityResolver
	Sessions           SessionRepository
	Runtime            SessionRuntime
	Now                func() time.Time
	NewID              func() (string, error)
}

type AuthService struct {
	challenges  ExchangeChallengeStore
	assertions  SubjectAssertionVerifier
	exchanges   IdentityExchangeCoordinator
	credentials ServiceCredentialVerifier
	mtls        MTLSIdentityResolver
	sessions    SessionRepository
	runtime     SessionRuntime
	now         func() time.Time
	newID       func() (string, error)
}

func NewAuthService(options AuthServiceOptions) (*AuthService, error) {
	if options.Challenges == nil || options.Assertions == nil || options.Exchanges == nil ||
		options.ServiceCredentials == nil || options.MTLSIdentities == nil || options.Sessions == nil {
		return nil, errors.New("management authentication exchange dependencies are required")
	}
	if err := options.Runtime.validateConfiguration(); err != nil {
		return nil, err
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = defaultUUID
	}
	return &AuthService{
		challenges: options.Challenges, assertions: options.Assertions,
		exchanges: options.Exchanges, credentials: options.ServiceCredentials, mtls: options.MTLSIdentities,
		sessions: options.Sessions, runtime: options.Runtime, now: now, newID: newID,
	}, nil
}

func (service *AuthService) Ready(ctx context.Context) error {
	if service == nil || service.challenges == nil || service.exchanges == nil {
		return ErrAuthenticationUnavailable
	}
	if err := service.challenges.Ready(ctx); err != nil {
		return err
	}
	return service.exchanges.Ready(ctx)
}

func (service *AuthService) CreateChallenge(ctx context.Context, issuerID, rateIdentity string) (ExchangeChallenge, error) {
	if service == nil || issuerID == "" || rateIdentity == "" {
		return ExchangeChallenge{}, ErrAuthenticationDenied
	}
	if err := service.assertions.ValidateIssuer(ctx, issuerID); err != nil {
		return ExchangeChallenge{}, ErrAuthenticationDenied
	}
	return service.challenges.Create(ctx, issuerID, rateIdentity, service.now().UTC())
}

func (service *AuthService) Exchange(ctx context.Context, issuerID, challengeID string, tokenType SubjectTokenType,
	subjectToken, invitationToken string,
) (IdentityExchangeResult, error) {
	now := service.now().UTC()
	if tokenType != SubjectTokenOIDCIDToken && tokenType != SubjectTokenRouterAssertion {
		return IdentityExchangeResult{}, ErrAuthenticationDenied
	}
	verified, exchangeErr := service.assertions.Verify(ctx, issuerID, tokenType, subjectToken, now)
	if exchangeErr != nil || verified.IssuerID != issuerID || verified.Nonce == "" || verified.Issuer == "" || verified.Subject == "" ||
		verified.AuthenticatedAt.IsZero() || verified.EvidenceExpiresAt.IsZero() || !now.Before(verified.EvidenceExpiresAt) {
		return IdentityExchangeResult{}, ErrAuthenticationDenied
	}
	if err := service.challenges.Consume(ctx, challengeID, issuerID, verified.Nonce, now); err != nil {
		return IdentityExchangeResult{}, ErrAuthenticationDenied
	}
	draft, exchangeErr := service.sessionDraft(VerifiedSessionSource{
		IssuerSessionID: verified.IssuerSessionID,
		AuthSourceKind:  AuthSourceIssuer, AuthSourceID: issuerID, EvidenceKind: EvidenceHuman,
		Human:           &HumanEvidence{AuthenticationTime: verified.AuthenticatedAt.Unix(), AAL: verified.AAL, AMR: verified.AMR},
		AuthenticatedAt: verified.AuthenticatedAt.UTC(), EvidenceExpiresAt: verified.EvidenceExpiresAt.UTC(),
	})
	if exchangeErr != nil {
		return IdentityExchangeResult{}, ErrAuthenticationUnavailable
	}
	result, exchangeErr := service.exchanges.ExchangeIdentity(ctx, IdentityExchangeRequest{
		Identity: verified, InvitationToken: invitationToken, Session: draft, RequestID: challengeID,
	}, service.runtime.issuePrepared)
	if exchangeErr != nil {
		return IdentityExchangeResult{}, exchangeErr
	}
	return result, nil
}

func (service *AuthService) ServiceToken(ctx context.Context, credential string) (IssuedToken, error) {
	now := service.now().UTC()
	verified, err := service.credentials.VerifyServiceCredential(ctx, credential, now)
	if err != nil || verified.PrincipalID == "" || verified.CredentialID == "" ||
		verified.SourceAssuredAt.IsZero() || verified.EvidenceExpiresAt.IsZero() || !now.Before(verified.EvidenceExpiresAt) {
		return IssuedToken{}, ErrAuthenticationDenied
	}
	return service.createSession(ctx, VerifiedSessionSource{
		PrincipalID: verified.PrincipalID, AuthSourceKind: AuthSourceServiceCredential,
		AuthSourceID: verified.CredentialID, EvidenceKind: EvidenceWorkload,
		Workload:        &WorkloadEvidence{Class: verified.WorkloadClass, SourceAssuredAt: verified.SourceAssuredAt.Unix()},
		AuthenticatedAt: now, EvidenceExpiresAt: verified.EvidenceExpiresAt.UTC(),
	})
}

func (service *AuthService) MTLSToken(ctx context.Context, evidence VerifiedMTLSEvidence) (IssuedToken, error) {
	now := service.now().UTC()
	verified, err := service.mtls.ResolveMTLSIdentity(ctx, evidence, now)
	if err != nil || verified.PrincipalID == "" || verified.MappingID == "" ||
		verified.SourceAssuredAt.IsZero() || verified.EvidenceExpiresAt.IsZero() || !now.Before(verified.EvidenceExpiresAt) {
		return IssuedToken{}, ErrAuthenticationDenied
	}
	return service.createSession(ctx, VerifiedSessionSource{
		PrincipalID: verified.PrincipalID, AuthSourceKind: AuthSourceMTLS,
		AuthSourceID: verified.MappingID, EvidenceKind: EvidenceWorkload,
		Workload:        &WorkloadEvidence{Class: verified.WorkloadClass, SourceAssuredAt: verified.SourceAssuredAt.Unix()},
		AuthenticatedAt: now, EvidenceExpiresAt: verified.EvidenceExpiresAt.UTC(),
	})
}

func (service *AuthService) createSession(ctx context.Context, source VerifiedSessionSource) (IssuedToken, error) {
	draft, err := service.sessionDraft(source)
	if err != nil {
		return IssuedToken{}, ErrAuthenticationUnavailable
	}
	committed, err := service.sessions.Create(ctx, draft)
	if err != nil {
		if errors.Is(err, ErrSessionLimitExceeded) || errors.Is(err, ErrSessionInactive) || errors.Is(err, ErrSessionNotFound) {
			return IssuedToken{}, ErrAuthenticationDenied
		}
		return IssuedToken{}, ErrAuthenticationUnavailable
	}
	issueAt := service.now().UTC()
	if issueAt.Before(committed.CreatedAt) {
		issueAt = committed.CreatedAt
	}
	return service.runtime.Issue(ctx, committed.ID, issueAt)
}

func (service *AuthService) sessionDraft(source VerifiedSessionSource) (SessionDraft, error) {
	sessionID, err := service.newID()
	if err != nil {
		return SessionDraft{}, err
	}
	tokenID, err := service.newID()
	if err != nil {
		return SessionDraft{}, err
	}
	return SessionDraft{
		ID: sessionID, PrincipalID: source.PrincipalID, IssuerSessionID: source.IssuerSessionID,
		TokenID: tokenID, Audience: service.runtime.Codec.Audience, AuthSourceKind: source.AuthSourceKind,
		AuthSourceID: source.AuthSourceID, EvidenceKind: source.EvidenceKind,
		Human: source.Human, Workload: source.Workload, AuthenticatedAt: source.AuthenticatedAt,
		EvidenceExpiresAt: source.EvidenceExpiresAt,
	}, nil
}

func defaultUUID() (string, error) { return uuid.NewString(), nil }
