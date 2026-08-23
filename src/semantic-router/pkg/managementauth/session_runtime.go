package managementauth

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/google/uuid"
)

type AuthenticatedSession struct {
	Claims      Claims
	Session     LiveSession
	NamespaceID string
}

type IssuedToken struct {
	AccessToken         string
	TokenType           string
	ExpiresIn           time.Duration
	ManagementSessionID string
}

type TokenIDGenerator func() (string, error)

// VerifiedSessionSource is reconstructed only after a fresh issuer assertion,
// service credential, or mTLS identity has been verified. It is not populated
// from the current Management bearer token.
type VerifiedSessionSource struct {
	PrincipalID       string
	IssuerSessionID   *string
	AuthSourceKind    AuthSourceKind
	AuthSourceID      string
	EvidenceKind      EvidenceKind
	Human             *HumanEvidence
	Workload          *WorkloadEvidence
	AuthenticatedAt   time.Time
	EvidenceExpiresAt time.Time
}

// SessionRuntime verifies Router-signed JWTs against live session state and
// applied deny barriers. It intentionally performs no positive local caching.
type SessionRuntime struct {
	Codec        TokenCodec
	Sessions     SessionRepository
	Barriers     RevocationBarrierStore
	PolicyLoader SessionPolicyLoader
	NewTokenID   TokenIDGenerator
}

func (r SessionRuntime) Authenticate(
	ctx context.Context,
	token string,
	namespaceID string,
	now time.Time,
) (AuthenticatedSession, error) {
	if err := r.validateConfiguration(); err != nil {
		return AuthenticatedSession{}, err
	}
	accessTokenTTL, authenticateErr := r.accessTokenTTL(ctx)
	if authenticateErr != nil {
		return AuthenticatedSession{}, authenticateErr
	}
	if namespaceID != "" {
		if _, err := uuid.Parse(namespaceID); err != nil {
			return AuthenticatedSession{}, ErrAuthenticationDenied
		}
	}
	claims, authenticateErr := r.Codec.Verify(token, now)
	if authenticateErr != nil {
		return AuthenticatedSession{}, ErrAuthenticationDenied
	}
	session, authenticateErr := r.loadLive(ctx, claims.SessionID, now)
	if authenticateErr != nil {
		return AuthenticatedSession{}, authenticateErr
	}
	if err := validateClaimsAgainstSession(claims, session, accessTokenTTL); err != nil {
		return AuthenticatedSession{}, ErrAuthenticationDenied
	}
	if err := r.checkBarriers(ctx, session, namespaceID); err != nil {
		return AuthenticatedSession{}, err
	}
	return AuthenticatedSession{Claims: claims, Session: session, NamespaceID: namespaceID}, nil
}

// Issue issues the first short-lived access token for an already committed
// session. Session creation and identity-source verification happen before this
// seam; this method rechecks current source state and barriers before signing.
func (r SessionRuntime) Issue(
	ctx context.Context,
	sessionID string,
	now time.Time,
) (IssuedToken, error) {
	if err := r.validateConfiguration(); err != nil {
		return IssuedToken{}, err
	}
	accessTokenTTL, err := r.accessTokenTTL(ctx)
	if err != nil {
		return IssuedToken{}, err
	}
	session, err := r.loadLive(ctx, sessionID, now)
	if err != nil {
		return IssuedToken{}, err
	}
	if err := r.checkBarriers(ctx, session, ""); err != nil {
		return IssuedToken{}, err
	}
	return r.issueFor(session, now, accessTokenTTL)
}

// issuePrepared signs a LiveSession assembled inside the caller's open
// PostgreSQL transaction. It deliberately does not reload the uncommitted row;
// every other lifecycle and bearer path must use Issue or Authenticate.
func (r SessionRuntime) issuePrepared(
	ctx context.Context,
	session LiveSession,
	now time.Time,
) (IssuedToken, error) {
	if err := r.validateConfiguration(); err != nil {
		return IssuedToken{}, err
	}
	accessTokenTTL, err := r.accessTokenTTL(ctx)
	if err != nil {
		return IssuedToken{}, err
	}
	if err := session.ValidateAt(now); err != nil || session.Audience != r.Codec.Audience {
		return IssuedToken{}, ErrAuthenticationDenied
	}
	if err := r.checkBarriers(ctx, session, ""); err != nil {
		return IssuedToken{}, err
	}
	return r.issueFor(session, now, accessTokenTTL)
}

// ReissueVerified rotates the access-token JTI only after the caller has
// freshly verified the original authentication source. An old bearer token is
// neither accepted nor sufficient. Reissue never extends the durable session
// or authentication-evidence lifetime.
func (r SessionRuntime) ReissueVerified(
	ctx context.Context,
	sessionID string,
	verified VerifiedSessionSource,
	now time.Time,
) (IssuedToken, error) {
	if err := r.validateConfiguration(); err != nil {
		return IssuedToken{}, err
	}
	accessTokenTTL, reissueVerifiedErr := r.accessTokenTTL(ctx)
	if reissueVerifiedErr != nil {
		return IssuedToken{}, reissueVerifiedErr
	}
	session, reissueVerifiedErr := r.loadLive(ctx, sessionID, now)
	if reissueVerifiedErr != nil {
		return IssuedToken{}, reissueVerifiedErr
	}
	if !verified.matches(session, now) {
		return IssuedToken{}, ErrAuthenticationDenied
	}
	if err := r.checkBarriers(ctx, session, ""); err != nil {
		return IssuedToken{}, err
	}
	newTokenID, reissueVerifiedErr := r.tokenIDGenerator()()
	if reissueVerifiedErr != nil || !canonicalText(newTokenID, 1, 256) || newTokenID == session.TokenID {
		return IssuedToken{}, fmt.Errorf("generate management token id: %w", ErrAuthenticationUnavailable)
	}
	prospective := session
	prospective.TokenID = newTokenID
	issued, reissueVerifiedErr := r.issueFor(prospective, now, accessTokenTTL)
	if reissueVerifiedErr != nil {
		return IssuedToken{}, reissueVerifiedErr
	}
	refreshed, reissueVerifiedErr := r.Sessions.RotateTokenID(
		ctx,
		session.ID,
		session.TokenID,
		newTokenID,
	)
	if reissueVerifiedErr != nil {
		if errors.Is(reissueVerifiedErr, ErrSessionConflict) || errors.Is(reissueVerifiedErr, ErrSessionInactive) || errors.Is(reissueVerifiedErr, ErrSessionNotFound) {
			return IssuedToken{}, ErrAuthenticationDenied
		}
		return IssuedToken{}, fmt.Errorf("refresh management session: %w", ErrAuthenticationUnavailable)
	}
	if err := refreshed.ValidateAt(now); err != nil || refreshed.TokenID != newTokenID {
		return IssuedToken{}, fmt.Errorf("validate refreshed management session: %w", ErrAuthenticationUnavailable)
	}
	if err := r.checkBarriers(ctx, refreshed, ""); err != nil {
		return IssuedToken{}, err
	}
	return issued, nil
}

func (v VerifiedSessionSource) matches(session LiveSession, now time.Time) bool {
	if now.IsZero() || v.EvidenceExpiresAt.IsZero() || now.Before(v.AuthenticatedAt) || !now.Before(v.EvidenceExpiresAt) {
		return false
	}
	if v.PrincipalID != session.PrincipalID || v.AuthSourceKind != session.AuthSourceKind ||
		v.AuthSourceID != session.AuthSourceID || v.EvidenceKind != session.EvidenceKind ||
		!equalOptionalString(v.IssuerSessionID, session.IssuerSessionID) ||
		!v.AuthenticatedAt.Equal(session.AuthenticatedAt) || v.EvidenceExpiresAt.Before(session.ExpiresAt) {
		return false
	}
	if session.Human != nil {
		return v.Human != nil && v.Workload == nil &&
			v.Human.AuthenticationTime == session.Human.AuthenticationTime &&
			v.Human.AAL == session.Human.AAL && slices.Equal(v.Human.AMR, session.Human.AMR)
	}
	return session.Workload != nil && v.Workload != nil && v.Human == nil &&
		v.Workload.Class == session.Workload.Class &&
		v.Workload.SourceAssuredAt == session.Workload.SourceAssuredAt
}

func equalOptionalString(left, right *string) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

// Revoke records the durable revocation and then installs the global session
// deny barrier. A barrier failure is returned even though durable revocation has
// committed; retries are idempotent and must finish the barrier before success.
func (r SessionRuntime) Revoke(
	ctx context.Context,
	sessionID string,
	expectedTokenID string,
) (SessionMutation, error) {
	if err := r.validateConfiguration(); err != nil {
		return SessionMutation{}, err
	}
	if _, err := uuid.Parse(sessionID); err != nil || !canonicalText(expectedTokenID, 1, 256) {
		return SessionMutation{}, ErrSessionNotFound
	}
	mutation, err := r.Sessions.Revoke(ctx, sessionID, expectedTokenID)
	if err != nil {
		return SessionMutation{}, err
	}
	if err := r.Barriers.InstallDeny(ctx, BarrierManagementSession, sessionID); err != nil {
		return mutation, fmt.Errorf("install management session deny barrier: %w", ErrAuthenticationUnavailable)
	}
	return mutation, nil
}

func (r SessionRuntime) issueFor(session LiveSession, now time.Time, accessTokenTTL time.Duration) (IssuedToken, error) {
	expiresAt := now.Add(accessTokenTTL)
	if session.ExpiresAt.Before(expiresAt) {
		expiresAt = session.ExpiresAt
	}
	if !expiresAt.After(now) {
		return IssuedToken{}, ErrAuthenticationDenied
	}
	claims := Claims{
		Issuer:         r.Codec.Issuer,
		Subject:        session.PrincipalID,
		SessionID:      session.ID,
		TokenID:        session.TokenID,
		Audience:       session.Audience,
		IssuedAt:       now.Unix(),
		ExpiresAt:      expiresAt.Unix(),
		AuthSourceKind: string(session.AuthSourceKind),
		AuthSourceID:   session.AuthSourceID,
		EvidenceKind:   session.EvidenceKind,
	}
	if session.Human != nil {
		claims.Human = &HumanEvidence{
			AuthenticationTime: session.Human.AuthenticationTime,
			AAL:                session.Human.AAL,
			AMR:                slices.Clone(session.Human.AMR),
		}
	}
	if session.Workload != nil {
		claims.Workload = &WorkloadEvidence{
			Class:           session.Workload.Class,
			SourceAssuredAt: session.Workload.SourceAssuredAt,
		}
	}
	token, err := r.Codec.Issue(claims)
	if err != nil {
		return IssuedToken{}, fmt.Errorf("issue management token: %w", err)
	}
	return IssuedToken{
		AccessToken:         token,
		TokenType:           "Bearer",
		ExpiresIn:           expiresAt.Sub(now),
		ManagementSessionID: session.ID,
	}, nil
}

func (r SessionRuntime) loadLive(ctx context.Context, sessionID string, now time.Time) (LiveSession, error) {
	session, err := r.Sessions.Get(ctx, sessionID)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) || errors.Is(err, ErrSessionInactive) {
			return LiveSession{}, ErrAuthenticationDenied
		}
		return LiveSession{}, fmt.Errorf("load management session: %w", ErrAuthenticationUnavailable)
	}
	if err := session.ValidateAt(now); err != nil {
		if errors.Is(err, ErrSessionInactive) {
			return LiveSession{}, ErrAuthenticationDenied
		}
		return LiveSession{}, fmt.Errorf("validate management session: %w", ErrAuthenticationUnavailable)
	}
	if session.Audience != r.Codec.Audience {
		return LiveSession{}, ErrAuthenticationDenied
	}
	return session, nil
}

func (r SessionRuntime) checkBarriers(ctx context.Context, session LiveSession, namespaceID string) error {
	state, err := r.Barriers.Check(ctx, BarrierCheck{
		SessionID:      session.ID,
		PrincipalID:    session.PrincipalID,
		AuthSourceKind: session.AuthSourceKind,
		AuthSourceID:   session.AuthSourceID,
		NamespaceID:    namespaceID,
	})
	if err != nil || !state.Ready {
		return fmt.Errorf("check management revocation barriers: %w", ErrAuthenticationUnavailable)
	}
	if !state.Allows() {
		return ErrAuthenticationDenied
	}
	return nil
}

func (r SessionRuntime) validateConfiguration() error {
	if r.Sessions == nil || r.Barriers == nil || r.PolicyLoader == nil {
		return errors.New("management session runtime requires durable sessions, policy, and revocation barriers")
	}
	if err := r.Codec.validateConfiguration(false); err != nil {
		return fmt.Errorf("management session token codec: %w", err)
	}
	return nil
}

func (r SessionRuntime) accessTokenTTL(ctx context.Context) (time.Duration, error) {
	policy, err := r.PolicyLoader.LoadSessionPolicy(ctx)
	if err != nil || policy.Validate() != nil {
		return 0, fmt.Errorf("load management session policy: %w", ErrAuthenticationUnavailable)
	}
	return policy.AccessTokenTTL, nil
}

func (r SessionRuntime) tokenIDGenerator() TokenIDGenerator {
	if r.NewTokenID != nil {
		return r.NewTokenID
	}
	return func() (string, error) {
		value, err := uuid.NewRandom()
		return value.String(), err
	}
}

func validateClaimsAgainstSession(claims Claims, session LiveSession, maximumTTL time.Duration) error {
	if claims.Subject != session.PrincipalID || claims.SessionID != session.ID ||
		claims.TokenID != session.TokenID || claims.Audience != session.Audience ||
		claims.AuthSourceKind != string(session.AuthSourceKind) || claims.AuthSourceID != session.AuthSourceID ||
		claims.EvidenceKind != session.EvidenceKind || claims.ExpiresAt > session.ExpiresAt.Unix() ||
		claims.ExpiresAt-claims.IssuedAt > int64(maximumTTL/time.Second) {
		return ErrAuthenticationDenied
	}
	if session.Human != nil {
		if claims.Human == nil || claims.Workload != nil ||
			claims.Human.AuthenticationTime != session.Human.AuthenticationTime ||
			claims.Human.AAL != session.Human.AAL || !slices.Equal(claims.Human.AMR, session.Human.AMR) {
			return ErrAuthenticationDenied
		}
	}
	if session.Workload != nil {
		if claims.Workload == nil || claims.Human != nil ||
			claims.Workload.Class != session.Workload.Class ||
			claims.Workload.SourceAssuredAt != session.Workload.SourceAssuredAt {
			return ErrAuthenticationDenied
		}
	}
	return nil
}
