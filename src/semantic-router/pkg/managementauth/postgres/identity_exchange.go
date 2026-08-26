package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

// IdentityExchangeCoordinator creates sessions for identities that already
// exist in durable Management state. Invitation onboarding is an Access
// capability and is therefore rejected when this coordinator is selected.
type IdentityExchangeCoordinator struct {
	db       *sql.DB
	sessions *Store
}

func NewIdentityExchangeCoordinator(
	db *sql.DB,
	sessions *Store,
) (*IdentityExchangeCoordinator, error) {
	if db == nil || sessions == nil {
		return nil, errors.New("management identity exchange PostgreSQL dependencies are required")
	}
	return &IdentityExchangeCoordinator{db: db, sessions: sessions}, nil
}

func (coordinator *IdentityExchangeCoordinator) Ready(ctx context.Context) error {
	if coordinator == nil || coordinator.db == nil || coordinator.sessions == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := coordinator.db.PingContext(ctx); err != nil {
		return fmt.Errorf("check Management identity exchange PostgreSQL: %w", err)
	}
	return nil
}

func (coordinator *IdentityExchangeCoordinator) ExchangeIdentity(
	ctx context.Context,
	request managementauth.IdentityExchangeRequest,
	issue managementauth.PreparedSessionIssuer,
) (managementauth.IdentityExchangeResult, error) {
	if coordinator == nil || coordinator.db == nil || coordinator.sessions == nil || issue == nil {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	if request.InvitationToken != "" || !existingIdentityMatchesSession(request.Identity, request.Session) {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationDenied
	}
	for attempt := 0; attempt < 4; attempt++ {
		result, err := coordinator.exchangeIdentityOnce(ctx, request, issue)
		if err == nil {
			return result, nil
		}
		if !retryableTransactionError(err) || ctx.Err() != nil {
			return managementauth.IdentityExchangeResult{}, err
		}
	}
	return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
}

func (coordinator *IdentityExchangeCoordinator) exchangeIdentityOnce(
	ctx context.Context,
	request managementauth.IdentityExchangeRequest,
	issue managementauth.PreparedSessionIssuer,
) (managementauth.IdentityExchangeResult, error) {
	tx, err := coordinator.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	defer func() { _ = tx.Rollback() }()
	if err := coordinator.sessions.RejectLoggedOutIssuerIdentityInTransaction(ctx, tx, request.Identity); err != nil {
		return managementauth.IdentityExchangeResult{}, err
	}
	var principalID string
	err = tx.QueryRowContext(ctx, `SELECT id::text FROM management_principals
WHERE issuer=$1 AND subject=$2 AND status='active'`, request.Identity.Issuer, request.Identity.Subject).Scan(&principalID)
	if errors.Is(err, sql.ErrNoRows) {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationDenied
	}
	if err != nil {
		if retryableTransactionError(err) {
			return managementauth.IdentityExchangeResult{}, err
		}
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	draft := request.Session
	draft.PrincipalID = principalID
	if _, issued, found, err := coordinator.sessions.ReissueIssuerSessionInTransaction(
		ctx,
		tx,
		draft,
		issue,
	); err != nil {
		return managementauth.IdentityExchangeResult{}, err
	} else if found {
		if err := tx.Commit(); err != nil {
			if retryableTransactionError(err) {
				return managementauth.IdentityExchangeResult{}, err
			}
			return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
		}
		return managementauth.IdentityExchangeResult{Issued: issued}, nil
	}
	live, err := coordinator.sessions.CreateInTransaction(ctx, tx, draft)
	if err != nil {
		return managementauth.IdentityExchangeResult{}, err
	}
	issued, err := issue(ctx, live, live.CreatedAt)
	if err != nil || !validIssuedIdentitySession(issued, live) {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	if err := tx.Commit(); err != nil {
		if retryableTransactionError(err) {
			return managementauth.IdentityExchangeResult{}, err
		}
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	return managementauth.IdentityExchangeResult{Issued: issued}, nil
}

func existingIdentityMatchesSession(
	identity managementauth.VerifiedExternalIdentity,
	session managementauth.SessionDraft,
) bool {
	if identity.IssuerID == "" || identity.Issuer == "" || identity.Subject == "" ||
		identity.AuthenticatedAt.IsZero() || identity.EvidenceExpiresAt.IsZero() ||
		session.PrincipalID != "" || session.AuthSourceKind != managementauth.AuthSourceIssuer ||
		session.AuthSourceID != identity.IssuerID || session.EvidenceKind != managementauth.EvidenceHuman ||
		session.Human == nil || session.Workload != nil ||
		!session.AuthenticatedAt.Equal(identity.AuthenticatedAt) ||
		!session.EvidenceExpiresAt.Equal(identity.EvidenceExpiresAt) ||
		session.Human.AuthenticationTime != identity.AuthenticatedAt.Unix() ||
		session.Human.AAL != identity.AAL || !slices.Equal(session.Human.AMR, identity.AMR) {
		return false
	}
	if (session.IssuerSessionID == nil) != (identity.IssuerSessionID == nil) {
		return false
	}
	return session.IssuerSessionID == nil || *session.IssuerSessionID == *identity.IssuerSessionID
}

func validIssuedIdentitySession(
	issued managementauth.IssuedToken,
	session managementauth.LiveSession,
) bool {
	return issued.AccessToken != "" && issued.TokenType == "Bearer" && issued.ExpiresIn > 0 &&
		issued.ManagementSessionID == session.ID
}

var _ managementauth.IdentityExchangeCoordinator = (*IdentityExchangeCoordinator)(nil)
