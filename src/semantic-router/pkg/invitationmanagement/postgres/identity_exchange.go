package postgres

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	authpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
)

// AtomicExchangeStore composes invitation persistence with the Management
// session transaction seam. Embedded Store methods satisfy the ordinary
// invitation repository without adding a second database authority.
type AtomicExchangeStore struct {
	*Store
	sessions *authpostgres.Store
}

func NewAtomicExchangeStore(invitations *Store, sessions *authpostgres.Store) (*AtomicExchangeStore, error) {
	if invitations == nil || invitations.database == nil || sessions == nil {
		return nil, invitationmanagement.ErrUnavailable
	}
	return &AtomicExchangeStore{Store: invitations, sessions: sessions}, nil
}

func (store *AtomicExchangeStore) ExchangeIdentity(ctx context.Context,
	mutation invitationmanagement.AtomicIdentityExchangeMutation,
) (invitationmanagement.AtomicIdentityExchangeResult, error) {
	if store == nil || store.Store == nil || store.sessions == nil || mutation.IssueSession == nil ||
		mutation.Session.PrincipalID != "" {
		return invitationmanagement.AtomicIdentityExchangeResult{}, invitationmanagement.ErrInvalidRequest
	}
	if mutation.Acceptance == nil {
		if mutation.OpenAcceptance != nil || mutation.Identity.Issuer == "" || mutation.Identity.Subject == "" {
			return invitationmanagement.AtomicIdentityExchangeResult{}, invitationmanagement.ErrInvalidRequest
		}
		return inTransaction(ctx, store.Store, sql.LevelSerializable,
			func(tx *sql.Tx) (invitationmanagement.AtomicIdentityExchangeResult, error) {
				if err := store.sessions.RejectLoggedOutIssuerIdentityInTransaction(
					ctx, tx, mutation.Identity,
				); err != nil {
					return invitationmanagement.AtomicIdentityExchangeResult{}, err
				}
				principalID, err := findPrincipal(ctx, tx, mutation.Identity.Issuer, mutation.Identity.Subject)
				if err != nil {
					if errors.Is(err, invitationmanagement.ErrNotFound) {
						return invitationmanagement.AtomicIdentityExchangeResult{}, managementauth.ErrAuthenticationDenied
					}
					return invitationmanagement.AtomicIdentityExchangeResult{}, err
				}
				draft := mutation.Session
				draft.PrincipalID = principalID
				if live, issued, found, err := store.sessions.ReissueIssuerSessionInTransaction(
					ctx,
					tx,
					draft,
					mutation.IssueSession,
				); err != nil {
					return invitationmanagement.AtomicIdentityExchangeResult{}, err
				} else if found {
					return invitationmanagement.AtomicIdentityExchangeResult{Session: live, Issued: issued}, nil
				}
				live, err := store.sessions.CreateInTransaction(ctx, tx, draft)
				if err != nil {
					return invitationmanagement.AtomicIdentityExchangeResult{}, err
				}
				issued, err := mutation.IssueSession(ctx, live, live.CreatedAt)
				if err != nil || !validIssuedSession(issued, live) {
					if err == nil {
						err = managementauth.ErrAuthenticationUnavailable
					}
					return invitationmanagement.AtomicIdentityExchangeResult{}, err
				}
				return invitationmanagement.AtomicIdentityExchangeResult{Session: live, Issued: issued}, nil
			})
	}
	if mutation.OpenAcceptance == nil {
		return invitationmanagement.AtomicIdentityExchangeResult{}, invitationmanagement.ErrInvalidRequest
	}
	return store.exchangeInvitation(ctx, mutation)
}

func (store *AtomicExchangeStore) exchangeInvitation(ctx context.Context,
	mutation invitationmanagement.AtomicIdentityExchangeMutation,
) (invitationmanagement.AtomicIdentityExchangeResult, error) {
	return inTransaction(ctx, store.Store, sql.LevelSerializable,
		func(tx *sql.Tx) (invitationmanagement.AtomicIdentityExchangeResult, error) {
			if err := store.sessions.RejectLoggedOutIssuerIdentityInTransaction(
				ctx, tx, mutation.Identity,
			); err != nil {
				return invitationmanagement.AtomicIdentityExchangeResult{}, err
			}
			var live managementauth.LiveSession
			var issued managementauth.IssuedToken
			hooks := acceptanceSessionHooks{
				create: func(ctx context.Context, tx *sql.Tx, principalID string, _ time.Time) (string, error) {
					draft := mutation.Session
					draft.PrincipalID = principalID
					var err error
					live, err = store.sessions.CreateInTransaction(ctx, tx, draft)
					if err != nil {
						return "", err
					}
					// Session creation reads its own database timestamp after the
					// invitation lock timestamp. Validate/sign at the session's
					// authoritative CreatedAt so sub-millisecond ordering cannot
					// make a freshly created session appear to be from the future.
					issued, err = mutation.IssueSession(ctx, live, live.CreatedAt)
					if err != nil || !validIssuedSession(issued, live) {
						if err == nil {
							err = managementauth.ErrAuthenticationUnavailable
						}
						return "", err
					}
					return live.ID, nil
				},
				replay: func(ctx context.Context, tx *sql.Tx, sessionID, principalID string, now time.Time) error {
					var err error
					live, err = store.sessions.GetInTransaction(ctx, tx, sessionID)
					if err != nil || live.PrincipalID != principalID ||
						live.AuthSourceKind != managementauth.AuthSourceIssuer ||
						live.AuthSourceID != mutation.Session.AuthSourceID ||
						live.EvidenceKind != managementauth.EvidenceHuman {
						return managementauth.ErrAuthenticationDenied
					}
					issued, err = mutation.IssueSession(ctx, live, now)
					if err == nil && !validIssuedSession(issued, live) {
						err = managementauth.ErrAuthenticationUnavailable
					}
					return err
				},
			}
			envelope, err := acceptInTransaction(ctx, tx, *mutation.Acceptance, hooks)
			if err != nil {
				return invitationmanagement.AtomicIdentityExchangeResult{}, err
			}
			body, result, err := mutation.OpenAcceptance(envelope)
			if err != nil {
				return invitationmanagement.AtomicIdentityExchangeResult{}, err
			}
			deliveryActor := invitationmanagement.Actor{
				PrincipalID: result.PrincipalID, ActorChain: []string{result.PrincipalID},
				RequestID: mutation.Acceptance.Actor.RequestID,
				SourceIP:  mutation.Acceptance.Actor.SourceIP, Reason: "Deliver invitation onboarding result.",
			}
			if err := markAcceptanceDeliveredInTransaction(ctx, tx, envelope.Invitation.ID, deliveryActor); err != nil {
				zeroBytes(body)
				return invitationmanagement.AtomicIdentityExchangeResult{}, err
			}
			return invitationmanagement.AtomicIdentityExchangeResult{
				Session: live, Issued: issued, Acceptance: &result,
				CanonicalJSON: body, Replayed: envelope.Replayed,
			}, nil
		})
}

func validIssuedSession(issued managementauth.IssuedToken, session managementauth.LiveSession) bool {
	return issued.AccessToken != "" && issued.TokenType == "Bearer" && issued.ExpiresIn > 0 &&
		issued.ManagementSessionID == session.ID
}

func zeroBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

var _ invitationmanagement.AtomicIdentityExchangeRepository = (*AtomicExchangeStore)(nil)
