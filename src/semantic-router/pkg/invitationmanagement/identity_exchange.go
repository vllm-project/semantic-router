package invitationmanagement

import (
	"context"
	"errors"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

// AtomicIdentityExchangeMutation contains either an invitation acceptance or
// an existing-principal session exchange. Exactly one mode is selected by
// whether Acceptance is nil.
type AtomicIdentityExchangeMutation struct {
	Identity       managementauth.VerifiedExternalIdentity
	Acceptance     *AcceptMutation
	Session        managementauth.SessionDraft
	IssueSession   managementauth.PreparedSessionIssuer
	OpenAcceptance func(AcceptanceEnvelope) ([]byte, AcceptanceResult, error)
}

type AtomicIdentityExchangeResult struct {
	Session       managementauth.LiveSession
	Issued        managementauth.IssuedToken
	Acceptance    *AcceptanceResult
	CanonicalJSON []byte
	Replayed      bool
}

// AtomicIdentityExchangeRepository owns the single serializable PostgreSQL
// transaction spanning invitation materialization and Management session
// issuance. Implementations must not call either ordinary repository method.
type AtomicIdentityExchangeRepository interface {
	ExchangeIdentity(context.Context, AtomicIdentityExchangeMutation) (AtomicIdentityExchangeResult, error)
}

// IdentityExchangeCoordinator adapts verified issuer evidence to the narrow
// Management authentication coordinator contract.
type IdentityExchangeCoordinator struct {
	service    *Service
	repository AtomicIdentityExchangeRepository
}

func NewIdentityExchangeCoordinator(service *Service) (*IdentityExchangeCoordinator, error) {
	if service == nil || service.repository == nil {
		return nil, ErrUnavailable
	}
	repository, ok := service.repository.(AtomicIdentityExchangeRepository)
	if !ok {
		return nil, ErrUnavailable
	}
	return &IdentityExchangeCoordinator{service: service, repository: repository}, nil
}

func (coordinator *IdentityExchangeCoordinator) Ready(ctx context.Context) error {
	if coordinator == nil || coordinator.service == nil || coordinator.repository == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := coordinator.service.Ready(ctx); err != nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	return nil
}

func (coordinator *IdentityExchangeCoordinator) ExchangeIdentity(ctx context.Context,
	request managementauth.IdentityExchangeRequest,
	issue managementauth.PreparedSessionIssuer,
) (managementauth.IdentityExchangeResult, error) {
	if coordinator == nil || coordinator.service == nil || coordinator.repository == nil || issue == nil {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationUnavailable
	}
	if !verifiedIdentityMatchesSession(request.Identity, request.Session) {
		return managementauth.IdentityExchangeResult{}, managementauth.ErrAuthenticationDenied
	}
	mutation := AtomicIdentityExchangeMutation{
		Identity: request.Identity, Session: request.Session, IssueSession: issue,
	}
	if request.InvitationToken != "" {
		acceptance, err := coordinator.service.prepareAcceptance(ctx, AcceptRequest{
			Token: request.InvitationToken,
			Identity: AcceptanceIdentity{
				Issuer: request.Identity.Issuer, Subject: request.Identity.Subject,
				VerifiedEmail: request.Identity.VerifiedEmail, DisplayName: request.Identity.DisplayName,
			},
			AuthenticationSourceKind: string(managementauth.AuthSourceIssuer),
			AuthenticationSourceID:   request.Identity.IssuerID,
			EvidenceKind:             string(managementauth.EvidenceHuman),
			RequestID:                request.RequestID,
		})
		if err != nil {
			return managementauth.IdentityExchangeResult{}, mapExchangeError(err)
		}
		if acceptance.FirstKey != nil {
			defer zero(acceptance.FirstKey.Plaintext)
		}
		mutation.Acceptance = &acceptance
		mutation.OpenAcceptance = coordinator.service.openAcceptance
	}
	result, err := coordinator.repository.ExchangeIdentity(ctx, mutation)
	if err != nil {
		return managementauth.IdentityExchangeResult{}, mapExchangeError(err)
	}
	defer zero(result.CanonicalJSON)
	if result.Acceptance != nil {
		namespaceID := ""
		if mutation.Acceptance != nil {
			namespaceID = mutation.Acceptance.NamespaceID
		}
		if err := coordinator.service.waitFirstKeyActive(ctx, namespaceID, *result.Acceptance); err != nil {
			return managementauth.IdentityExchangeResult{}, mapExchangeError(err)
		}
	}
	exchange := managementauth.IdentityExchangeResult{Issued: result.Issued, Replayed: result.Replayed}
	if result.Acceptance != nil {
		exchange.Onboarding = &managementauth.InvitationOnboarding{
			InvitationID:      result.Acceptance.InvitationID,
			PrincipalID:       result.Acceptance.PrincipalID,
			UserID:            result.Acceptance.UserID,
			TeamID:            result.Acceptance.TeamID,
			APIKeyID:          result.Acceptance.APIKeyID,
			APIKey:            result.Acceptance.APIKey,
			DeliveryExpiresAt: result.Acceptance.DeliveryExpiresAt,
		}
	}
	return exchange, nil
}

func verifiedIdentityMatchesSession(identity managementauth.VerifiedExternalIdentity,
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

func mapExchangeError(err error) error {
	switch {
	case errors.Is(err, ErrExpired):
		return managementauth.ErrInvitationExpired
	case errors.Is(err, ErrSecretExpired):
		return managementauth.ErrInvitationResultExpired
	case errors.Is(err, ErrConflict), errors.Is(err, ErrAlreadyAccepted),
		errors.Is(err, ErrDefaultsChanged), errors.Is(err, ErrDelegationDenied):
		return managementauth.ErrInvitationConflict
	case errors.Is(err, ErrIdentityMismatch), errors.Is(err, ErrInvalidToken),
		errors.Is(err, ErrInvalidRequest), errors.Is(err, ErrNotFound):
		return managementauth.ErrAuthenticationDenied
	case errors.Is(err, managementauth.ErrAuthenticationDenied),
		errors.Is(err, managementauth.ErrSessionInactive),
		errors.Is(err, managementauth.ErrSessionNotFound),
		errors.Is(err, managementauth.ErrSessionLimitExceeded),
		errors.Is(err, managementauth.ErrSessionConflict):
		return managementauth.ErrAuthenticationDenied
	default:
		return managementauth.ErrAuthenticationUnavailable
	}
}

var _ managementauth.IdentityExchangeCoordinator = (*IdentityExchangeCoordinator)(nil)
