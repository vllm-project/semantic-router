package managementidentity

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type LifecycleService struct {
	repository LifecycleRepository
	sessions   managementauth.SessionRuntime
	barriers   BarrierAdmin
	issuerKeys IssuerKeyCache
	logout     managementauth.BackchannelLogoutVerifier
}

func NewLifecycleService(
	repository LifecycleRepository,
	sessions managementauth.SessionRuntime,
	barriers BarrierAdmin,
	issuerKeys IssuerKeyCache,
	logout managementauth.BackchannelLogoutVerifier,
) (*LifecycleService, error) {
	if repository == nil || sessions.Sessions == nil || sessions.Barriers == nil ||
		sessions.PolicyLoader == nil || barriers == nil || issuerKeys == nil || logout == nil {
		return nil, errors.New("management identity lifecycle dependencies are required")
	}
	return &LifecycleService{
		repository: repository, sessions: sessions, barriers: barriers,
		issuerKeys: issuerKeys, logout: logout,
	}, nil
}

func (service *LifecycleService) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.barriers == nil ||
		service.issuerKeys == nil || service.logout == nil {
		return errors.New("management identity lifecycle service is unavailable")
	}
	if err := service.repository.Ready(ctx); err != nil {
		return err
	}
	return service.barriers.Ready(ctx)
}

func (service *LifecycleService) Me(
	ctx context.Context,
	session managementauth.AuthenticatedSession,
) (SelfView, error) {
	if !canonicalUUID(session.Session.PrincipalID) || !canonicalUUID(session.Session.ID) ||
		session.Claims.Subject != session.Session.PrincipalID || session.Claims.SessionID != session.Session.ID {
		return SelfView{}, ErrInvalidLifecycleRequest
	}
	return service.repository.LoadSelf(ctx, session.Session.PrincipalID, session.Session.ID)
}

func (service *LifecycleService) ListManagementSessions(
	ctx context.Context,
	principalID string,
	request ListRequest,
) (ManagementSessionPage, error) {
	if !canonicalUUID(principalID) {
		return ManagementSessionPage{}, ErrNotFound
	}
	return service.repository.ListManagementSessions(ctx, principalID, request)
}

func (service *LifecycleService) RevokeSelfManagementSession(
	ctx context.Context,
	principalID string,
	sessionID string,
	actor MutationActor,
) (managementauth.SessionMutation, error) {
	if !canonicalUUID(principalID) || !canonicalUUID(sessionID) || validateActor(actor) != nil ||
		actor.PrincipalID != principalID {
		return managementauth.SessionMutation{}, ErrInvalidLifecycleRequest
	}
	mutation, err := service.repository.RevokeSelfManagementSession(ctx, sessionID, principalID, actor)
	if err != nil {
		return managementauth.SessionMutation{}, err
	}
	if err := service.installSessionBarriers(ctx, []string{sessionID}); err != nil {
		return mutation, err
	}
	return mutation, nil
}

func (service *LifecycleService) RevokeManagementSession(
	ctx context.Context,
	request SessionRevocationCommand,
) (managementauth.SessionMutation, MutationResult, error) {
	if !canonicalUUID(request.SessionID) || validateActor(request.Actor) != nil {
		return managementauth.SessionMutation{}, MutationResult{}, ErrInvalidLifecycleRequest
	}
	mutation, result, err := service.repository.RevokeManagementSession(ctx, request)
	if err != nil {
		return managementauth.SessionMutation{}, MutationResult{}, err
	}
	if err := service.installSessionBarriers(ctx, []string{request.SessionID}); err != nil {
		return mutation, result, err
	}
	return mutation, result, nil
}

func (service *LifecycleService) RevokePrincipalManagementSessions(
	ctx context.Context,
	request PrincipalSessionRevocationCommand,
) (PrincipalSessionRevocation, error) {
	if !canonicalUUID(request.PrincipalID) || validateActor(request.Actor) != nil {
		return PrincipalSessionRevocation{}, ErrInvalidLifecycleRequest
	}
	result, err := service.repository.RevokePrincipalManagementSessions(ctx, request)
	if err != nil {
		return PrincipalSessionRevocation{}, err
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	return result, nil
}

func (service *LifecycleService) GetTrustedIdentityIssuer(
	ctx context.Context,
	issuerID string,
) (TrustedIdentityIssuer, error) {
	if !canonicalUUID(issuerID) {
		return TrustedIdentityIssuer{}, ErrNotFound
	}
	return service.repository.GetTrustedIdentityIssuer(ctx, issuerID)
}

func (service *LifecycleService) ListTrustedIdentityIssuers(
	ctx context.Context,
	request ListRequest,
) (TrustedIdentityIssuerPage, error) {
	return service.repository.ListTrustedIdentityIssuers(ctx, request)
}

func (service *LifecycleService) CreateTrustedIdentityIssuer(
	ctx context.Context,
	request CreateTrustedIdentityIssuer,
) (IssuerMutation, error) {
	if err := validateTrustedIssuer(request.Issuer); err != nil ||
		request.Issuer.Status != managementauth.ResourceActive ||
		request.Issuer.Revision != 1 || validateActor(request.Actor) != nil {
		return IssuerMutation{}, ErrInvalidLifecycleRequest
	}
	result, err := service.repository.CreateTrustedIdentityIssuer(ctx, request)
	if err != nil {
		return IssuerMutation{}, err
	}
	service.issuerKeys.Invalidate(result.Issuer.ID)
	if err := service.issuerKeys.Refresh(ctx, result.Issuer.VerificationValue()); err != nil {
		return result, fmt.Errorf("refresh created trusted-issuer keys: %w", err)
	}
	return result, nil
}

func (service *LifecycleService) UpdateTrustedIdentityIssuer(
	ctx context.Context,
	request UpdateTrustedIdentityIssuer,
) (IssuerMutation, error) {
	if !canonicalUUID(request.ID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil || noIssuerPatch(request) {
		return IssuerMutation{}, ErrInvalidLifecycleRequest
	}
	current, updateTrustedIdentityIssuerErr := service.repository.GetTrustedIdentityIssuer(ctx, request.ID)
	if updateTrustedIdentityIssuerErr != nil {
		return IssuerMutation{}, updateTrustedIdentityIssuerErr
	}
	if current.Revision != request.ExpectedRevision {
		return IssuerMutation{}, ErrRevisionConflict
	}
	candidate := applyIssuerPatch(current, request)
	if err := validateTrustedIssuer(candidate); err != nil {
		return IssuerMutation{}, ErrInvalidLifecycleRequest
	}
	barrierID := issuerBarrierID(request.ID)
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return IssuerMutation{}, fmt.Errorf("install trusted-issuer revocation barrier: %w", err)
	}
	result, updateTrustedIdentityIssuerErr := service.repository.UpdateTrustedIdentityIssuer(ctx, request)
	if updateTrustedIdentityIssuerErr != nil {
		return IssuerMutation{}, updateTrustedIdentityIssuerErr
	}
	service.issuerKeys.Invalidate(request.ID)
	if result.Issuer.Status == managementauth.ResourceDisabled {
		return result, nil
	}
	if err := service.installSessionBarriers(ctx, result.Sessions); err != nil {
		return result, err
	}
	if err := service.issuerKeys.Refresh(ctx, result.Issuer.VerificationValue()); err != nil {
		return result, fmt.Errorf("refresh updated trusted-issuer keys: %w", err)
	}
	if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return result, fmt.Errorf("release trusted-issuer revocation barrier: %w", err)
	}
	return result, nil
}

func (service *LifecycleService) DeleteTrustedIdentityIssuer(
	ctx context.Context,
	issuerID string,
	expectedRevision uint64,
	actor MutationActor,
) (IssuerMutation, error) {
	if !canonicalUUID(issuerID) || expectedRevision == 0 || validateActor(actor) != nil {
		return IssuerMutation{}, ErrInvalidLifecycleRequest
	}
	current, deleteTrustedIdentityIssuerErr := service.repository.GetTrustedIdentityIssuer(ctx, issuerID)
	if deleteTrustedIdentityIssuerErr != nil {
		return IssuerMutation{}, deleteTrustedIdentityIssuerErr
	}
	if current.Revision != expectedRevision {
		return IssuerMutation{}, ErrRevisionConflict
	}
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource, issuerBarrierID(issuerID)); err != nil {
		return IssuerMutation{}, fmt.Errorf("install trusted-issuer delete barrier: %w", err)
	}
	result, deleteTrustedIdentityIssuerErr := service.repository.DeleteTrustedIdentityIssuer(ctx, issuerID, expectedRevision, actor)
	if deleteTrustedIdentityIssuerErr != nil {
		return IssuerMutation{}, deleteTrustedIdentityIssuerErr
	}
	service.issuerKeys.Invalidate(issuerID)
	return result, nil
}

func (service *LifecycleService) RefreshTrustedIdentityIssuer(
	ctx context.Context,
	request RefreshTrustedIdentityIssuer,
) (IssuerMutation, error) {
	if !canonicalUUID(request.ID) || validateActor(request.Actor) != nil {
		return IssuerMutation{}, ErrInvalidLifecycleRequest
	}
	current, err := service.repository.GetTrustedIdentityIssuer(ctx, request.ID)
	if err != nil {
		return IssuerMutation{}, err
	}
	if current.Status != managementauth.ResourceActive {
		return IssuerMutation{}, ErrNotFound
	}
	barrierID := issuerBarrierID(request.ID)
	result, err := service.repository.RefreshTrustedIdentityIssuer(ctx, request)
	if err != nil {
		return IssuerMutation{}, err
	}
	service.issuerKeys.Invalidate(request.ID)
	if err := service.issuerKeys.Refresh(ctx, result.Issuer.VerificationValue()); err != nil {
		return result, fmt.Errorf("refresh trusted-issuer keys: %w", err)
	}
	if err := service.installSessionBarriers(ctx, result.Sessions); err != nil {
		return result, err
	}
	if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return result, fmt.Errorf("release trusted-issuer source barrier: %w", err)
	}
	return result, nil
}

func (service *LifecycleService) BackchannelLogout(
	ctx context.Context,
	issuerID string,
	logoutToken string,
	requestID string,
	now time.Time,
) (BackchannelLogoutResult, error) {
	if !canonicalUUID(issuerID) || logoutToken == "" || requestID == "" || now.IsZero() {
		return BackchannelLogoutResult{}, managementauth.ErrAuthenticationDenied
	}
	identity, err := service.logout.VerifyBackchannelLogout(ctx, issuerID, logoutToken, now.UTC())
	if err != nil || identity.IssuerID != issuerID {
		return BackchannelLogoutResult{}, managementauth.ErrAuthenticationDenied
	}
	result, err := service.repository.ApplyBackchannelLogout(ctx, BackchannelLogout{
		Identity: identity, RequestID: requestID,
	})
	if err != nil {
		return BackchannelLogoutResult{}, err
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	return result, nil
}

func (service *LifecycleService) installSessionBarriers(ctx context.Context, sessionIDs []string) error {
	for _, sessionID := range sessionIDs {
		if !canonicalUUID(sessionID) {
			return errors.New("management session revocation returned an invalid identifier")
		}
		if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementSession, sessionID); err != nil {
			return fmt.Errorf("install Management session revocation barrier: %w", err)
		}
	}
	return nil
}

func validateTrustedIssuer(issuer TrustedIdentityIssuer) error {
	if !canonicalUUID(issuer.ID) || issuer.Revision == 0 || issuer.CreatedAt.IsZero() || issuer.UpdatedAt.IsZero() ||
		(issuer.Status != managementauth.ResourceActive && issuer.Status != managementauth.ResourceDisabled) {
		return ErrInvalidLifecycleRequest
	}
	return issuer.VerificationValue().Validate()
}

func noIssuerPatch(request UpdateTrustedIdentityIssuer) bool {
	return request.DiscoveryURL == nil && request.JWKSURL == nil && request.Audiences == nil &&
		request.ClaimMapping == nil && request.AssuranceMapping == nil && request.Status == nil
}

func applyIssuerPatch(current TrustedIdentityIssuer, request UpdateTrustedIdentityIssuer) TrustedIdentityIssuer {
	result := current
	if request.DiscoveryURL != nil {
		result.DiscoveryURL = *request.DiscoveryURL
	}
	if request.JWKSURL != nil {
		result.JWKSURL = *request.JWKSURL
	}
	if request.Audiences != nil {
		result.Audiences = append([]string(nil), (*request.Audiences)...)
	}
	if request.ClaimMapping != nil {
		result.ClaimMapping = cloneStringMap(*request.ClaimMapping)
	}
	if request.AssuranceMapping != nil {
		result.AssuranceMapping = cloneStringMap(*request.AssuranceMapping)
	}
	if request.Status != nil {
		result.Status = *request.Status
	}
	result.Revision++
	return result
}

func issuerBarrierID(issuerID string) string {
	return string(managementauth.AuthSourceIssuer) + ":" + issuerID
}
