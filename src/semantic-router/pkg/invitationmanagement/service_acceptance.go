package invitationmanagement

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type onboardingIDs struct {
	principalID     string
	userID          string
	roleBindingIDs  []string
	accessBindingID string
	rateBindingID   string
}

func normalizeAcceptRequest(request AcceptRequest) AcceptRequest {
	request.Identity.Issuer = strings.TrimSpace(request.Identity.Issuer)
	request.Identity.Subject = strings.TrimSpace(request.Identity.Subject)
	request.Identity.VerifiedEmail = accesscontrol.NormalizeEmail(request.Identity.VerifiedEmail)
	request.Identity.DisplayName = strings.TrimSpace(request.Identity.DisplayName)
	return request
}

func validateAcceptRequest(request AcceptRequest) error {
	if validateAcceptanceIdentity(request.Identity) != nil || validateText(request.RequestID, 256) != nil ||
		validateText(request.AuthenticationSourceKind, 64) != nil ||
		validateText(request.AuthenticationSourceID, 512) != nil ||
		(request.EvidenceKind != "human" && request.EvidenceKind != "workload") {
		return ErrInvalidRequest
	}
	return nil
}

func (service *Service) authorizeInvitationAcceptance(
	ctx context.Context,
	request AcceptRequest,
	now time.Time,
) (Invitation, []byte, string, error) {
	invitationID, err := tokenInvitationID(request.Token)
	if err != nil {
		return Invitation{}, nil, "", ErrIdentityMismatch
	}
	invitation, digest, pepperVersion, err := service.repository.GetByID(ctx, invitationID)
	if err != nil {
		return Invitation{}, nil, "", concealInvitationError(err)
	}
	if err := service.tokens.Verify(request.Token, digest, pepperVersion); err != nil {
		return Invitation{}, nil, "", mapTokenError(err)
	}
	if !identityMatches(invitation.Expected, request.Identity) {
		return Invitation{}, nil, "", ErrIdentityMismatch
	}
	if invitation.Status == StatusPending && !now.Before(invitation.ExpiresAt) {
		return Invitation{}, nil, "", ErrExpired
	}
	return invitation, digest, pepperVersion, nil
}

func (service *Service) allocateOnboardingIDs(snapshot OnboardingSnapshot) (onboardingIDs, error) {
	userID, err := service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	principalID, err := service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	roleBindingIDs, err := service.nextIDs(len(snapshot.RoleGrants))
	if err != nil {
		return onboardingIDs{}, err
	}
	ids := onboardingIDs{
		principalID: principalID, userID: userID, roleBindingIDs: roleBindingIDs,
	}
	if snapshot.Team != nil {
		return ids, nil
	}
	ids.accessBindingID, err = service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	ids.rateBindingID, err = service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	return ids, nil
}

func (service *Service) prepareAutomaticFirstKey(
	invitation Invitation,
	userID string,
	now time.Time,
) (*PreparedFirstKey, error) {
	if !invitation.Snapshot.AutomaticFirstKey {
		return nil, nil
	}
	if service.firstKeys == nil {
		return nil, ErrUnavailable
	}
	teamID := ""
	if invitation.Snapshot.Team != nil {
		teamID = invitation.Snapshot.Team.TeamID
	}
	prepared, err := service.firstKeys.PrepareFirstKey(FirstKeyRequest{
		NamespaceID: invitation.NamespaceID, UserID: userID, ContextTeamID: teamID,
		Name: invitation.DisplayName, Now: now,
	})
	if err != nil {
		return nil, err
	}
	return &prepared, nil
}

func (service *Service) acceptanceSealer(
	invitation Invitation,
	userID string,
	firstKey *PreparedFirstKey,
	now time.Time,
) func(AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
	return func(result AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
		deliveryExpiresAt := now.Add(service.secretDeliveryTTL)
		result.DeliveryExpiresAt = deliveryExpiresAt
		if firstKey != nil {
			result.APIKeyID, result.APIKey = string(firstKey.Key.ID), string(firstKey.Plaintext)
		}
		body, err := json.Marshal(result)
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		envelope, err := service.responseKEK.Seal(
			body, acceptanceAAD(invitation.ID, userID, invitation.Revision+1),
		)
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		return envelope, deliveryExpiresAt, nil
	}
}

type onboardingCommandBody struct {
	PrincipalID    string               `json:"principalId"`
	Email          string               `json:"email"`
	DisplayName    string               `json:"displayName"`
	RoleGrants     []RequestedRoleGrant `json:"roleGrants"`
	Team           *TeamAssignment      `json:"team,omitempty"`
	CreateFirstKey bool                 `json:"createFirstKey"`
}

func (service *Service) bindOnboardingCommand(
	request PrivilegedOnboardingRequest,
	now time.Time,
) (managementcommand.Command, error) {
	return service.bindCommand(
		request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/onboarding", request.IdempotencyKey,
		onboardingCommandBody{
			PrincipalID: request.PrincipalID, Email: request.Email, DisplayName: request.DisplayName,
			RoleGrants: request.RoleGrants, Team: request.Team, CreateFirstKey: request.CreateFirstKey,
		},
		now,
	)
}

func (service *Service) resolveOnboardingSnapshot(
	ctx context.Context,
	request PrivilegedOnboardingRequest,
) (OnboardingSnapshot, error) {
	if request.PreparedSnapshot == nil {
		return service.repository.ResolveSnapshot(
			ctx, request.NamespaceID, request.Actor.PrincipalID, request.RoleGrants, request.Team,
		)
	}
	snapshot := cloneSnapshot(*request.PreparedSnapshot)
	if !snapshotMatchesRequest(snapshot, request.RoleGrants, request.Team) {
		return OnboardingSnapshot{}, ErrInvalidRequest
	}
	return snapshot, nil
}

func (service *Service) allocateOnboardingMaterializationIDs(
	snapshot OnboardingSnapshot,
) (onboardingIDs, error) {
	userID, err := service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	roleBindingIDs, err := service.nextIDs(len(snapshot.RoleGrants))
	if err != nil {
		return onboardingIDs{}, err
	}
	ids := onboardingIDs{userID: userID, roleBindingIDs: roleBindingIDs}
	if snapshot.Team != nil {
		return ids, nil
	}
	ids.accessBindingID, err = service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	ids.rateBindingID, err = service.nextID()
	if err != nil {
		return onboardingIDs{}, err
	}
	return ids, nil
}

func (service *Service) prepareOnboardingFirstKey(
	request PrivilegedOnboardingRequest,
	userID string,
	now time.Time,
) (*PreparedFirstKey, error) {
	if !request.CreateFirstKey {
		return nil, nil
	}
	if service.firstKeys == nil {
		return nil, ErrUnavailable
	}
	teamID := ""
	if request.Team != nil {
		teamID = request.Team.TeamID
	}
	value, err := service.firstKeys.PrepareFirstKey(FirstKeyRequest{
		NamespaceID: request.NamespaceID, UserID: userID, ContextTeamID: teamID,
		Name: request.DisplayName, Now: now,
	})
	if err != nil {
		return nil, err
	}
	return &value, nil
}

func (service *Service) onboardingSealer(
	namespaceID string,
	userID string,
	firstKey *PreparedFirstKey,
	now time.Time,
) func(AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
	return func(result AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
		expiresAt := now.Add(service.secretDeliveryTTL)
		result.DeliveryExpiresAt = expiresAt
		if firstKey != nil {
			result.APIKeyID, result.APIKey = string(firstKey.Key.ID), string(firstKey.Plaintext)
		}
		body, err := json.Marshal(result)
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		envelope, err := service.responseKEK.Seal(body, onboardingAAD(namespaceID, userID, 1))
		return envelope, expiresAt, err
	}
}
