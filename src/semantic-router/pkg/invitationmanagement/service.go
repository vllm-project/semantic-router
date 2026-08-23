package invitationmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize          = 50
	maximumPageSize          = 200
	minimumInvitationTTL     = 5 * time.Minute
	maximumInvitationTTL     = 30 * 24 * time.Hour
	minimumIdempotencyTTL    = time.Minute
	maximumIdempotencyTTL    = 7 * 24 * time.Hour
	minimumSecretDeliveryTTL = time.Minute
	maximumSecretDeliveryTTL = 24 * time.Hour
)

type Options struct {
	Repository        Repository
	Commands          *managementcommand.Codec
	CursorKeyring     securitykeyring.Symmetric
	InvitationPeppers accesscredential.PepperKeyring
	ResponseKEK       accesscredential.KEKKeyring
	FirstKeys         FirstKeyPreparer
	IdempotencyTTL    time.Duration
	SecretDeliveryTTL time.Duration
	Now               func() time.Time
	NewID             func() string
}

type Service struct {
	repository        Repository
	commands          *managementcommand.Codec
	cursors           cursorCodec
	tokens            *tokenCodec
	responseKEK       accesscredential.KEKKeyring
	firstKeys         FirstKeyPreparer
	idempotencyTTL    time.Duration
	secretDeliveryTTL time.Duration
	now               func() time.Time
	newID             func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Commands == nil ||
		options.IdempotencyTTL < minimumIdempotencyTTL || options.IdempotencyTTL > maximumIdempotencyTTL ||
		options.SecretDeliveryTTL < minimumSecretDeliveryTTL || options.SecretDeliveryTTL > maximumSecretDeliveryTTL ||
		options.ResponseKEK.Validate() != nil {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	tokens, err := newTokenCodec(options.InvitationPeppers)
	if err != nil {
		cursors.close()
		return nil, err
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = uuid.NewString
	}
	return &Service{
		repository: options.Repository, commands: options.Commands, cursors: cursors, tokens: tokens,
		responseKEK: options.ResponseKEK.Clone(), firstKeys: options.FirstKeys,
		idempotencyTTL: options.IdempotencyTTL, secretDeliveryTTL: options.SecretDeliveryTTL,
		now: now, newID: newID,
	}, nil
}

func (service *Service) Close() {
	if service == nil {
		return
	}
	service.tokens.Close()
	service.cursors.close()
	service.responseKEK.Close()
	if service.firstKeys != nil {
		service.firstKeys.Close()
		service.firstKeys = nil
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.tokens == nil || service.commands == nil {
		return ErrUnavailable
	}
	responseVersions := make([]string, 0, len(service.responseKEK.Keys))
	for version := range service.responseKEK.Keys {
		responseVersions = append(responseVersions, version)
	}
	if err := service.repository.Ready(ctx, service.commands, service.tokens.Versions(), responseVersions); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) Get(ctx context.Context, namespaceID, invitationID string) (Invitation, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(invitationID) {
		return Invitation{}, ErrInvalidRequest
	}
	value, err := service.repository.Get(ctx, namespaceID, invitationID)
	if err == nil {
		value.Status = value.EffectiveStatus(service.now().UTC())
	}
	return value, err
}

func (service *Service) List(ctx context.Context, request ListRequest) (Page, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		(request.Status != "" && !request.Status.Valid()) || request.PageSize < 0 || request.PageSize > maximumPageSize {
		return Page{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	query := InvitationQuery{NamespaceID: request.NamespaceID, Status: request.Status, Limit: pageSize, Now: service.now().UTC()}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "invitations" || cursor.NamespaceID != request.NamespaceID ||
			cursor.Status != request.Status || cursor.ExpiresAt.IsZero() || !canonicalUUID(cursor.ID) {
			return Page{}, ErrInvalidRequest
		}
		query.After = &InvitationCursor{ExpiresAt: cursor.ExpiresAt, ID: cursor.ID}
	}
	stored, err := service.repository.List(ctx, query)
	if err != nil {
		return Page{}, err
	}
	result := Page{Items: stored.Items, HasMore: stored.HasMore, PageSize: pageSize}
	for index := range result.Items {
		result.Items[index].Status = result.Items[index].EffectiveStatus(query.Now)
	}
	if result.HasMore {
		if len(result.Items) == 0 {
			return Page{}, ErrUnavailable
		}
		last := result.Items[len(result.Items)-1]
		result.NextCursor, err = service.cursors.encode(cursorPayload{
			Kind: "invitations", NamespaceID: request.NamespaceID,
			Status: request.Status, ExpiresAt: last.ExpiresAt, ID: last.ID,
		})
	}
	return result, err
}

func (service *Service) Create(ctx context.Context, request CreateRequest) (SecretResult, error) {
	if service == nil {
		return SecretResult{}, ErrUnavailable
	}
	now := service.timeNow()
	request.Expected.Issuer = strings.TrimSpace(request.Expected.Issuer)
	request.Expected.Subject = strings.TrimSpace(request.Expected.Subject)
	request.Expected.Email = accesscontrol.NormalizeEmail(request.Expected.Email)
	request.DisplayName = strings.TrimSpace(request.DisplayName)
	request.ExpiresAt = request.ExpiresAt.UTC().Truncate(time.Microsecond)
	if validateActor(request.NamespaceID, request.Actor) != nil ||
		validateExpected(request.Expected) != nil || validateText(request.DisplayName, 200) != nil ||
		request.ExpiresAt.Before(now.Add(minimumInvitationTTL)) || request.ExpiresAt.After(now.Add(maximumInvitationTTL)) ||
		validateRequestedGrants(request.RoleGrants) != nil || validateTeam(request.Team) != nil {
		return SecretResult{}, ErrInvalidRequest
	}
	canonicalRequest := struct {
		Expected    ExpectedIdentity     `json:"expected"`
		DisplayName string               `json:"displayName"`
		RoleGrants  []RequestedRoleGrant `json:"roleGrants"`
		Team        *TeamAssignment      `json:"team,omitempty"`
		ExpiresAt   time.Time            `json:"expiresAt"`
	}{request.Expected, request.DisplayName, request.RoleGrants, request.Team, request.ExpiresAt.UTC()}
	command, createErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/invitations", request.IdempotencyKey, canonicalRequest, now)
	if createErr != nil {
		return SecretResult{}, createErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return SecretResult{}, err
		}
		return service.replayInvitationSecret(ctx, command, stored, now)
	}
	snapshot, createErr := service.repository.ResolveSnapshot(ctx, request.NamespaceID, request.Actor.PrincipalID, request.RoleGrants, request.Team)
	if createErr != nil {
		return SecretResult{}, createErr
	}
	id, createErr := service.nextID()
	if createErr != nil {
		return SecretResult{}, createErr
	}
	token, digest, pepperVersion, createErr := service.tokens.Issue(id)
	if createErr != nil {
		return SecretResult{}, createErr
	}
	deliveryExpiresAt := now.Add(service.secretDeliveryTTL)
	invitation := Invitation{
		ID: id, NamespaceID: request.NamespaceID, CreatedByPrincipalID: request.Actor.PrincipalID,
		Expected: request.Expected, DisplayName: request.DisplayName, Snapshot: snapshot,
		ExpiresAt: request.ExpiresAt.UTC(), Status: StatusPending, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	body, createErr := json.Marshal(issuedInvitation{Data: invitation, Token: token, DeliveryExpiresAt: deliveryExpiresAt})
	if createErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	envelope, createErr := service.responseKEK.Seal(body, invitationSecretAAD(command.Endpoint, request.NamespaceID, id, 1))
	if createErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	created, createErr := service.repository.Create(ctx, CreateMutation{
		Invitation: invitation, Requested: request.RoleGrants, Team: request.Team, Command: command,
		TokenHMAC: digest, PepperVersion: pepperVersion, Response: envelope,
		ResponseExpiresAt: deliveryExpiresAt, Actor: request.Actor,
	})
	if createErr != nil {
		return SecretResult{}, createErr
	}
	if created.Replayed {
		if created.Stored == nil {
			return SecretResult{}, ErrUnavailable
		}
		return service.replayInvitationSecret(ctx, command, *created.Stored, now)
	}
	return SecretResult{Invitation: created.Invitation, Token: token, CanonicalJSON: body}, nil
}

func (service *Service) Rotate(ctx context.Context, request RotateRequest) (SecretResult, error) {
	if service == nil {
		return SecretResult{}, ErrUnavailable
	}
	now := service.timeNow()
	if validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.InvitationID) || request.ExpectedRevision == 0 {
		return SecretResult{}, ErrInvalidRequest
	}
	current, rotateErr := service.repository.Get(ctx, request.NamespaceID, request.InvitationID)
	if rotateErr != nil {
		return SecretResult{}, rotateErr
	}
	expiresAt := current.ExpiresAt
	if request.ExpiresAt != nil {
		expiresAt = request.ExpiresAt.UTC().Truncate(time.Microsecond)
	}
	if expiresAt.Before(now.Add(minimumInvitationTTL)) || expiresAt.After(now.Add(maximumInvitationTTL)) {
		return SecretResult{}, ErrInvalidRequest
	}
	endpoint := "/management/v1/invitations/" + request.InvitationID + ":rotate-token"
	command, rotateErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID, endpoint,
		request.IdempotencyKey, struct {
			ExpectedRevision uint64    `json:"expectedRevision"`
			ExpiresAt        time.Time `json:"expiresAt"`
		}{request.ExpectedRevision, expiresAt}, now)
	if rotateErr != nil {
		return SecretResult{}, rotateErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return SecretResult{}, err
		}
		return service.replayInvitationSecret(ctx, command, stored, now)
	}
	token, digest, pepperVersion, rotateErr := service.tokens.Issue(request.InvitationID)
	if rotateErr != nil {
		return SecretResult{}, rotateErr
	}
	updated := current
	updated.ExpiresAt, updated.Revision, updated.UpdatedAt = expiresAt, current.Revision+1, now
	deliveryExpiresAt := now.Add(service.secretDeliveryTTL)
	body, rotateErr := json.Marshal(issuedInvitation{Data: updated, Token: token, DeliveryExpiresAt: deliveryExpiresAt})
	if rotateErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	envelope, rotateErr := service.responseKEK.Seal(body, invitationSecretAAD(endpoint, request.NamespaceID, request.InvitationID, updated.Revision))
	if rotateErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	rotated, rotateErr := service.repository.Rotate(ctx, RotateMutation{
		NamespaceID: request.NamespaceID, InvitationID: request.InvitationID, ExpectedRevision: request.ExpectedRevision,
		ExpiresAt: &expiresAt, Command: command, TokenHMAC: digest, PepperVersion: pepperVersion,
		Response: envelope, ResponseExpiresAt: deliveryExpiresAt, Actor: request.Actor,
	})
	if rotateErr != nil {
		return SecretResult{}, rotateErr
	}
	if rotated.Replayed {
		if rotated.Stored == nil {
			return SecretResult{}, ErrUnavailable
		}
		return service.replayInvitationSecret(ctx, command, *rotated.Stored, now)
	}
	return SecretResult{Invitation: rotated.Invitation, Token: token, CanonicalJSON: body}, nil
}

func (service *Service) Revoke(ctx context.Context, request RevokeRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.InvitationID) || request.ExpectedRevision == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.Revoke(ctx, request)
}

// PrepareOnboarding resolves the immutable policy and delegation snapshot an
// adapter must use for exact authorization targets and the subsequent write.
func (service *Service) PrepareOnboarding(ctx context.Context, namespaceID, actorID string,
	grants []RequestedRoleGrant, team *TeamAssignment,
) (OnboardingSnapshot, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(actorID) ||
		validateRequestedGrants(grants) != nil || validateTeam(team) != nil {
		return OnboardingSnapshot{}, ErrInvalidRequest
	}
	return service.repository.ResolveSnapshot(ctx, namespaceID, actorID, grants, team)
}

// prepareAcceptance validates a Router-verified external identity and prepares
// secret-bearing transaction input. Only the atomic identity-exchange
// coordinator may execute the returned mutation.
func (service *Service) prepareAcceptance(ctx context.Context, request AcceptRequest) (AcceptMutation, error) {
	if service == nil {
		return AcceptMutation{}, ErrUnavailable
	}
	now := service.timeNow()
	request.Identity.Issuer = strings.TrimSpace(request.Identity.Issuer)
	request.Identity.Subject = strings.TrimSpace(request.Identity.Subject)
	request.Identity.VerifiedEmail = accesscontrol.NormalizeEmail(request.Identity.VerifiedEmail)
	request.Identity.DisplayName = strings.TrimSpace(request.Identity.DisplayName)
	if validateAcceptanceIdentity(request.Identity) != nil || validateText(request.RequestID, 256) != nil ||
		validateText(request.AuthenticationSourceKind, 64) != nil ||
		validateText(request.AuthenticationSourceID, 512) != nil ||
		(request.EvidenceKind != "human" && request.EvidenceKind != "workload") {
		return AcceptMutation{}, ErrInvalidRequest
	}
	invitationID, prepareAcceptanceErr := tokenInvitationID(request.Token)
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, ErrIdentityMismatch
	}
	invitation, digest, pepperVersion, prepareAcceptanceErr := service.repository.GetByID(ctx, invitationID)
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, concealInvitationError(prepareAcceptanceErr)
	}
	if err := service.tokens.Verify(request.Token, digest, pepperVersion); err != nil {
		return AcceptMutation{}, mapTokenError(err)
	}
	if !identityMatches(invitation.Expected, request.Identity) {
		return AcceptMutation{}, ErrIdentityMismatch
	}
	if invitation.Status == StatusPending && !now.Before(invitation.ExpiresAt) {
		return AcceptMutation{}, ErrExpired
	}
	userID, prepareAcceptanceErr := service.nextID()
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, prepareAcceptanceErr
	}
	principalID, prepareAcceptanceErr := service.nextID()
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, prepareAcceptanceErr
	}
	roleBindingIDs, prepareAcceptanceErr := service.nextIDs(len(invitation.Snapshot.RoleGrants))
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, prepareAcceptanceErr
	}
	accessBindingID, prepareAcceptanceErr := service.nextID()
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, prepareAcceptanceErr
	}
	rateBindingID, prepareAcceptanceErr := service.nextID()
	if prepareAcceptanceErr != nil {
		return AcceptMutation{}, prepareAcceptanceErr
	}
	var firstKey *PreparedFirstKey
	if invitation.Snapshot.AutomaticFirstKey {
		if service.firstKeys == nil {
			return AcceptMutation{}, ErrUnavailable
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
			return AcceptMutation{}, err
		}
		firstKey = &prepared
	}
	seal := func(result AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
		deliveryExpiresAt := now.Add(service.secretDeliveryTTL)
		result.DeliveryExpiresAt = deliveryExpiresAt
		if firstKey != nil {
			result.APIKeyID, result.APIKey = string(firstKey.Key.ID), string(firstKey.Plaintext)
		}
		body, err := json.Marshal(result)
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		envelope, err := service.responseKEK.Seal(body,
			acceptanceAAD(invitation.ID, userID, invitation.Revision+1))
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		return envelope, deliveryExpiresAt, nil
	}
	return AcceptMutation{
		InvitationID: invitation.ID, TokenHMAC: append([]byte(nil), digest...), PepperVersion: pepperVersion,
		Identity: request.Identity, PrincipalID: principalID, UserID: userID,
		RoleBindingIDs: roleBindingIDs, AccessBindingID: accessBindingID,
		RateLimitBindingID: rateBindingID,
		FirstKey:           firstKey, SealResult: seal,
		AuthenticationSourceKind: request.AuthenticationSourceKind,
		AuthenticationSourceID:   request.AuthenticationSourceID, EvidenceKind: request.EvidenceKind,
		Actor: Actor{
			PrincipalID: principalID, ActorChain: []string{principalID}, RequestID: request.RequestID,
			SourceIP: request.SourceIP, Reason: "Accept invitation.",
		},
	}, nil
}

func (service *Service) Onboard(ctx context.Context, request PrivilegedOnboardingRequest) (PrivilegedOnboardingResult, error) {
	if service == nil {
		return PrivilegedOnboardingResult{}, ErrUnavailable
	}
	now := service.timeNow()
	request.Email = accesscontrol.NormalizeEmail(request.Email)
	request.DisplayName = strings.TrimSpace(request.DisplayName)
	if validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.PrincipalID) || !validEmail(request.Email) ||
		validateText(request.DisplayName, 200) != nil || validateRequestedGrants(request.RoleGrants) != nil ||
		validateTeam(request.Team) != nil {
		return PrivilegedOnboardingResult{}, ErrInvalidRequest
	}
	command, onboardErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/onboarding", request.IdempotencyKey, struct {
			PrincipalID    string               `json:"principalId"`
			Email          string               `json:"email"`
			DisplayName    string               `json:"displayName"`
			RoleGrants     []RequestedRoleGrant `json:"roleGrants"`
			Team           *TeamAssignment      `json:"team,omitempty"`
			CreateFirstKey bool                 `json:"createFirstKey"`
		}{request.PrincipalID, request.Email, request.DisplayName, request.RoleGrants, request.Team, request.CreateFirstKey}, now)
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return PrivilegedOnboardingResult{}, err
		}
		return service.replayOnboarding(command, stored, now)
	}
	var snapshot OnboardingSnapshot
	if request.PreparedSnapshot == nil {
		var err error
		snapshot, err = service.repository.ResolveSnapshot(ctx, request.NamespaceID, request.Actor.PrincipalID, request.RoleGrants, request.Team)
		if err != nil {
			return PrivilegedOnboardingResult{}, err
		}
	} else {
		snapshot = cloneSnapshot(*request.PreparedSnapshot)
		if !snapshotMatchesRequest(snapshot, request.RoleGrants, request.Team) {
			return PrivilegedOnboardingResult{}, ErrInvalidRequest
		}
	}
	snapshot.AutomaticFirstKey = request.CreateFirstKey
	userID, onboardErr := service.nextID()
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	roleBindingIDs, onboardErr := service.nextIDs(len(snapshot.RoleGrants))
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	accessBindingID, onboardErr := service.nextID()
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	rateBindingID, onboardErr := service.nextID()
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	var firstKey *PreparedFirstKey
	if request.CreateFirstKey {
		if service.firstKeys == nil {
			return PrivilegedOnboardingResult{}, ErrUnavailable
		}
		teamID := ""
		if request.Team != nil {
			teamID = request.Team.TeamID
		}
		value, err := service.firstKeys.PrepareFirstKey(FirstKeyRequest{
			NamespaceID: request.NamespaceID,
			UserID:      userID, ContextTeamID: teamID, Name: request.DisplayName, Now: now,
		})
		if err != nil {
			return PrivilegedOnboardingResult{}, err
		}
		firstKey = &value
		defer zero(firstKey.Plaintext)
	}
	seal := func(result AcceptanceResult) (accesscredential.Envelope, time.Time, error) {
		expiresAt := now.Add(service.secretDeliveryTTL)
		result.DeliveryExpiresAt = expiresAt
		if firstKey != nil {
			result.APIKeyID, result.APIKey = string(firstKey.Key.ID), string(firstKey.Plaintext)
		}
		body, err := json.Marshal(result)
		if err != nil {
			return accesscredential.Envelope{}, time.Time{}, ErrUnavailable
		}
		envelope, err := service.responseKEK.Seal(body, onboardingAAD(request.NamespaceID, userID, 1))
		return envelope, expiresAt, err
	}
	envelope, onboardErr := service.repository.Onboard(ctx, PrivilegedOnboardingMutation{
		NamespaceID: request.NamespaceID, PrincipalID: request.PrincipalID, UserID: userID,
		Email: request.Email, DisplayName: request.DisplayName, Snapshot: snapshot,
		RoleBindingIDs: roleBindingIDs, AccessBindingID: accessBindingID, RateLimitBindingID: rateBindingID,
		FirstKey: firstKey, Command: command, SealResult: seal, Actor: request.Actor,
	})
	if onboardErr != nil {
		return PrivilegedOnboardingResult{}, onboardErr
	}
	body, result, onboardErr := service.openAcceptance(envelope)
	return PrivilegedOnboardingResult{Result: result, CanonicalJSON: body, Replayed: envelope.Replayed}, onboardErr
}

type issuedInvitation struct {
	Data              Invitation `json:"data"`
	Token             string     `json:"token"`
	DeliveryExpiresAt time.Time  `json:"deliveryExpiresAt"`
}

func (service *Service) replayInvitationSecret(_ context.Context, command managementcommand.Command, stored StoredSecret, now time.Time) (SecretResult, error) {
	if stored.Result.ResourceType != "invitation" || stored.Secret.ExpiresAt.IsZero() || !now.Before(stored.Secret.ExpiresAt) {
		return SecretResult{}, ErrSecretExpired
	}
	envelope := accesscredential.Envelope{
		KeyVersion: stored.Secret.KEKVersion,
		Nonce:      stored.Secret.Nonce, Ciphertext: stored.Secret.Ciphertext,
	}
	body, err := service.responseKEK.Open(envelope, invitationSecretAAD(command.Endpoint,
		command.Scope.NamespaceID, stored.Result.ResourceID, stored.Result.ResourceRevision))
	if err != nil {
		return SecretResult{}, ErrUnavailable
	}
	var decoded issuedInvitation
	if json.Unmarshal(body, &decoded) != nil || decoded.Data.ID != stored.Result.ResourceID ||
		decoded.Data.Revision != stored.Result.ResourceRevision || decoded.Token == "" {
		zero(body)
		return SecretResult{}, ErrUnavailable
	}
	return SecretResult{Invitation: decoded.Data, Token: decoded.Token, CanonicalJSON: body, Replayed: true}, nil
}

func (service *Service) replayOnboarding(command managementcommand.Command, stored StoredSecret, now time.Time) (PrivilegedOnboardingResult, error) {
	if stored.Result.ResourceType != "onboarding" || !now.Before(stored.Secret.ExpiresAt) {
		return PrivilegedOnboardingResult{}, ErrSecretExpired
	}
	envelope := accesscredential.Envelope{KeyVersion: stored.Secret.KEKVersion, Nonce: stored.Secret.Nonce, Ciphertext: stored.Secret.Ciphertext}
	body, err := service.responseKEK.Open(envelope, onboardingAAD(command.Scope.NamespaceID,
		stored.Result.ResourceID, stored.Result.ResourceRevision))
	if err != nil {
		return PrivilegedOnboardingResult{}, ErrUnavailable
	}
	var result AcceptanceResult
	if json.Unmarshal(body, &result) != nil || result.UserID != stored.Result.ResourceID {
		zero(body)
		return PrivilegedOnboardingResult{}, ErrUnavailable
	}
	return PrivilegedOnboardingResult{Result: result, CanonicalJSON: body, Replayed: true}, nil
}

func (service *Service) openAcceptance(stored AcceptanceEnvelope) ([]byte, AcceptanceResult, error) {
	now := service.timeNow()
	if stored.ExpiresAt.IsZero() || !now.Before(stored.ExpiresAt) {
		return nil, AcceptanceResult{}, ErrSecretExpired
	}
	aad := onboardingAAD(stored.Invitation.NamespaceID, stored.Invitation.AcceptedUserID, 1)
	if stored.Invitation.ID != "" {
		aad = acceptanceAAD(stored.Invitation.ID, stored.Invitation.AcceptedUserID, stored.Invitation.Revision)
	}
	body, err := service.responseKEK.Open(stored.Envelope, aad)
	if err != nil {
		return nil, AcceptanceResult{}, ErrUnavailable
	}
	var result AcceptanceResult
	if json.Unmarshal(body, &result) != nil || result.UserID != stored.Invitation.AcceptedUserID {
		zero(body)
		return nil, AcceptanceResult{}, ErrUnavailable
	}
	return body, result, nil
}

func (service *Service) bindCommand(namespaceID, principalID, endpoint, idempotencyKey string, body any, now time.Time) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID), principalID,
		endpoint, idempotencyKey, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	return command, nil
}

func (service *Service) nextID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", ErrUnavailable
	}
	return value, nil
}

func (service *Service) nextIDs(count int) ([]string, error) {
	result := make([]string, count)
	for index := range result {
		value, err := service.nextID()
		if err != nil {
			return nil, err
		}
		result[index] = value
	}
	return result, nil
}

func (service *Service) timeNow() time.Time { return service.now().UTC().Truncate(time.Microsecond) }

func invitationSecretAAD(endpoint, namespaceID, invitationID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/invitation-secret/v1\x00%s\x00%s\x00%s\x00%d", endpoint, namespaceID, invitationID, revision))
}

func acceptanceAAD(invitationID, userID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/invitation-acceptance/v1\x00%s\x00%s\x00%d", invitationID, userID, revision))
}

func onboardingAAD(namespaceID, userID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/privileged-onboarding/v1\x00%s\x00%s\x00%d", namespaceID, userID, revision))
}

func validateExpected(expected ExpectedIdentity) error {
	if validateText(expected.Issuer, 512) != nil || (expected.Subject == "" && expected.Email == "") ||
		(expected.Subject != "" && validateText(expected.Subject, 512) != nil) ||
		(expected.Email != "" && !validEmail(expected.Email)) {
		return ErrInvalidRequest
	}
	return nil
}

func validateAcceptanceIdentity(identity AcceptanceIdentity) error {
	if validateText(identity.Issuer, 512) != nil || validateText(identity.Subject, 512) != nil ||
		!validEmail(identity.VerifiedEmail) || (identity.DisplayName != "" && validateText(identity.DisplayName, 200) != nil) {
		return ErrInvalidRequest
	}
	return nil
}

func identityMatches(expected ExpectedIdentity, actual AcceptanceIdentity) bool {
	return expected.Issuer == actual.Issuer && (expected.Subject == "" || expected.Subject == actual.Subject) &&
		(expected.Email == "" || expected.Email == actual.VerifiedEmail)
}

func validateRequestedGrants(grants []RequestedRoleGrant) error {
	if len(grants) == 0 || len(grants) > 16 {
		return ErrInvalidRequest
	}
	seen := make(map[string]struct{}, len(grants))
	for _, grant := range grants {
		if !canonicalUUID(grant.RoleID) || (grant.ScopeKind != "namespace" && grant.ScopeKind != "user") {
			return ErrInvalidRequest
		}
		identity := grant.RoleID + "\x00" + grant.ScopeKind
		if _, duplicate := seen[identity]; duplicate {
			return ErrInvalidRequest
		}
		seen[identity] = struct{}{}
		permissions := make([]accesscontrol.Permission, len(grant.DelegationCeiling))
		for index, permission := range grant.DelegationCeiling {
			permissions[index] = accesscontrol.Permission(permission)
		}
		set, err := accesscontrol.NewPermissionSet(permissions...)
		if err != nil || set.ValidateDelegable() != nil {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateTeam(team *TeamAssignment) error {
	if team == nil {
		return nil
	}
	if !canonicalUUID(team.TeamID) || !team.Role.Valid() {
		return ErrInvalidRequest
	}
	return nil
}

func validateActor(namespaceID string, actor Actor) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) ||
		validateText(actor.RequestID, 256) != nil || validateText(actor.Reason, 512) != nil ||
		len(actor.ActorChain) == 0 || len(actor.ActorChain) > 32 || actor.ActorChain[0] != actor.PrincipalID {
		return ErrInvalidRequest
	}
	for _, principalID := range actor.ActorChain {
		if !canonicalUUID(principalID) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateText(value string, maximum int) error {
	if value == "" || strings.TrimSpace(value) != value || len(value) > maximum {
		return ErrInvalidRequest
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validEmail(value string) bool {
	if value == "" || len(value) > 320 || strings.Count(value, "@") != 1 || strings.ContainsAny(value, " \t\r\n") {
		return false
	}
	parts := strings.SplitN(value, "@", 2)
	return parts[0] != "" && parts[1] != ""
}

func snapshotMatchesRequest(snapshot OnboardingSnapshot, grants []RequestedRoleGrant, team *TeamAssignment) bool {
	if len(snapshot.RoleGrants) != len(grants) || (snapshot.Team == nil) != (team == nil) {
		return false
	}
	if team != nil && (snapshot.Team.TeamID != team.TeamID || snapshot.Team.Role != team.Role) {
		return false
	}
	for index := range grants {
		pinned, requested := snapshot.RoleGrants[index], grants[index]
		if pinned.RoleID != requested.RoleID || pinned.ScopeKind != requested.ScopeKind ||
			!slices.Equal(pinned.DelegationCeiling, requested.DelegationCeiling) {
			return false
		}
	}
	return snapshot.SelfServicePolicyRevision > 0 && canonicalUUID(snapshot.AccessPolicyID) &&
		snapshot.AccessPolicyRevision > 0 && canonicalUUID(snapshot.RateLimitPolicyID) &&
		snapshot.RateLimitPolicyRevision > 0
}

func cloneSnapshot(snapshot OnboardingSnapshot) OnboardingSnapshot {
	result := snapshot
	result.RoleGrants = append([]RoleGrant(nil), snapshot.RoleGrants...)
	for index := range result.RoleGrants {
		result.RoleGrants[index].DelegationCeiling = append([]string(nil), result.RoleGrants[index].DelegationCeiling...)
	}
	if snapshot.Team != nil {
		team := *snapshot.Team
		result.Team = &team
	}
	return result
}

func concealInvitationError(err error) error {
	if errors.Is(err, ErrNotFound) {
		return ErrIdentityMismatch
	}
	return err
}
