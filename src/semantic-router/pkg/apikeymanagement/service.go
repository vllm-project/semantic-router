package apikeymanagement

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize          = 50
	maximumPageSize          = 200
	maximumAccessOverrides   = 12
	maximumCredentialOverlap = 24 * time.Hour
)

type Options struct {
	Repository         Repository
	Waiter             PublicationWaiter
	Commands           *managementcommand.Codec
	CursorKeyring      securitykeyring.Symmetric
	APIKeyPeppers      accesscredential.PepperKeyring
	ResponseKEK        accesscredential.KEKKeyring
	RevealKEK          *accesscredential.KEKKeyring
	DefaultRevealable  bool
	IdempotencyTTL     time.Duration
	SecretDeliveryTTL  time.Duration
	PublicationTimeout time.Duration
	Now                func() time.Time
	NewID              func() string
}

type Service struct {
	repository         Repository
	waiter             PublicationWaiter
	commands           *managementcommand.Codec
	cursors            cursorCodec
	peppers            accesscredential.PepperKeyring
	responseKEK        accesscredential.KEKKeyring
	revealKEK          *accesscredential.KEKKeyring
	defaultRevealable  bool
	idempotencyTTL     time.Duration
	secretTTL          time.Duration
	publicationTimeout time.Duration
	now                func() time.Time
	newID              func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Waiter == nil || options.Commands == nil || options.APIKeyPeppers.Validate() != nil ||
		options.ResponseKEK.Validate() != nil || options.IdempotencyTTL < time.Minute ||
		options.IdempotencyTTL > 7*24*time.Hour || options.SecretDeliveryTTL < time.Minute ||
		options.SecretDeliveryTTL > options.IdempotencyTTL || options.PublicationTimeout <= 0 ||
		options.PublicationTimeout > 30*time.Second {
		return nil, ErrUnavailable
	}
	if options.RevealKEK != nil {
		if err := options.RevealKEK.Validate(); err != nil {
			return nil, ErrUnavailable
		}
	} else if options.DefaultRevealable {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = uuid.NewString
	}
	peppers := options.APIKeyPeppers.Clone()
	responseKEK := options.ResponseKEK.Clone()
	var revealKEK *accesscredential.KEKKeyring
	if options.RevealKEK != nil {
		owned := options.RevealKEK.Clone()
		revealKEK = &owned
	}
	return &Service{
		repository: options.Repository, waiter: options.Waiter, commands: options.Commands, cursors: cursors,
		peppers: peppers, responseKEK: responseKEK,
		revealKEK: revealKEK, defaultRevealable: options.DefaultRevealable,
		idempotencyTTL: options.IdempotencyTTL, secretTTL: options.SecretDeliveryTTL,
		publicationTimeout: options.PublicationTimeout,
		now:                now, newID: newID,
	}, nil
}

func (service *Service) Close() {
	if service == nil {
		return
	}
	service.cursors.close()
	service.peppers.Close()
	service.responseKEK.Close()
	if service.revealKEK != nil {
		service.revealKEK.Close()
		service.revealKEK = nil
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) Get(ctx context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(keyID) {
		return accesscontrol.APIKey{}, ErrInvalidRequest
	}
	return service.repository.GetKey(ctx, namespaceID, keyID)
}

func (service *Service) List(ctx context.Context, request ListKeysRequest) (KeyPage, error) {
	pageSize, listErr := validatePage(request.NamespaceID, request.PageSize)
	if service == nil || listErr != nil || !validKeyStatusFilter(request.Status) ||
		!validOwnerFilter(request.OwnerKind, request.OwnerID) {
		return KeyPage{}, ErrInvalidRequest
	}
	search, listErr := managementsearch.Normalize(request.Search)
	if listErr != nil {
		return KeyPage{}, ErrInvalidRequest
	}
	scopeDigest, listErr := request.Scope.Digest()
	if listErr != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return KeyPage{}, ErrInvalidRequest
	}
	query := KeyQuery{
		NamespaceID: request.NamespaceID, Status: request.Status,
		OwnerKind: request.OwnerKind, OwnerID: request.OwnerID, Search: search,
		Scope: request.Scope, Limit: pageSize, IncludeTotal: request.IncludeTotal,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "keys" || cursor.NamespaceID != request.NamespaceID ||
			cursor.Status != string(request.Status) || cursor.OwnerKind != string(request.OwnerKind) ||
			cursor.OwnerID != request.OwnerID || cursor.Search != search || cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return KeyPage{}, ErrInvalidRequest
		}
		query.After = &KeyCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if !request.Scope.All && len(request.Scope.APIKeyIDs) == 0 &&
		len(request.Scope.UserIDs) == 0 && len(request.Scope.TeamIDs) == 0 {
		result := KeyPage{Items: []accesscontrol.APIKey{}, PageSize: pageSize}
		if request.IncludeTotal {
			count := uint64(0)
			result.TotalCount = &count
		}
		return result, nil
	}
	page, listErr := service.repository.ListKeys(ctx, query)
	if listErr != nil {
		return KeyPage{}, listErr
	}
	result := KeyPage{
		Items: page.Items, HasMore: page.HasMore, PageSize: pageSize, TotalCount: page.TotalCount,
	}
	if page.HasMore {
		if len(page.Items) == 0 {
			return KeyPage{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, listErr = service.cursors.encode(cursorPayload{
			Kind: "keys", NamespaceID: request.NamespaceID,
			OwnerKind: string(request.OwnerKind), OwnerID: request.OwnerID, Status: string(request.Status),
			Search: search, ScopeDigest: scopeDigest, CreatedAt: last.CreatedAt, ID: string(last.ID),
		})
	}
	return result, listErr
}

func (service *Service) Create(ctx context.Context, request CreateRequest) (SecretMutationResult, error) {
	if service == nil {
		return SecretMutationResult{}, ErrUnavailable
	}
	request.Name = strings.TrimSpace(request.Name)
	now := service.timeNow()
	accessPolicyIDs, createErr := canonicalAccessPolicyIDs(request.AccessPolicyIDs)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	rateLimitOverride, createErr := canonicalRateLimitOverride(request.RateLimitOverride)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	revealable := service.defaultRevealable
	if request.Revealable != nil {
		revealable = *request.Revealable
	}
	if validateActor(request.NamespaceID, request.Actor) != nil ||
		validateName(request.Name) != nil || validateOwner(request.NamespaceID, request.Owner, request.ContextTeamID) != nil ||
		validateFutureExpiry(request.ExpiresAt, now) != nil || (revealable && service.revealKEK == nil) {
		return SecretMutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Name              string                  `json:"name"`
		Owner             Owner                   `json:"owner"`
		ContextTeamID     string                  `json:"contextTeamId,omitempty"`
		ExpiresAt         *time.Time              `json:"expiresAt,omitempty"`
		Revealable        bool                    `json:"revealable"`
		AccessPolicyIDs   []string                `json:"accessPolicyIds,omitempty"`
		RateLimitOverride *RateLimitOverrideInput `json:"rateLimitOverride,omitempty"`
	}{
		request.Name, request.Owner, request.ContextTeamID, canonicalTime(request.ExpiresAt), revealable,
		accessPolicyIDs, rateLimitOverride,
	}
	command, createErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/api-keys", request.IdempotencyKey, canonical, now)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return SecretMutationResult{}, err
		}
		return service.replaySecret(ctx, command, stored, service.timeNow())
	}
	keyID, credentialID, createErr := service.twoIDs()
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	key := accesscontrol.APIKey{
		NamespaceID: accesscontrol.NamespaceID(request.NamespaceID), ID: accesscontrol.APIKeyID(keyID),
		Name: request.Name, Owner: accesscontrol.SubjectRef{
			NamespaceID: accesscontrol.NamespaceID(request.NamespaceID),
			ID:          accesscontrol.SubjectID(request.Owner.ID), Kind: request.Owner.Kind,
		},
		ContextTeamID: accesscontrol.TeamID(request.ContextTeamID), Status: accesscontrol.APIKeyStatusActive,
		ExpiresAt: canonicalTime(request.ExpiresAt), PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	credential, plaintext, createErr := service.issueCredential(request.NamespaceID, keyID, credentialID, key.ExpiresAt, revealable, now)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	accessBindings, accessReceipts, createErr := service.compileAccessBindings(request.NamespaceID, keyID, accessPolicyIDs, now)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	compiledRateOverride, rateReceipt, createErr := service.compileRateLimitOverride(
		request.NamespaceID, keyID, rateLimitOverride, now)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	deliveryExpiry := now.Add(service.secretTTL)
	body, createErr := marshalIssuedSecret(key, credential, plaintext, accessReceipts, rateReceipt, deliveryExpiry)
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	responseEnvelope, createErr := service.responseKEK.Seal(body, responseAAD(command.Endpoint, request.NamespaceID, keyID, 1))
	if createErr != nil {
		return SecretMutationResult{}, ErrUnavailable
	}
	created, createErr := service.repository.CreateKey(ctx, CreateMutation{
		Key: key, Credential: credential,
		AccessBindings: accessBindings, RateLimitOverride: compiledRateOverride,
		Command: command, Response: responseEnvelope, ResponseExpiresAt: deliveryExpiry, Actor: request.Actor,
	})
	if createErr != nil {
		return SecretMutationResult{}, createErr
	}
	if created.Replayed {
		if created.Stored == nil {
			return SecretMutationResult{}, ErrUnavailable
		}
		return service.replaySecret(ctx, command, *created.Stored, service.timeNow())
	}
	if err := service.waitAPIKeyActive(ctx, request.NamespaceID, keyID, credential.KID); err != nil {
		return SecretMutationResult{}, err
	}
	return secretMutation(created.Key, credential, plaintext, accessReceipts, rateReceipt, body, false), nil
}

func (service *Service) Rename(ctx context.Context, request RenameRequest) (MutationResult, error) {
	request.Name = strings.TrimSpace(request.Name)
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil ||
		validateName(request.Name) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	key, err := service.currentAtRevision(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision)
	if err != nil {
		return MutationResult{}, err
	}
	key.Name = request.Name
	return service.repository.UpdateKey(ctx, key, request.ExpectedRevision, request.Actor, "api_key.rename")
}

func (service *Service) Enable(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	return service.setStatus(ctx, request, accesscontrol.APIKeyStatusActive, "api_key.enable")
}

func (service *Service) Disable(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	return service.setStatus(ctx, request, accesscontrol.APIKeyStatusDisabled, "api_key.disable")
}

func (service *Service) Renew(ctx context.Context, request RenewRequest) (MutationResult, error) {
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil ||
		validateFutureExpiry(request.ExpiresAt, service.timeNow()) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.timeNow()
	command, renewErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/api-keys/"+request.KeyID+":renew", request.IdempotencyKey,
		struct {
			ExpiresAt *time.Time `json:"expiresAt"`
		}{canonicalTime(request.ExpiresAt)}, now)
	if renewErr != nil {
		return MutationResult{}, renewErr
	}
	if replay, found, err := service.repository.ReplayMutation(ctx, command); err != nil || found {
		return replay, err
	}
	key, renewErr := service.currentAtRevision(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision)
	if renewErr != nil {
		return MutationResult{}, renewErr
	}
	key.ExpiresAt = canonicalTime(request.ExpiresAt)
	return service.repository.UpdateKeyAction(ctx, UpdateMutation{
		Key: key, ExpectedRevision: request.ExpectedRevision,
		Command: command, Actor: request.Actor, Action: "api_key.renew", Reason: "Renew API key.",
	})
}

func (service *Service) Reassign(ctx context.Context, request ReassignRequest) (MutationResult, error) {
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil ||
		validateOwner(request.NamespaceID, request.Owner, request.ContextTeamID) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.timeNow()
	command, reassignErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/api-keys/"+request.KeyID+":reassign", request.IdempotencyKey,
		struct {
			Owner         Owner  `json:"owner"`
			ContextTeamID string `json:"contextTeamId,omitempty"`
		}{request.Owner, request.ContextTeamID}, now)
	if reassignErr != nil {
		return MutationResult{}, reassignErr
	}
	if replay, found, err := service.repository.ReplayMutation(ctx, command); err != nil || found {
		return replay, err
	}
	key, reassignErr := service.currentAtRevision(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision)
	if reassignErr != nil {
		return MutationResult{}, reassignErr
	}
	key.Owner = accesscontrol.SubjectRef{NamespaceID: key.NamespaceID, ID: accesscontrol.SubjectID(request.Owner.ID), Kind: request.Owner.Kind}
	key.ContextTeamID = accesscontrol.TeamID(request.ContextTeamID)
	key.PolicyEpoch++
	key.DelegationEpoch++
	return service.repository.UpdateKeyAction(ctx, UpdateMutation{
		Key: key, ExpectedRevision: request.ExpectedRevision,
		Command: command, Actor: request.Actor, Action: "api_key.reassign", Reason: "Reassign API key.",
	})
}

func (service *Service) Delete(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteKey(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor)
}

func (service *Service) ListCredentials(ctx context.Context, request ListCredentialsRequest) (CredentialPage, error) {
	pageSize, listCredentialsErr := validatePage(request.NamespaceID, request.PageSize)
	if service == nil || listCredentialsErr != nil || !canonicalUUID(request.KeyID) || !validCredentialStatusFilter(request.Status) {
		return CredentialPage{}, ErrInvalidRequest
	}
	query := CredentialQuery{NamespaceID: request.NamespaceID, KeyID: request.KeyID, Status: request.Status, Limit: pageSize}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "credentials" || cursor.NamespaceID != request.NamespaceID ||
			cursor.KeyID != request.KeyID || cursor.Status != string(request.Status) ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return CredentialPage{}, ErrInvalidRequest
		}
		query.After = &CredentialCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	page, listCredentialsErr := service.repository.ListCredentials(ctx, query)
	if listCredentialsErr != nil {
		return CredentialPage{}, listCredentialsErr
	}
	items := make([]CredentialMetadata, len(page.Items))
	for index := range page.Items {
		items[index] = credentialMetadata(page.Items[index].Credential, service.timeNow())
	}
	result := CredentialPage{Items: items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return CredentialPage{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1].Credential
		result.NextCursor, listCredentialsErr = service.cursors.encode(cursorPayload{
			Kind: "credentials", NamespaceID: request.NamespaceID,
			KeyID: request.KeyID, Status: string(request.Status), CreatedAt: last.CreatedAt, ID: string(last.ID),
		})
	}
	return result, listCredentialsErr
}

func (service *Service) Rotate(ctx context.Context, request RotateRequest) (SecretMutationResult, error) {
	if service == nil {
		return SecretMutationResult{}, ErrUnavailable
	}
	now := service.timeNow()
	revealable := service.defaultRevealable
	if request.Revealable != nil {
		revealable = *request.Revealable
	}
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil ||
		request.Overlap < 0 || request.Overlap > maximumCredentialOverlap || (revealable && service.revealKEK == nil) {
		return SecretMutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		OverlapNanoseconds int64 `json:"overlapNanoseconds"`
		Revealable         bool  `json:"revealable"`
	}{int64(request.Overlap), revealable}
	endpoint := "/management/v1/api-keys/" + request.KeyID + "/credentials:rotate"
	command, rotateErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID, endpoint, request.IdempotencyKey, canonical, now)
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return SecretMutationResult{}, err
		}
		return service.replaySecret(ctx, command, stored, service.timeNow())
	}
	key, rotateErr := service.currentAtRevision(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision)
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	active, rotateErr := service.repository.GetActiveCredential(ctx, request.NamespaceID, request.KeyID)
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	previousID := string(active.Credential.ID)
	credentialID, rotateErr := service.nextID()
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	credential, plaintext, rotateErr := service.issueCredential(request.NamespaceID, request.KeyID, credentialID, key.ExpiresAt, revealable, now)
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	deliveryExpiry := now.Add(service.secretTTL)
	updatedKey := key
	updatedKey.Revision++
	updatedKey.UpdatedAt = now
	body, rotateErr := marshalIssuedSecret(updatedKey, credential, plaintext, nil, nil, deliveryExpiry)
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	responseEnvelope, rotateErr := service.responseKEK.Seal(body, responseAAD(command.Endpoint, request.NamespaceID, request.KeyID, uint64(updatedKey.Revision)))
	if rotateErr != nil {
		return SecretMutationResult{}, ErrUnavailable
	}
	var retireAt *time.Time
	if request.Overlap > 0 {
		value := now.Add(request.Overlap)
		retireAt = &value
	}
	result, rotateErr := service.repository.RotateCredential(ctx, RotateMutation{
		NamespaceID: request.NamespaceID,
		KeyID:       request.KeyID, ExpectedRevision: request.ExpectedRevision, Credential: credential,
		PreviousCredentialID: previousID, RetireAt: retireAt, Command: command,
		Response: responseEnvelope, ResponseExpiresAt: deliveryExpiry, Actor: request.Actor,
	})
	if rotateErr != nil {
		return SecretMutationResult{}, rotateErr
	}
	if result.Replayed {
		if result.Stored == nil {
			return SecretMutationResult{}, ErrUnavailable
		}
		return service.replaySecret(ctx, command, *result.Stored, service.timeNow())
	}
	if err := service.waitAPIKeyActive(ctx, request.NamespaceID, request.KeyID, credential.KID); err != nil {
		return SecretMutationResult{}, err
	}
	return secretMutation(result.Key, credential, plaintext, nil, nil, body, false), nil
}

func (service *Service) RevokeCredential(ctx context.Context, request RevokeCredentialRequest) (MutationResult, error) {
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil ||
		!canonicalUUID(request.CredentialID) {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.RevokeCredential(ctx, request.NamespaceID, request.KeyID, request.CredentialID,
		request.ExpectedRevision, request.Actor)
}

func (service *Service) Reveal(ctx context.Context, request RevealRequest) (string, error) {
	if service == nil || service.revealKEK == nil {
		return "", ErrRevealDisabled
	}
	if validateActor(request.NamespaceID, request.Actor) != nil || !canonicalUUID(request.KeyID) || !canonicalUUID(request.CredentialID) {
		return "", ErrInvalidRequest
	}
	snapshot, err := service.repository.GetRevealSnapshot(ctx, request.NamespaceID, request.KeyID, request.CredentialID)
	if err != nil {
		return "", err
	}
	credential := snapshot.Credential
	plaintext, err := service.revealKEK.Open(accesscredential.Envelope{
		KeyVersion: credential.KEKVersion,
		Nonce:      credential.CiphertextNonce, Ciphertext: credential.SecretCiphertext,
	},
		revealAAD(request.NamespaceID, request.KeyID, request.CredentialID, credential.KID))
	if err != nil {
		return "", ErrCredentialUnavailable
	}
	if err := service.waitAPIKeyActive(ctx, request.NamespaceID, request.KeyID, credential.KID); err != nil {
		zero(plaintext)
		return "", err
	}
	if err := service.repository.RecordReveal(ctx, snapshot, request.Actor); err != nil {
		zero(plaintext)
		return "", err
	}
	result := string(plaintext)
	zero(plaintext)
	return result, nil
}
