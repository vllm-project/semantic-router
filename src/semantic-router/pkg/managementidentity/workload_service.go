package managementidentity

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	workloadIdentityPageSize     = 50
	workloadIdentityMaximumPage  = 200
	workloadCredentialMinimumTTL = time.Hour
	workloadCredentialMaximumTTL = 365 * 24 * time.Hour
	workloadCredentialMaxOverlap = 24 * time.Hour
)

type WorkloadIdentityOptions struct {
	Repository          WorkloadIdentityRepository
	Commands            *managementcommand.Codec
	CursorKeyring       securitykeyring.Symmetric
	CredentialPeppers   securitykeyring.Symmetric
	ResponseKEK         accesscredential.KEKKeyring
	Barriers            BarrierAdmin
	SessionPolicy       managementauth.SessionPolicyLoader
	IdempotencyTTL      time.Duration
	SecretDeliveryTTL   time.Duration
	MTLSListenerEnabled bool
	Now                 func() time.Time
	NewID               func() string
	RandomBytes         func([]byte) error
}

type WorkloadIdentityService struct {
	repository          WorkloadIdentityRepository
	commands            *managementcommand.Codec
	cursors             workloadCursorCodec
	peppers             securitykeyring.Symmetric
	responseKEK         accesscredential.KEKKeyring
	barriers            BarrierAdmin
	sessionPolicy       managementauth.SessionPolicyLoader
	idempotencyTTL      time.Duration
	secretDeliveryTTL   time.Duration
	mtlsListenerEnabled bool
	now                 func() time.Time
	newID               func() string
	randomBytes         func([]byte) error
}

func NewWorkloadIdentityService(options WorkloadIdentityOptions) (*WorkloadIdentityService, error) {
	if options.Repository == nil || options.Commands == nil || options.Barriers == nil || options.SessionPolicy == nil ||
		options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour ||
		options.SecretDeliveryTTL < time.Minute || options.SecretDeliveryTTL > options.IdempotencyTTL ||
		validateWorkloadPeppers(options.CredentialPeppers) != nil || options.ResponseKEK.Validate() != nil {
		return nil, ErrWorkloadUnavailable
	}
	cursors, err := newWorkloadCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, ErrWorkloadUnavailable
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = uuid.NewString
	}
	randomBytes := options.RandomBytes
	if randomBytes == nil {
		randomBytes = func(target []byte) error { _, err := rand.Read(target); return err }
	}
	service := &WorkloadIdentityService{
		repository: options.Repository, commands: options.Commands, cursors: cursors,
		peppers: cloneWorkloadSymmetric(options.CredentialPeppers), responseKEK: cloneWorkloadKEK(options.ResponseKEK),
		barriers: options.Barriers, sessionPolicy: options.SessionPolicy,
		idempotencyTTL: options.IdempotencyTTL, secretDeliveryTTL: options.SecretDeliveryTTL,
		mtlsListenerEnabled: options.MTLSListenerEnabled, now: now, newID: newID, randomBytes: randomBytes,
	}
	return service, nil
}

func (service *WorkloadIdentityService) Close() {
	if service == nil {
		return
	}
	service.cursors.close()
	for _, key := range service.peppers.Keys {
		zeroWorkloadBytes(key)
	}
	for _, key := range service.responseKEK.Keys {
		zeroWorkloadBytes(key)
	}
	service.peppers = securitykeyring.Symmetric{}
	service.responseKEK = accesscredential.KEKKeyring{}
}

func (service *WorkloadIdentityService) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil || service.barriers == nil {
		return ErrWorkloadUnavailable
	}
	if err := service.repository.ReadyWorkloadIdentity(ctx, service.commands, service.mtlsListenerEnabled); err != nil {
		return err
	}
	return service.barriers.Ready(ctx)
}

func (service *WorkloadIdentityService) GetServiceAccount(ctx context.Context, id string) (ServiceAccount, error) {
	if service == nil || !canonicalUUID(id) {
		return ServiceAccount{}, ErrNotFound
	}
	return service.repository.GetServiceAccount(ctx, id)
}

func (service *WorkloadIdentityService) ListServiceAccounts(ctx context.Context, request ServiceAccountListRequest) (WorkloadPage[ServiceAccount], error) {
	pageSize, scopeDigest, err := validateServiceAccountList(request)
	if service == nil || err != nil {
		return WorkloadPage[ServiceAccount]{}, ErrInvalidWorkloadRequest
	}
	query := ServiceAccountQuery{Scope: canonicalServiceAccountScope(request.Scope), Status: request.Status, Limit: pageSize}
	if request.Cursor != "" {
		cursor, decodeErr := service.cursors.decode(request.Cursor)
		if decodeErr != nil || cursor.Kind != "service_accounts" || cursor.Status != string(request.Status) ||
			cursor.ScopeDigest != scopeDigest || !canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return WorkloadPage[ServiceAccount]{}, ErrInvalidWorkloadRequest
		}
		query.After = &ServiceAccountCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if !query.Scope.All && len(query.Scope.IDs) == 0 {
		return WorkloadPage[ServiceAccount]{Items: []ServiceAccount{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListServiceAccounts(ctx, query)
	if err != nil {
		return WorkloadPage[ServiceAccount]{}, err
	}
	result := WorkloadPage[ServiceAccount]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return WorkloadPage[ServiceAccount]{}, ErrWorkloadUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.cursors.encode(workloadCursorPayload{
			Kind: "service_accounts", Status: string(request.Status), ScopeDigest: scopeDigest,
			CreatedAt: last.CreatedAt, ID: last.ID,
		})
	}
	return result, err
}

func (service *WorkloadIdentityService) ListServiceCredentials(ctx context.Context, request ServiceCredentialListRequest) (WorkloadPage[ServiceCredential], error) {
	pageSize := canonicalWorkloadPageSize(request.PageSize)
	if service == nil || pageSize == 0 || !canonicalUUID(request.ServiceAccountID) {
		return WorkloadPage[ServiceCredential]{}, ErrInvalidWorkloadRequest
	}
	scopeDigest := workloadDigest("service_credentials", request.ServiceAccountID)
	query := ServiceCredentialQuery{ServiceAccountID: request.ServiceAccountID, Limit: pageSize}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "service_credentials" || cursor.OwnerID != request.ServiceAccountID ||
			cursor.ScopeDigest != scopeDigest || !canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return WorkloadPage[ServiceCredential]{}, ErrInvalidWorkloadRequest
		}
		query.After = &ServiceCredentialCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	page, err := service.repository.ListServiceCredentials(ctx, query)
	if err != nil {
		return WorkloadPage[ServiceCredential]{}, err
	}
	result := WorkloadPage[ServiceCredential]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return WorkloadPage[ServiceCredential]{}, ErrWorkloadUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.cursors.encode(workloadCursorPayload{
			Kind: "service_credentials", OwnerID: request.ServiceAccountID, ScopeDigest: scopeDigest,
			CreatedAt: last.CreatedAt, ID: last.ID,
		})
	}
	return result, err
}

func (service *WorkloadIdentityService) CreateServiceAccount(ctx context.Context, request CreateServiceAccountRequest) (ServiceCredentialSecretResult, error) {
	if service == nil {
		return ServiceCredentialSecretResult{}, ErrInvalidWorkloadRequest
	}
	now := service.timeNow()
	request.DisplayName = strings.TrimSpace(request.DisplayName)
	if validateWorkloadActor(request.Actor) != nil || validateServiceAccountOwner(request.OwnerScope, request.NamespaceID) != nil ||
		request.DisplayName == "" || len(request.DisplayName) > 200 || !validWorkloadClass(request.CredentialClass) ||
		validateCredentialExpiry(now, request.CredentialExpiresAt) != nil {
		return ServiceCredentialSecretResult{}, ErrInvalidWorkloadRequest
	}
	if request.CredentialClass == WorkloadStrong && !service.allowsStrongSource(ctx, request.Actor.Session, now) {
		return ServiceCredentialSecretResult{}, managementauth.ErrAuthenticationDenied
	}
	canonical := struct {
		DisplayName string                   `json:"displayName"`
		OwnerScope  ServiceAccountOwnerScope `json:"ownerScope"`
		NamespaceID string                   `json:"namespaceId,omitempty"`
		ExpiresAt   time.Time                `json:"credentialExpiresAt"`
		Class       WorkloadClass            `json:"credentialClass"`
	}{request.DisplayName, request.OwnerScope, request.NamespaceID, request.CredentialExpiresAt.UTC(), request.CredentialClass}
	command, createServiceAccountErr := service.bindCommand(request.OwnerScope, request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/service-accounts", request.IdempotencyKey, canonical, now)
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, createServiceAccountErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return ServiceCredentialSecretResult{}, err
		}
		return service.replayServiceCredential(ctx, command, stored, now)
	}
	accountID, principalID, credentialID, createServiceAccountErr := service.threeIDs()
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, createServiceAccountErr
	}
	credential, plaintext, digest, pepperVersion, createServiceAccountErr := service.issueCredential(credentialID, accountID, request.CredentialClass, now, request.CredentialExpiresAt.UTC())
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, createServiceAccountErr
	}
	defer zeroWorkloadString(&plaintext)
	account := ServiceAccount{
		ID: accountID, PrincipalID: principalID, DisplayName: request.DisplayName,
		OwnerScope: request.OwnerScope, NamespaceID: request.NamespaceID, Status: ServiceAccountActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	deliveryExpiry := now.Add(service.secretDeliveryTTL)
	body, createServiceAccountErr := json.Marshal(ServiceCredentialSecret{ServiceAccount: account, Credential: credential, Secret: plaintext, DeliveryExpiry: deliveryExpiry})
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	defer zeroWorkloadBytes(body)
	envelope, createServiceAccountErr := service.responseKEK.Seal(body, workloadResponseAAD(command.Endpoint, account.ID, account.Revision))
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	created, createServiceAccountErr := service.repository.CreateServiceAccount(ctx, ServiceAccountCreateMutation{
		Account: account, Credential: credential, SecretHMAC: digest, PepperVersion: pepperVersion,
		Command: command, Response: envelope, ResponseExpiresAt: deliveryExpiry, Actor: request.Actor.MutationActor(),
	})
	if createServiceAccountErr != nil {
		return ServiceCredentialSecretResult{}, createServiceAccountErr
	}
	if created.Replayed {
		if created.Stored == nil {
			return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
		}
		return service.replayServiceCredential(ctx, command, *created.Stored, service.timeNow())
	}
	return ServiceCredentialSecretResult{ServiceAccount: account, Credential: credential, Secret: plaintext, DeliveryExpiry: deliveryExpiry}, nil
}

func (service *WorkloadIdentityService) PatchServiceAccount(ctx context.Context, request PatchServiceAccountRequest) (WorkloadMutationResult, error) {
	if service == nil || !canonicalUUID(request.ID) || request.ExpectedRevision == 0 || validateWorkloadActor(request.Actor) != nil ||
		(request.DisplayName == nil && request.Status == nil) {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	current, patchServiceAccountErr := service.repository.GetServiceAccount(ctx, request.ID)
	if patchServiceAccountErr != nil {
		return WorkloadMutationResult{}, patchServiceAccountErr
	}
	if current.Revision != request.ExpectedRevision {
		return WorkloadMutationResult{}, ErrRevisionConflict
	}
	if request.DisplayName != nil {
		value := strings.TrimSpace(*request.DisplayName)
		if value == "" || len(value) > 200 {
			return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
		}
		current.DisplayName = value
	}
	if request.Status != nil {
		if *request.Status != ServiceAccountActive && *request.Status != ServiceAccountDisabled {
			return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
		}
		current.Status = *request.Status
	}
	disabling := request.Status != nil && *request.Status == ServiceAccountDisabled
	if disabling {
		if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementPrincipal, current.PrincipalID); err != nil {
			return WorkloadMutationResult{}, err
		}
	}
	result, patchServiceAccountErr := service.repository.PatchServiceAccount(ctx, current, request.ExpectedRevision, request.Actor.MutationActor())
	if patchServiceAccountErr != nil {
		return WorkloadMutationResult{}, patchServiceAccountErr
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	if request.Status != nil && *request.Status == ServiceAccountActive {
		if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierManagementPrincipal, current.PrincipalID); err != nil {
			return result, err
		}
	}
	return result, nil
}

func (service *WorkloadIdentityService) DeleteServiceAccount(ctx context.Context, request DeleteServiceAccountRequest) (WorkloadMutationResult, error) {
	if service == nil || !canonicalUUID(request.ID) || request.ExpectedRevision == 0 || validateWorkloadActor(request.Actor) != nil {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	current, deleteServiceAccountErr := service.repository.GetServiceAccount(ctx, request.ID)
	if deleteServiceAccountErr != nil {
		return WorkloadMutationResult{}, deleteServiceAccountErr
	}
	if current.Revision != request.ExpectedRevision {
		return WorkloadMutationResult{}, ErrRevisionConflict
	}
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementPrincipal, current.PrincipalID); err != nil {
		return WorkloadMutationResult{}, err
	}
	result, deleteServiceAccountErr := service.repository.DeleteServiceAccount(ctx, request.ID, request.ExpectedRevision, request.Actor.MutationActor())
	if deleteServiceAccountErr != nil {
		return WorkloadMutationResult{}, deleteServiceAccountErr
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	return result, nil
}

func (service *WorkloadIdentityService) RotateServiceCredential(ctx context.Context, request RotateServiceCredentialRequest) (ServiceCredentialSecretResult, error) {
	if service == nil {
		return ServiceCredentialSecretResult{}, ErrInvalidWorkloadRequest
	}
	now := service.timeNow()
	if !canonicalUUID(request.ServiceAccountID) || request.ExpectedRevision == 0 ||
		validateWorkloadActor(request.Actor) != nil || !validWorkloadClass(request.WorkloadClass) ||
		validateCredentialExpiry(now, request.ExpiresAt) != nil || request.Overlap < 0 || request.Overlap > workloadCredentialMaxOverlap {
		return ServiceCredentialSecretResult{}, ErrInvalidWorkloadRequest
	}
	if request.WorkloadClass == WorkloadStrong && !service.allowsStrongSource(ctx, request.Actor.Session, now) {
		return ServiceCredentialSecretResult{}, managementauth.ErrAuthenticationDenied
	}
	canonical := struct {
		ExpiresAt time.Time     `json:"expiresAt"`
		Class     WorkloadClass `json:"workloadClass"`
		Overlap   int64         `json:"overlapNanoseconds"`
	}{request.ExpiresAt.UTC(), request.WorkloadClass, int64(request.Overlap)}
	account, rotateServiceCredentialErr := service.repository.GetServiceAccount(ctx, request.ServiceAccountID)
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, rotateServiceCredentialErr
	}
	if account.Revision != request.ExpectedRevision || account.Status != ServiceAccountActive {
		return ServiceCredentialSecretResult{}, ErrRevisionConflict
	}
	command, rotateServiceCredentialErr := service.bindCommand(account.OwnerScope, account.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/service-accounts/"+account.ID+"/credentials:rotate", request.IdempotencyKey, canonical, now)
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, rotateServiceCredentialErr
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return ServiceCredentialSecretResult{}, err
		}
		return service.replayServiceCredential(ctx, command, stored, now)
	}
	credentialID, rotateServiceCredentialErr := service.nextID()
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, rotateServiceCredentialErr
	}
	credential, plaintext, digest, pepperVersion, rotateServiceCredentialErr := service.issueCredential(credentialID, account.ID, request.WorkloadClass, now, request.ExpiresAt.UTC())
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, rotateServiceCredentialErr
	}
	defer zeroWorkloadString(&plaintext)
	updated := account
	updated.Revision++
	updated.UpdatedAt = now
	deliveryExpiry := now.Add(service.secretDeliveryTTL)
	body, rotateServiceCredentialErr := json.Marshal(ServiceCredentialSecret{ServiceAccount: updated, Credential: credential, Secret: plaintext, DeliveryExpiry: deliveryExpiry})
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	defer zeroWorkloadBytes(body)
	envelope, rotateServiceCredentialErr := service.responseKEK.Seal(body, workloadResponseAAD(command.Endpoint, account.ID, updated.Revision))
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	retireAt := now.Add(request.Overlap)
	result, rotateServiceCredentialErr := service.repository.RotateServiceCredential(ctx, ServiceCredentialRotateMutation{
		AccountID: account.ID, ExpectedRevision: request.ExpectedRevision, Credential: credential,
		SecretHMAC: digest, PepperVersion: pepperVersion, RetireAt: retireAt, Command: command,
		Response: envelope, ResponseExpiresAt: deliveryExpiry, Actor: request.Actor.MutationActor(),
	})
	if rotateServiceCredentialErr != nil {
		return ServiceCredentialSecretResult{}, rotateServiceCredentialErr
	}
	if result.Replayed {
		if result.Stored == nil {
			return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
		}
		return service.replayServiceCredential(ctx, command, *result.Stored, service.timeNow())
	}
	if request.Overlap == 0 {
		if err := service.installSourceAndSessionBarriers(ctx, result.RevokedCredentialIDs, result.SessionIDs); err != nil {
			return ServiceCredentialSecretResult{}, err
		}
	}
	return ServiceCredentialSecretResult{ServiceAccount: updated, Credential: credential, Secret: plaintext, DeliveryExpiry: deliveryExpiry}, nil
}

func (service *WorkloadIdentityService) RevokeServiceCredential(ctx context.Context, request RevokeServiceCredentialRequest) (WorkloadMutationResult, error) {
	if service == nil || !canonicalUUID(request.ServiceAccountID) || !canonicalUUID(request.CredentialID) ||
		request.ExpectedRevision == 0 || validateWorkloadActor(request.Actor) != nil {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	barrierID := authSourceBarrierID(managementauth.AuthSourceServiceCredential, request.CredentialID)
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return WorkloadMutationResult{}, err
	}
	result, err := service.repository.RevokeServiceCredential(ctx, request.ServiceAccountID, request.CredentialID,
		request.ExpectedRevision, request.Actor.MutationActor())
	if err != nil {
		return WorkloadMutationResult{}, err
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	return result, nil
}

func (service *WorkloadIdentityService) GetMTLSMapping(ctx context.Context, id string) (MTLSIdentityMapping, error) {
	if service == nil || !canonicalUUID(id) {
		return MTLSIdentityMapping{}, ErrNotFound
	}
	return service.repository.GetMTLSMapping(ctx, id)
}

func (service *WorkloadIdentityService) ListMTLSMappings(ctx context.Context, request MTLSMappingListRequest) (WorkloadPage[MTLSIdentityMapping], error) {
	pageSize := canonicalWorkloadPageSize(request.PageSize)
	if service == nil || pageSize == 0 || (request.Status != "" && request.Status != managementauth.ResourceActive && request.Status != managementauth.ResourceDisabled) {
		return WorkloadPage[MTLSIdentityMapping]{}, ErrInvalidWorkloadRequest
	}
	scopeDigest := workloadDigest("mtls_mappings", string(request.Status))
	query := MTLSMappingQuery{Status: string(request.Status), Limit: pageSize}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "mtls_mappings" || cursor.Status != string(request.Status) ||
			cursor.ScopeDigest != scopeDigest || !canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return WorkloadPage[MTLSIdentityMapping]{}, ErrInvalidWorkloadRequest
		}
		query.After = &MTLSMappingCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	page, err := service.repository.ListMTLSMappings(ctx, query)
	if err != nil {
		return WorkloadPage[MTLSIdentityMapping]{}, err
	}
	result := WorkloadPage[MTLSIdentityMapping]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return WorkloadPage[MTLSIdentityMapping]{}, ErrWorkloadUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.cursors.encode(workloadCursorPayload{
			Kind: "mtls_mappings", Status: string(request.Status), ScopeDigest: scopeDigest,
			CreatedAt: last.CreatedAt, ID: last.ID,
		})
	}
	return result, err
}

func (service *WorkloadIdentityService) CreateMTLSMapping(ctx context.Context, request CreateMTLSMappingRequest) (WorkloadMutationResult, error) {
	if service == nil {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	now := service.timeNow()
	request.MatcherValue = strings.TrimSpace(request.MatcherValue)
	if !service.mtlsListenerEnabled || !canonicalUUID(request.PrincipalID) ||
		validateWorkloadActor(request.Actor) != nil || validateMTLSMatcher(request.MatcherKind, request.MatcherValue) != nil ||
		!validWorkloadClass(request.WorkloadClass) {
		if !service.mtlsListenerEnabled {
			return WorkloadMutationResult{}, ErrMTLSListenerUnavailable
		}
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	if request.WorkloadClass == WorkloadStrong && !service.allowsStrongSource(ctx, request.Actor.Session, now) {
		return WorkloadMutationResult{}, managementauth.ErrAuthenticationDenied
	}
	canonical := struct {
		MatcherKind   MTLSMatcherKind `json:"matcherKind"`
		MatcherValue  string          `json:"matcherValue"`
		PrincipalID   string          `json:"principalId"`
		WorkloadClass WorkloadClass   `json:"workloadClass"`
	}{request.MatcherKind, request.MatcherValue, request.PrincipalID, request.WorkloadClass}
	command, err := service.bindCommand(ServiceAccountOwnerCluster, "", request.Actor.PrincipalID,
		"/management/v1/mtls-identity-mappings", request.IdempotencyKey, canonical, now)
	if err != nil {
		return WorkloadMutationResult{}, err
	}
	id, err := service.nextID()
	if err != nil {
		return WorkloadMutationResult{}, err
	}
	return service.repository.CreateMTLSMapping(ctx, MTLSMappingCreateMutation{Mapping: MTLSIdentityMapping{
		ID: id, MatcherKind: request.MatcherKind, MatcherValue: request.MatcherValue, PrincipalID: request.PrincipalID,
		WorkloadClass: request.WorkloadClass, SourceAssuredAt: now, Status: managementauth.ResourceActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}, Command: command, Actor: request.Actor.MutationActor()})
}

func (service *WorkloadIdentityService) PatchMTLSMapping(ctx context.Context, request PatchMTLSMappingRequest) (WorkloadMutationResult, error) {
	if service == nil {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	now := service.timeNow()
	if !canonicalUUID(request.ID) || request.ExpectedRevision == 0 || validateWorkloadActor(request.Actor) != nil ||
		(request.Status == nil && request.WorkloadClass == nil) {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	current, patchMTLSMappingErr := service.repository.GetMTLSMapping(ctx, request.ID)
	if patchMTLSMappingErr != nil {
		return WorkloadMutationResult{}, patchMTLSMappingErr
	}
	if current.Revision != request.ExpectedRevision {
		return WorkloadMutationResult{}, ErrRevisionConflict
	}
	if request.Status != nil {
		if *request.Status != managementauth.ResourceActive && *request.Status != managementauth.ResourceDisabled {
			return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
		}
		if *request.Status == managementauth.ResourceActive && !service.mtlsListenerEnabled {
			return WorkloadMutationResult{}, ErrMTLSListenerUnavailable
		}
		current.Status = *request.Status
	}
	if request.WorkloadClass != nil {
		if !validWorkloadClass(*request.WorkloadClass) {
			return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
		}
		if *request.WorkloadClass == WorkloadStrong && !service.allowsStrongSource(ctx, request.Actor.Session, now) {
			return WorkloadMutationResult{}, managementauth.ErrAuthenticationDenied
		}
		current.WorkloadClass = *request.WorkloadClass
		current.SourceAssuredAt = now
	}
	barrierID := authSourceBarrierID(managementauth.AuthSourceMTLS, request.ID)
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return WorkloadMutationResult{}, err
	}
	result, patchMTLSMappingErr := service.repository.PatchMTLSMapping(ctx, current, request.ExpectedRevision, request.Actor.MutationActor())
	if patchMTLSMappingErr != nil {
		return WorkloadMutationResult{}, patchMTLSMappingErr
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	if current.Status == managementauth.ResourceActive {
		if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
			return result, err
		}
	}
	return result, nil
}

func (service *WorkloadIdentityService) DeleteMTLSMapping(ctx context.Context, request DeleteMTLSMappingRequest) (WorkloadMutationResult, error) {
	if service == nil || !canonicalUUID(request.ID) || request.ExpectedRevision == 0 || validateWorkloadActor(request.Actor) != nil {
		return WorkloadMutationResult{}, ErrInvalidWorkloadRequest
	}
	barrierID := authSourceBarrierID(managementauth.AuthSourceMTLS, request.ID)
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource, barrierID); err != nil {
		return WorkloadMutationResult{}, err
	}
	result, err := service.repository.DeleteMTLSMapping(ctx, request.ID, request.ExpectedRevision, request.Actor.MutationActor())
	if err != nil {
		return WorkloadMutationResult{}, err
	}
	if err := service.installSessionBarriers(ctx, result.SessionIDs); err != nil {
		return result, err
	}
	return result, nil
}

func (service *WorkloadIdentityService) ResolveMTLSIdentity(ctx context.Context, evidence managementauth.VerifiedMTLSEvidence, now time.Time) (managementauth.VerifiedMTLSIdentity, error) {
	if service == nil || !service.mtlsListenerEnabled || now.IsZero() || evidence.CertificateNotAfter.IsZero() ||
		validateMTLSMatcher(MTLSMatcherKind(evidence.MatcherKind), evidence.MatcherValue) != nil {
		return managementauth.VerifiedMTLSIdentity{}, managementauth.ErrAuthenticationDenied
	}
	resolved, err := service.repository.ResolveMTLSIdentity(ctx, evidence.MatcherKind, evidence.MatcherValue, now.UTC())
	if err != nil || resolved.MappingID == "" || resolved.PrincipalID == "" || resolved.SourceAssuredAt.After(now) ||
		!now.Before(evidence.CertificateNotAfter) {
		return managementauth.VerifiedMTLSIdentity{}, managementauth.ErrAuthenticationDenied
	}
	return managementauth.VerifiedMTLSIdentity{
		PrincipalID: resolved.PrincipalID, MappingID: resolved.MappingID,
		WorkloadClass: resolved.WorkloadClass, SourceAssuredAt: resolved.SourceAssuredAt.UTC(),
		EvidenceExpiresAt: evidence.CertificateNotAfter.UTC(),
	}, nil
}

func (service *WorkloadIdentityService) replayServiceCredential(ctx context.Context, command managementcommand.Command, stored StoredWorkloadSecret, now time.Time) (ServiceCredentialSecretResult, error) {
	if stored.Result.ResourceType != "service_account" || !now.Before(stored.Secret.ExpiresAt) {
		return ServiceCredentialSecretResult{}, ErrWorkloadSecretExpired
	}
	plaintext, err := service.responseKEK.Open(accesscredential.Envelope{
		KeyVersion: stored.Secret.KEKVersion, Nonce: stored.Secret.Nonce, Ciphertext: stored.Secret.Ciphertext,
	}, workloadResponseAAD(command.Endpoint, stored.Result.ResourceID, stored.Result.ResourceRevision))
	if err != nil {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	defer zeroWorkloadBytes(plaintext)
	var issued ServiceCredentialSecret
	if json.Unmarshal(plaintext, &issued) != nil || issued.ServiceAccount.ID != stored.Result.ResourceID ||
		issued.Credential.ServiceAccountID != issued.ServiceAccount.ID || issued.Secret == "" {
		return ServiceCredentialSecretResult{}, ErrWorkloadUnavailable
	}
	account, err := service.repository.GetServiceAccount(ctx, issued.ServiceAccount.ID)
	if err != nil || account.Status != ServiceAccountActive {
		return ServiceCredentialSecretResult{}, ErrServiceCredentialUnavailable
	}
	credential, err := service.repository.GetServiceCredential(ctx, account.ID, issued.Credential.ID)
	if err != nil || (credential.Status != ServiceCredentialActive && credential.Status != ServiceCredentialRetiring) ||
		now.Before(credential.NotBefore) || !now.Before(credential.ExpiresAt) {
		return ServiceCredentialSecretResult{}, ErrServiceCredentialUnavailable
	}
	return ServiceCredentialSecretResult{
		ServiceAccount: issued.ServiceAccount, Credential: credential, Secret: issued.Secret,
		DeliveryExpiry: issued.DeliveryExpiry, Replayed: true,
	}, nil
}

func (service *WorkloadIdentityService) bindCommand(owner ServiceAccountOwnerScope, namespaceID, principalID, endpoint, key string, body any, now time.Time) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidWorkloadRequest
	}
	scope := managementcommand.ClusterCommandScope()
	if owner == ServiceAccountOwnerNamespace {
		scope = managementcommand.NamespaceCommandScope(namespaceID)
	}
	command, err := service.commands.Bind(scope, principalID, endpoint, key, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		if errors.Is(err, managementcommand.ErrConflict) {
			return managementcommand.Command{}, err
		}
		return managementcommand.Command{}, ErrInvalidWorkloadRequest
	}
	return command, nil
}

func (service *WorkloadIdentityService) issueCredential(id, accountID string, class WorkloadClass, now, expiresAt time.Time) (ServiceCredential, string, []byte, string, error) {
	secret := make([]byte, 32)
	if err := service.randomBytes(secret); err != nil {
		return ServiceCredential{}, "", nil, "", ErrWorkloadUnavailable
	}
	defer zeroWorkloadBytes(secret)
	plaintext := "vsm_" + id + "_" + base64.RawURLEncoding.EncodeToString(secret)
	pepperVersion := service.peppers.ActiveVersion
	digest := ComputeServiceCredentialHMAC(service.peppers.Keys[pepperVersion], id, secret)
	credential := ServiceCredential{
		ID: id, ServiceAccountID: accountID, PublicID: id, WorkloadClass: class,
		SourceAssuredAt: now, Status: ServiceCredentialActive, NotBefore: now,
		ExpiresAt: expiresAt, CreatedAt: now,
	}
	return credential, plaintext, append([]byte(nil), digest[:]...), pepperVersion, nil
}

func (service *WorkloadIdentityService) allowsStrongSource(ctx context.Context, session managementauth.LiveSession, now time.Time) bool {
	policy, err := service.sessionPolicy.LoadSessionPolicy(ctx)
	if err != nil {
		return false
	}
	requirement, found := policy.ActionRequirements["cluster_sensitive"]
	return found && requirement.Allows(session, now)
}

func (service *WorkloadIdentityService) installSessionBarriers(ctx context.Context, sessionIDs []string) error {
	for _, sessionID := range sessionIDs {
		if !canonicalUUID(sessionID) {
			return ErrWorkloadUnavailable
		}
		if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementSession, sessionID); err != nil {
			return err
		}
	}
	return nil
}

func (service *WorkloadIdentityService) installSourceAndSessionBarriers(ctx context.Context, credentialIDs, sessionIDs []string) error {
	for _, credentialID := range credentialIDs {
		if err := service.barriers.InstallDeny(ctx, managementauth.BarrierAuthenticationSource,
			authSourceBarrierID(managementauth.AuthSourceServiceCredential, credentialID)); err != nil {
			return err
		}
	}
	return service.installSessionBarriers(ctx, sessionIDs)
}

func authSourceBarrierID(kind managementauth.AuthSourceKind, id string) string {
	return string(kind) + ":" + id
}

func (service *WorkloadIdentityService) threeIDs() (string, string, string, error) {
	first, err := service.nextID()
	if err != nil {
		return "", "", "", err
	}
	second, err := service.nextID()
	if err != nil {
		return "", "", "", err
	}
	third, err := service.nextID()
	return first, second, third, err
}

func (service *WorkloadIdentityService) nextID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", ErrWorkloadUnavailable
	}
	return value, nil
}

func (service *WorkloadIdentityService) timeNow() time.Time { return service.now().UTC() }
