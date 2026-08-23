package management

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	createEndpoint       = "/management/v1/provider-credentials"
	rotateEndpointPrefix = "/management/v1/provider-credentials/"
	defaultPageSize      = 50
	maximumPageSize      = 200
)

type Options struct {
	Repository      Repository
	Catalog         Catalog
	Egress          EgressPolicy
	CredentialCodec providercredential.Codec
	CommandCodec    *managementcommand.Codec
	CursorKeyring   securitykeyring.Symmetric
	IdempotencyTTL  time.Duration
	RetiringOverlap time.Duration
	Now             func() time.Time
	NewID           func() string
}

type Service struct {
	repository      Repository
	catalog         Catalog
	egress          EgressPolicy
	credentials     providercredential.Codec
	commands        *managementcommand.Codec
	cursors         cursorCodec
	idempotencyTTL  time.Duration
	retiringOverlap time.Duration
	now             func() time.Time
	newID           func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Catalog == nil || options.Egress == nil || options.CommandCodec == nil {
		return nil, fmt.Errorf("%w: repository, catalog, egress, and command dependencies are required", ErrUnavailable)
	}
	if err := options.CredentialCodec.Keyring.Validate(); err != nil {
		return nil, fmt.Errorf("%w: ProviderCredential keyring: %w", ErrUnavailable, err)
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	if options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour {
		return nil, fmt.Errorf("%w: idempotency TTL must be between 1m and 7d", ErrUnavailable)
	}
	if options.RetiringOverlap < time.Second || options.RetiringOverlap > 15*time.Minute {
		return nil, fmt.Errorf("%w: retiring overlap must be between 1s and 15m", ErrUnavailable)
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
		repository: options.Repository, catalog: options.Catalog, egress: options.Egress,
		credentials: options.CredentialCodec, commands: options.CommandCodec, cursors: cursors,
		idempotencyTTL: options.IdempotencyTTL, retiringOverlap: options.RetiringOverlap,
		now: now, newID: newID,
	}, nil
}

// Ready verifies that every unexpired durable command can still be replayed
// with a retained HMAC key version. Managed startup and readiness must fail
// closed when an operator removes a referenced version too early.
func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.ValidateManagementCommandHMACVersions(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

// Close erases the cursor-signing keys cloned for this service.
func (service *Service) Close() error {
	if service != nil {
		service.cursors.close()
	}
	return nil
}

func (service *Service) Create(ctx context.Context, request CreateRequest) (MutationResult, error) {
	if service == nil {
		return MutationResult{}, ErrUnavailable
	}
	if err := validateNamespaceActor(request.NamespaceID, request.Actor); err != nil ||
		providercredential.ValidateName(request.Name) != nil ||
		providercredential.ValidateProviderID(request.ProviderID) != nil || len(request.Secret) == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	command, createErr := service.createCommand(request, now)
	if createErr != nil {
		return MutationResult{}, createErr
	}
	if replay, found, err := service.repository.ReplayProviderCredentialCommand(ctx, command); err != nil {
		return MutationResult{}, err
	} else if found {
		return mutationResult(replay), nil
	}
	detail, createErr := service.catalog.Get(ctx, request.ProviderID)
	if createErr != nil {
		return MutationResult{}, createErr
	}
	origin, createErr := service.bindNewCredential(detail, request.ProviderID, request.BaseURL)
	if createErr != nil {
		return MutationResult{}, createErr
	}
	credentialID, versionID, createErr := service.newPair()
	if createErr != nil {
		return MutationResult{}, createErr
	}
	credential := providercredential.Credential{
		ID: credentialID, NamespaceID: request.NamespaceID, Name: request.Name,
		ProviderID:          detail.Provider.ID,
		CredentialMode:      providercredential.Mode(detail.Provider.Credential.Mode),
		CredentialAdapterID: detail.Provider.Credential.AdapterID,
		CatalogRevision:     detail.CatalogRevision, NormalizedOrigin: origin,
		Status: providercredential.StatusActive, ActiveVersionID: &versionID,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	secret := append([]byte(nil), request.Secret...)
	defer providercredential.Zero(secret)
	version, createErr := service.credentials.Seal(credential, versionID, secret, now)
	if createErr != nil {
		return MutationResult{}, fmt.Errorf("%w: seal ProviderCredential", ErrUnavailable)
	}
	result, createErr := service.repository.CreateProviderCredential(
		ctx, credential, version, command,
		mutationMeta(request.Actor, "provider_credential.create", "Create provider credential.", map[string]string{
			"providerId": detail.Provider.ID, "catalogRevision": detail.CatalogRevision,
			"normalizedOrigin": origin,
		}),
	)
	if createErr != nil {
		return MutationResult{}, createErr
	}
	return mutationResult(result), nil
}

func (service *Service) Get(ctx context.Context, namespaceID, credentialID string) (Metadata, error) {
	if service == nil {
		return Metadata{}, ErrUnavailable
	}
	if !canonicalUUID(namespaceID) || !canonicalUUID(credentialID) {
		return Metadata{}, ErrInvalidRequest
	}
	credential, err := service.repository.GetProviderCredential(
		ctx, accesscontrol.NamespaceID(namespaceID), credentialID,
	)
	if err != nil {
		return Metadata{}, err
	}
	return metadata(credential), nil
}

func (service *Service) List(ctx context.Context, request ListRequest) (ListResult, error) {
	if service == nil {
		return ListResult{}, ErrUnavailable
	}
	if !canonicalUUID(request.NamespaceID) ||
		(request.ProviderID != "" && providercredential.ValidateProviderID(request.ProviderID) != nil) ||
		(request.Status != "" && !validStatus(request.Status)) {
		return ListResult{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return ListResult{}, ErrInvalidRequest
	}
	scopeDigest, listErr := request.Scope.Digest()
	if listErr != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return ListResult{}, ErrInvalidRequest
	}
	query := accesspostgres.ProviderCredentialListRequest{
		ProviderID: request.ProviderID, Status: request.Status, PageSize: pageSize, Scope: request.Scope,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.NamespaceID != request.NamespaceID ||
			cursor.ProviderID != request.ProviderID || cursor.Status != request.Status ||
			cursor.ScopeDigest != scopeDigest ||
			!validStatus(cursor.AfterStatus) || !canonicalUUID(cursor.AfterID) {
			return ListResult{}, ErrInvalidRequest
		}
		query.AfterStatus, query.AfterID = cursor.AfterStatus, cursor.AfterID
	}
	if !request.Scope.All && len(request.Scope.IDs(accesscontrol.ScopeResourceProviderCredential)) == 0 {
		return ListResult{Credentials: []Metadata{}, PageSize: pageSize}, nil
	}
	page, listErr := service.repository.ListProviderCredentials(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), query,
	)
	if listErr != nil {
		return ListResult{}, listErr
	}
	result := ListResult{
		Credentials: make([]Metadata, len(page.Credentials)), HasMore: page.HasMore, PageSize: pageSize,
	}
	for index := range page.Credentials {
		result.Credentials[index] = metadata(page.Credentials[index])
	}
	if page.HasMore {
		if len(page.Credentials) == 0 {
			return ListResult{}, fmt.Errorf("%w: repository returned an empty continued page", ErrUnavailable)
		}
		last := page.Credentials[len(page.Credentials)-1]
		result.NextCursor, listErr = service.cursors.encode(listCursor{
			Version: 1, NamespaceID: request.NamespaceID, ProviderID: request.ProviderID,
			Status: request.Status, AfterStatus: last.Status, AfterID: last.ID, ScopeDigest: scopeDigest,
		})
		if listErr != nil {
			return ListResult{}, listErr
		}
	}
	return result, nil
}

func (service *Service) Rename(ctx context.Context, request RenameRequest) (MutationResult, error) {
	if err := service.validateLifecycle(request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Actor); err != nil ||
		providercredential.ValidateName(request.Name) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	result, err := service.repository.RenameProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
		accesscontrol.Revision(request.ExpectedRevision), request.Name,
		mutationMeta(request.Actor, "provider_credential.rename", "Rename provider credential.", nil),
	)
	if err != nil {
		return MutationResult{}, err
	}
	return mutationResult(result), nil
}

func (service *Service) Rotate(ctx context.Context, request RotateRequest) (MutationResult, error) {
	if err := service.validateLifecycle(request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Actor); err != nil || len(request.Secret) == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	command, rotateErr := service.rotateCommand(request, now)
	if rotateErr != nil {
		return MutationResult{}, rotateErr
	}
	if replay, found, err := service.repository.ReplayProviderCredentialCommand(ctx, command); err != nil {
		return MutationResult{}, err
	} else if found {
		return mutationResult(replay), nil
	}
	credential, rotateErr := service.repository.GetProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
	)
	if rotateErr != nil {
		return MutationResult{}, rotateErr
	}
	if credential.Status != providercredential.StatusActive || credential.ActiveVersionID == nil {
		return MutationResult{}, providercredential.ErrUnavailable
	}
	if err := service.validateCurrentBinding(ctx, credential); err != nil {
		return MutationResult{}, err
	}
	versionID, rotateErr := service.newUUID()
	if rotateErr != nil {
		return MutationResult{}, rotateErr
	}
	secret := append([]byte(nil), request.Secret...)
	defer providercredential.Zero(secret)
	version, rotateErr := service.credentials.Seal(credential, versionID, secret, now)
	if rotateErr != nil {
		return MutationResult{}, fmt.Errorf("%w: seal ProviderCredential", ErrUnavailable)
	}
	result, rotateErr := service.repository.RotateProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), credential.ID,
		accesscontrol.Revision(request.ExpectedRevision), accesspostgres.ProviderCredentialRotation{
			Version: version, PreviousVersionID: *credential.ActiveVersionID,
			RetireAt: now.Add(service.retiringOverlap),
		}, command, mutationMeta(request.Actor, "provider_credential.rotate", "Rotate provider credential.", nil),
	)
	if rotateErr != nil {
		return MutationResult{}, rotateErr
	}
	return mutationResult(result), nil
}

func (service *Service) Disable(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	if err := service.validateLifecycle(request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Actor); err != nil || len(request.Secret) != 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	result, err := service.repository.DisableProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
		accesscontrol.Revision(request.ExpectedRevision),
		mutationMeta(request.Actor, "provider_credential.disable", "Disable provider credential.", nil),
	)
	if err != nil {
		return MutationResult{}, err
	}
	return mutationResult(result), nil
}

func (service *Service) Reactivate(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	if err := service.validateLifecycle(request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Actor); err != nil || len(request.Secret) == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	credential, reactivateErr := service.repository.GetProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
	)
	if reactivateErr != nil {
		return MutationResult{}, reactivateErr
	}
	if credential.Status != providercredential.StatusDisabled {
		return MutationResult{}, providercredential.ErrUnavailable
	}
	if err := service.validateCurrentBinding(ctx, credential); err != nil {
		return MutationResult{}, err
	}
	now := service.now().UTC()
	versionID, reactivateErr := service.newUUID()
	if reactivateErr != nil {
		return MutationResult{}, reactivateErr
	}
	secret := append([]byte(nil), request.Secret...)
	defer providercredential.Zero(secret)
	version, reactivateErr := service.credentials.Seal(credential, versionID, secret, now)
	if reactivateErr != nil {
		return MutationResult{}, fmt.Errorf("%w: seal ProviderCredential", ErrUnavailable)
	}
	result, reactivateErr := service.repository.ReactivateProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
		accesscontrol.Revision(request.ExpectedRevision), version,
		mutationMeta(request.Actor, "provider_credential.reactivate", "Reactivate provider credential.", nil),
	)
	if reactivateErr != nil {
		return MutationResult{}, reactivateErr
	}
	return mutationResult(result), nil
}

func (service *Service) Delete(ctx context.Context, request LifecycleRequest) (MutationResult, error) {
	if err := service.validateLifecycle(request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Actor); err != nil || len(request.Secret) != 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	result, err := service.repository.DeleteProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
		accesscontrol.Revision(request.ExpectedRevision),
		mutationMeta(request.Actor, "provider_credential.delete", "Delete provider credential.", nil),
	)
	if err != nil {
		return MutationResult{}, err
	}
	return mutationResult(result), nil
}

func (service *Service) bindNewCredential(detail providercatalog.DetailResult, requestedProviderID, requestedOrigin string) (string, error) {
	provider := detail.Provider
	if detail.CatalogRevision == "" || provider.ID != requestedProviderID || provider.Credential.Mode == providercatalog.CredentialNone ||
		provider.Credential.AdapterID == "" {
		return "", ErrProviderMismatch
	}
	var origin string
	switch provider.Origin.Mode {
	case providercatalog.OriginFixed:
		if requestedOrigin != "" {
			return "", fmt.Errorf("%w: fixed Provider origin cannot be overridden", ErrInvalidRequest)
		}
		normalized, err := providercredential.NormalizeOrigin(provider.Origin.DefaultURL)
		if err != nil || normalized != provider.Origin.DefaultURL {
			return "", ErrProviderMismatch
		}
		origin = normalized
	case providercatalog.OriginUserSupplied:
		normalized, err := providercredential.NormalizeOrigin(requestedOrigin)
		if err != nil {
			return "", fmt.Errorf("%w: base URL is invalid", ErrInvalidRequest)
		}
		origin = normalized
	default:
		return "", fmt.Errorf("%w: Provider origin mode is invalid", ErrInvalidRequest)
	}
	if _, err := service.egress.AuthorizeOrigin(origin); err != nil {
		return "", ErrUnsafeOrigin
	}
	return origin, nil
}

func (service *Service) validateCurrentBinding(ctx context.Context, credential providercredential.Credential) error {
	detail, err := service.catalog.Get(ctx, credential.ProviderID)
	if err != nil {
		return err
	}
	provider := detail.Provider
	if provider.ID != credential.ProviderID || provider.Credential.Mode == providercatalog.CredentialNone ||
		providercredential.Mode(provider.Credential.Mode) != credential.CredentialMode ||
		provider.Credential.AdapterID != credential.CredentialAdapterID {
		return ErrProviderMismatch
	}
	if provider.Origin.Mode == providercatalog.OriginFixed &&
		provider.Origin.DefaultURL != credential.NormalizedOrigin {
		return ErrProviderMismatch
	}
	if provider.Origin.Mode != providercatalog.OriginFixed && provider.Origin.Mode != providercatalog.OriginUserSupplied {
		return ErrProviderMismatch
	}
	if _, err := service.egress.AuthorizeOrigin(credential.NormalizedOrigin); err != nil {
		return ErrUnsafeOrigin
	}
	return nil
}

func (service *Service) createCommand(request CreateRequest, now time.Time) (managementcommand.Command, error) {
	canonical, err := json.Marshal(struct {
		NamespaceID string `json:"namespaceId"`
		Name        string `json:"name"`
		ProviderID  string `json:"providerId"`
		BaseURL     string `json:"baseUrl"`
		Secret      []byte `json:"secret"`
	}{request.NamespaceID, request.Name, request.ProviderID, request.BaseURL, request.Secret})
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	defer providercredential.Zero(canonical)
	command, err := service.commands.Bind(
		managementcommand.NamespaceCommandScope(request.NamespaceID), request.Actor.PrincipalID, createEndpoint,
		request.IdempotencyKey, canonical, now, now.Add(service.idempotencyTTL),
	)
	if err != nil {
		return managementcommand.Command{}, fmt.Errorf("%w: idempotency identity is invalid", ErrInvalidRequest)
	}
	return command, nil
}

func (service *Service) rotateCommand(request RotateRequest, now time.Time) (managementcommand.Command, error) {
	canonical, err := json.Marshal(struct {
		NamespaceID      string `json:"namespaceId"`
		CredentialID     string `json:"credentialId"`
		ExpectedRevision uint64 `json:"expectedRevision"`
		Secret           []byte `json:"secret"`
	}{request.NamespaceID, request.CredentialID, request.ExpectedRevision, request.Secret})
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	defer providercredential.Zero(canonical)
	command, err := service.commands.Bind(
		managementcommand.NamespaceCommandScope(request.NamespaceID), request.Actor.PrincipalID,
		rotateEndpointPrefix+request.CredentialID+":rotate",
		request.IdempotencyKey, canonical, now, now.Add(service.idempotencyTTL),
	)
	if err != nil {
		return managementcommand.Command{}, fmt.Errorf("%w: idempotency identity is invalid", ErrInvalidRequest)
	}
	return command, nil
}

func (service *Service) validateLifecycle(namespaceID, credentialID string, revision uint64, actor Actor) error {
	if service == nil {
		return ErrUnavailable
	}
	if !canonicalUUID(namespaceID) || !canonicalUUID(credentialID) || revision == 0 {
		return ErrInvalidRequest
	}
	return validateNamespaceActor(namespaceID, actor)
}

func (service *Service) newPair() (string, string, error) {
	credentialID, err := service.newUUID()
	if err != nil {
		return "", "", err
	}
	versionID, err := service.newUUID()
	return credentialID, versionID, err
}

func (service *Service) newUUID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", fmt.Errorf("%w: ID generator returned an invalid UUID", ErrUnavailable)
	}
	return value, nil
}

func validateNamespaceActor(namespaceID string, actor Actor) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) || actor.RequestID == "" {
		return ErrInvalidRequest
	}
	for _, principalID := range actor.ActorChain {
		if !canonicalUUID(principalID) {
			return ErrInvalidRequest
		}
	}
	if actor.SourceIP.IsValid() && actor.SourceIP != actor.SourceIP.Unmap() {
		return ErrInvalidRequest
	}
	return nil
}

func mutationMeta(actor Actor, action, reason string, details map[string]string) accesspostgres.MutationMeta {
	principalID := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(actor.ActorChain[index])
	}
	return accesspostgres.MutationMeta{
		ActorPrincipalID: &principalID, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: action, Reason: reason, Details: details,
	}
}

func mutationResult(result accesspostgres.MutationResult[providercredential.Credential]) MutationResult {
	credentialID := result.ResourceID
	if credentialID == "" {
		credentialID = result.Value.ID
	}
	revision := uint64(result.ResourceRevision)
	if revision == 0 {
		revision = result.Value.Revision
	}
	return MutationResult{
		CredentialID: credentialID, Revision: revision, Replayed: result.Replayed,
	}
}

func metadata(credential providercredential.Credential) Metadata {
	result := Metadata{
		CredentialID: credential.ID, NamespaceID: credential.NamespaceID, Name: credential.Name,
		ProviderID: credential.ProviderID, CatalogRevision: credential.CatalogRevision,
		NormalizedOrigin: credential.NormalizedOrigin, Status: credential.Status,
		Revision: credential.Revision, CreatedAt: credential.CreatedAt,
		UpdatedAt: credential.UpdatedAt,
	}
	if credential.DeletedAt != nil {
		value := *credential.DeletedAt
		result.DeletedAt = &value
	}
	return result
}

func validStatus(status providercredential.Status) bool {
	return status == providercredential.StatusActive || status == providercredential.StatusDisabled ||
		status == providercredential.StatusDeleted
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}
