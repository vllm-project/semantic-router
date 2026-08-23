package delegationmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize               = 50
	maximumPageSize               = 200
	delegationCreateEndpoint      = "/management/v1/self/inference-sessions"
	delegationAdminEndpointPrefix = "/management/v1/api-keys/"
	delegatedCredentialSecretKind = "delegated_inference_credential"
)

// secretEnvelope is the encrypted, byte-stable one-time response record. The
// transport maps the same fields to its public SecretEnvelope contract; this
// domain package deliberately does not import the Management wire registry.
type secretEnvelope struct {
	ResourceID string     `json:"resourceId"`
	Kind       string     `json:"kind"`
	Secret     string     `json:"secret"`
	ExpiresAt  *time.Time `json:"expiresAt,omitempty"`
}

type Options struct {
	Repository        Repository
	Waiter            PublicationWaiter
	Commands          *managementcommand.Codec
	CursorKeyring     securitykeyring.Symmetric
	DelegationPeppers accesscredential.PepperKeyring
	ResponseKEK       accesscredential.KEKKeyring
	Audience          string
	IdempotencyTTL    time.Duration
	SecretDeliveryTTL time.Duration
	Now               func() time.Time
	NewID             func() string
}

type Service struct {
	repository     Repository
	waiter         PublicationWaiter
	commands       *managementcommand.Codec
	cursors        cursorCodec
	peppers        accesscredential.PepperKeyring
	responseKEK    accesscredential.KEKKeyring
	audience       string
	idempotencyTTL time.Duration
	secretTTL      time.Duration
	now            func() time.Time
	newID          func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Waiter == nil || options.Commands == nil ||
		options.DelegationPeppers.Validate() != nil || options.ResponseKEK.Validate() != nil ||
		strings.TrimSpace(options.Audience) == "" || options.Audience != strings.TrimSpace(options.Audience) ||
		options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour ||
		options.SecretDeliveryTTL < time.Minute || options.SecretDeliveryTTL > options.IdempotencyTTL {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, err
	}
	now, newID := options.Now, options.NewID
	if now == nil {
		now = time.Now
	}
	if newID == nil {
		newID = uuid.NewString
	}
	return &Service{
		repository: options.Repository, waiter: options.Waiter, commands: options.Commands,
		cursors: cursors, peppers: options.DelegationPeppers.Clone(), responseKEK: options.ResponseKEK.Clone(),
		audience: options.Audience, idempotencyTTL: options.IdempotencyTTL,
		secretTTL: options.SecretDeliveryTTL, now: now, newID: newID,
	}, nil
}

func (service *Service) Close() {
	if service == nil {
		return
	}
	service.cursors.close()
	service.peppers.Close()
	service.responseKEK.Close()
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) ResolveSelf(
	ctx context.Context, namespaceID, principalID, managementSessionID string,
) (SelfContext, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(principalID) || !canonicalUUID(managementSessionID) {
		return SelfContext{}, ErrInvalidRequest
	}
	return service.repository.ResolveSelf(ctx, namespaceID, principalID, managementSessionID, false)
}

func (service *Service) GetSession(ctx context.Context, namespaceID, sessionID string) (Session, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(sessionID) {
		return Session{}, ErrInvalidRequest
	}
	return service.repository.GetSession(ctx, namespaceID, sessionID)
}

func (service *Service) ListEligibleKeys(ctx context.Context, request ListRequest) (ResultPage[EligibleKey], error) {
	if err := validateList(request); err != nil || request.PrincipalID == "" || request.ManagementSessionID == "" {
		return ResultPage[EligibleKey]{}, ErrInvalidRequest
	}
	if _, err := service.repository.ResolveSelf(ctx, request.NamespaceID, request.PrincipalID, request.ManagementSessionID, false); err != nil {
		return ResultPage[EligibleKey]{}, err
	}
	query := EligibleKeyQuery{
		NamespaceID: request.NamespaceID, PrincipalID: request.PrincipalID,
		ManagementSessionID: request.ManagementSessionID, Limit: pageLimit(request.PageSize),
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "eligible_keys" || cursor.NamespaceID != request.NamespaceID ||
			cursor.PrincipalID != request.PrincipalID || cursor.APIKeyID != "" {
			return ResultPage[EligibleKey]{}, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	page, err := service.repository.ListEligibleKeys(ctx, query)
	if err != nil {
		return ResultPage[EligibleKey]{}, err
	}
	result := ResultPage[EligibleKey]{Items: page.Items, HasMore: page.HasMore, PageSize: query.Limit}
	if page.HasMore {
		if len(page.Items) == 0 {
			return ResultPage[EligibleKey]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.cursors.encode(cursorPayload{
			Kind: "eligible_keys", NamespaceID: request.NamespaceID,
			PrincipalID: request.PrincipalID, CreatedAt: last.CreatedAt, ID: last.KeyID,
		})
	}
	return result, err
}

func (service *Service) ListSessions(ctx context.Context, request ListRequest) (ResultPage[Session], error) {
	if err := validateList(request); err != nil || (request.PrincipalID == "" && request.APIKeyID == "") {
		return ResultPage[Session]{}, ErrInvalidRequest
	}
	query := SessionQuery{
		NamespaceID: request.NamespaceID, PrincipalID: request.PrincipalID,
		APIKeyID: request.APIKeyID, Limit: pageLimit(request.PageSize),
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "sessions" || cursor.NamespaceID != request.NamespaceID ||
			cursor.PrincipalID != request.PrincipalID || cursor.APIKeyID != request.APIKeyID {
			return ResultPage[Session]{}, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	page, err := service.repository.ListSessions(ctx, query)
	if err != nil {
		return ResultPage[Session]{}, err
	}
	result := ResultPage[Session]{Items: page.Items, HasMore: page.HasMore, PageSize: query.Limit}
	if page.HasMore {
		if len(page.Items) == 0 {
			return ResultPage[Session]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.cursors.encode(cursorPayload{
			Kind: "sessions", NamespaceID: request.NamespaceID,
			PrincipalID: request.PrincipalID, APIKeyID: request.APIKeyID, CreatedAt: last.CreatedAt, ID: last.ID,
		})
	}
	return result, err
}

func (service *Service) GetKey(ctx context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(keyID) {
		return accesscontrol.APIKey{}, ErrInvalidRequest
	}
	return service.repository.GetKey(ctx, namespaceID, keyID)
}

func (service *Service) Create(ctx context.Context, request CreateRequest) (SecretResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.KeyID) ||
		validateActor(request.Actor) != nil {
		return SecretResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	canonical := struct {
		KeyID string `json:"keyId"`
	}{request.KeyID}
	body, _ := json.Marshal(canonical)
	command, createErr := service.commands.Bind(managementcommand.NamespaceCommandScope(request.NamespaceID),
		request.Actor.PrincipalID, delegationCreateEndpoint, request.IdempotencyKey,
		body, now, now.Add(service.idempotencyTTL))
	if createErr != nil {
		return SecretResult{}, mapCommandError(createErr)
	}
	if stored, found, err := service.repository.ReplaySecret(ctx, command); err != nil || found {
		if err != nil {
			return SecretResult{}, err
		}
		return service.replay(ctx, command, stored, request.Actor, request.KeyID, now)
	}
	self, createErr := service.repository.ResolveSelf(ctx, request.NamespaceID, request.Actor.PrincipalID,
		request.Actor.ManagementSessionID, true)
	if createErr != nil {
		return SecretResult{}, createErr
	}
	key, createErr := service.repository.GetEligibleKey(ctx, request.NamespaceID, request.Actor.PrincipalID,
		request.Actor.ManagementSessionID, request.KeyID)
	if createErr != nil {
		return SecretResult{}, createErr
	}
	sessionID := service.newID()
	if !canonicalUUID(sessionID) {
		return SecretResult{}, ErrUnavailable
	}
	issued, createErr := service.peppers.Issue(accesscredential.KindDelegation, sessionID)
	if createErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	expiresAt := now.Add(self.Policy.DelegatedSessionTTL)
	if expiresAt.After(self.ManagementSessionExpires) {
		expiresAt = self.ManagementSessionExpires
	}
	if key.ExpiresAt != nil && expiresAt.After(*key.ExpiresAt) {
		expiresAt = key.ExpiresAt.UTC()
	}
	if !expiresAt.After(now) {
		return SecretResult{}, ErrNotEligible
	}
	session := Session{
		ID: sessionID, PublicID: sessionID, NamespaceID: request.NamespaceID,
		QuotaPartition: self.QuotaPartition, ManagementSessionID: request.Actor.ManagementSessionID,
		PrincipalID: request.Actor.PrincipalID, APIKeyID: request.KeyID, DelegationEpoch: key.DelegationEpoch,
		UserID: self.UserID, TeamID: key.TeamID, TokenHMAC: append([]byte(nil), issued.Digest.HMAC...),
		PepperVersion: issued.Digest.PepperVersion, Audience: service.audience, Status: SessionActive,
		NotBefore: now, ExpiresAt: expiresAt, Revision: 1, CreatedAt: now,
	}
	secretBody, createErr := json.Marshal(secretEnvelope{
		ResourceID: sessionID,
		Kind:       delegatedCredentialSecretKind, Secret: issued.Plaintext, ExpiresAt: &expiresAt,
	})
	if createErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	deliveryExpiry := now.Add(service.secretTTL)
	if deliveryExpiry.After(expiresAt) {
		deliveryExpiry = expiresAt
	}
	envelope, createErr := service.responseKEK.Seal(secretBody, responseAAD(command.Endpoint, request.NamespaceID, sessionID, 1))
	if createErr != nil {
		return SecretResult{}, ErrUnavailable
	}
	created, createErr := service.repository.Create(ctx, CreateMutation{
		Session: session, Command: command,
		Response: envelope, ResponseExpiresAt: deliveryExpiry, Actor: request.Actor,
	})
	if createErr != nil {
		return SecretResult{}, createErr
	}
	clear(session.TokenHMAC)
	if created.Replayed {
		if created.Stored == nil {
			return SecretResult{}, ErrUnavailable
		}
		return service.replay(ctx, command, *created.Stored, request.Actor, request.KeyID, service.now().UTC())
	}
	if err := service.waiter.WaitActive(ctx, created.Session, created.DesiredRevision); err != nil {
		return SecretResult{}, fmt.Errorf("%w: delegated credential publication: %w", ErrUnavailable, err)
	}
	return SecretResult{Session: created.Session, CanonicalJSON: secretBody}, nil
}

func (service *Service) Revoke(ctx context.Context, request RevokeRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.SessionID) ||
		validateActor(request.Actor) != nil || (request.PrincipalID == "" && request.APIKeyID == "") {
		return MutationResult{}, ErrInvalidRequest
	}
	result, err := service.repository.Revoke(ctx, request)
	if err != nil {
		return MutationResult{}, err
	}
	if err := service.waiter.WaitApplied(ctx, result.Session.NamespaceID, result.Session.QuotaPartition, result.DesiredRevision); err != nil {
		return MutationResult{}, fmt.Errorf("%w: delegated credential revocation publication: %w", ErrUnavailable, err)
	}
	return result, nil
}

func (service *Service) RevokeAll(ctx context.Context, request RevokeAllRequest) (RevokeAllResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.KeyID) || validateActor(request.Actor) != nil {
		return RevokeAllResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	canonical := struct {
		KeyID string `json:"keyId"`
	}{request.KeyID}
	body, _ := json.Marshal(canonical)
	endpoint := delegationAdminEndpointPrefix + request.KeyID + "/inference-sessions:revoke-all"
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(request.NamespaceID),
		request.Actor.PrincipalID, endpoint, request.IdempotencyKey, body, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return RevokeAllResult{}, mapCommandError(err)
	}
	result, err := service.repository.RevokeAll(ctx, RevokeAllMutation{
		NamespaceID: request.NamespaceID,
		KeyID:       request.KeyID, Command: command, Actor: request.Actor,
	})
	if err != nil {
		return RevokeAllResult{}, err
	}
	if result.QuotaPartition == "" {
		return RevokeAllResult{}, ErrUnavailable
	}
	if err := service.waiter.WaitApplied(ctx, request.NamespaceID, result.QuotaPartition, result.DesiredRevision); err != nil {
		return RevokeAllResult{}, fmt.Errorf("%w: delegated credential revoke-all publication: %w", ErrUnavailable, err)
	}
	return result, nil
}

func (service *Service) replay(
	ctx context.Context,
	command managementcommand.Command,
	stored StoredSecret,
	actor Actor,
	keyID string,
	now time.Time,
) (SecretResult, error) {
	if stored.Result.ResourceType != "delegated_inference_session" ||
		!canonicalUUID(stored.Result.ResourceID) || stored.Result.ResourceRevision != 1 ||
		stored.Result.ResponseStatus != 201 || stored.DesiredRevision == 0 {
		return SecretResult{}, ErrUnavailable
	}
	if !now.Before(stored.Secret.ExpiresAt) {
		return SecretResult{}, ErrSecretResultExpired
	}
	session, err := service.repository.GetSession(ctx, command.Scope.NamespaceID, stored.Result.ResourceID)
	if err != nil {
		return SecretResult{}, err
	}
	if session.Status != SessionActive || session.NotBefore.After(now) || !now.Before(session.ExpiresAt) ||
		session.PrincipalID != command.PrincipalID || session.PrincipalID != actor.PrincipalID ||
		session.ManagementSessionID != actor.ManagementSessionID || session.APIKeyID != keyID ||
		session.Audience != service.audience || session.PublicID != session.ID {
		return SecretResult{}, ErrCredentialInactive
	}
	plaintext, err := service.responseKEK.Open(accesscredential.Envelope{
		KeyVersion: stored.Secret.KEKVersion,
		Nonce:      stored.Secret.Nonce, Ciphertext: stored.Secret.Ciphertext,
	},
		responseAAD(command.Endpoint, command.Scope.NamespaceID, session.ID, stored.Result.ResourceRevision))
	if err != nil {
		return SecretResult{}, ErrUnavailable
	}
	defer clear(plaintext)
	var envelope secretEnvelope
	if json.Unmarshal(plaintext, &envelope) != nil || envelope.ResourceID != session.ID ||
		envelope.Kind != delegatedCredentialSecretKind || envelope.Secret == "" ||
		envelope.ExpiresAt == nil || !envelope.ExpiresAt.Equal(session.ExpiresAt) {
		return SecretResult{}, ErrUnavailable
	}
	if err := service.waiter.WaitActive(ctx, session, stored.DesiredRevision); err != nil {
		return SecretResult{}, fmt.Errorf("%w: delegated credential replay publication: %w", ErrUnavailable, err)
	}
	return SecretResult{Session: session, CanonicalJSON: append([]byte(nil), plaintext...), Replayed: true}, nil
}

func responseAAD(endpoint, namespaceID, sessionID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/delegated-inference-secret-response/v1\x00%s\x00%s\x00%s\x00%d", endpoint, namespaceID, sessionID, revision))
}

func validateList(request ListRequest) error {
	if !canonicalUUID(request.NamespaceID) || request.PageSize < 0 || request.PageSize > maximumPageSize {
		return ErrInvalidRequest
	}
	return nil
}

func pageLimit(value int) int {
	if value == 0 {
		return defaultPageSize
	}
	return value
}

func validateActor(actor Actor) error {
	if !canonicalUUID(actor.PrincipalID) || !canonicalUUID(actor.ManagementSessionID) || actor.RequestID == "" {
		return ErrInvalidRequest
	}
	return nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func mapCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return err
	}
	return ErrInvalidRequest
}
