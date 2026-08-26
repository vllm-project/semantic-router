package apikeymanagement

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func (service *Service) setStatus(ctx context.Context, request LifecycleRequest, status accesscontrol.APIKeyStatus, action string) (MutationResult, error) {
	if err := service.validateMutation(request.NamespaceID, request.KeyID, request.ExpectedRevision, request.Actor); err != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.timeNow()
	actionName := strings.TrimPrefix(action, "api_key.")
	command, setStatusErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/api-keys/"+request.KeyID+":"+actionName, request.IdempotencyKey,
		struct {
			Status accesscontrol.APIKeyStatus `json:"status"`
		}{status}, now)
	if setStatusErr != nil {
		return MutationResult{}, setStatusErr
	}
	if replay, found, err := service.repository.ReplayMutation(ctx, command); err != nil || found {
		return replay, err
	}
	key, setStatusErr := service.currentAtRevision(ctx, request.NamespaceID, request.KeyID, request.ExpectedRevision)
	if setStatusErr != nil {
		return MutationResult{}, setStatusErr
	}
	if status == accesscontrol.APIKeyStatusActive && key.ExpiresAt != nil && !key.ExpiresAt.After(service.timeNow()) {
		return MutationResult{}, ErrCredentialUnavailable
	}
	key.Status = status
	key.PolicyEpoch++
	key.DelegationEpoch++
	return service.repository.UpdateKeyAction(ctx, UpdateMutation{
		Key: key, ExpectedRevision: request.ExpectedRevision,
		Command: command, Actor: request.Actor, Action: action, Reason: "Change API-key lifecycle.",
	})
}

func (service *Service) currentAtRevision(ctx context.Context, namespaceID, keyID string, revision uint64) (accesscontrol.APIKey, error) {
	key, err := service.repository.GetKey(ctx, namespaceID, keyID)
	if err != nil {
		return accesscontrol.APIKey{}, err
	}
	if uint64(key.Revision) != revision {
		return accesscontrol.APIKey{}, ErrRevisionConflict
	}
	return key, nil
}

func (service *Service) validateMutation(namespaceID, keyID string, revision uint64, actor Actor) error {
	if service == nil || validateActor(namespaceID, actor) != nil || !canonicalUUID(keyID) || revision == 0 {
		return ErrInvalidRequest
	}
	return nil
}

func (service *Service) issueCredential(namespaceID, keyID, credentialID string, expiresAt *time.Time, revealable bool, now time.Time) (accesscontrol.CredentialVersion, string, error) {
	issued, err := service.peppers.Issue(accesscredential.KindAPIKey, credentialID)
	if err != nil {
		return accesscontrol.CredentialVersion{}, "", ErrUnavailable
	}
	credential := accesscontrol.CredentialVersion{
		ID:       accesscontrol.CredentialVersionID(credentialID),
		APIKeyID: accesscontrol.APIKeyID(keyID), KID: issued.Digest.PublicID,
		SecretHMAC: append([]byte(nil), issued.Digest.HMAC...), PepperVersion: issued.Digest.PepperVersion,
		Status: accesscontrol.CredentialStatusActive, NotBefore: now, ExpiresAt: canonicalTime(expiresAt), CreatedAt: now,
	}
	if revealable {
		envelope, err := service.revealKEK.Seal([]byte(issued.Plaintext), revealAAD(namespaceID, keyID, credentialID, credential.KID))
		if err != nil {
			return accesscontrol.CredentialVersion{}, "", ErrUnavailable
		}
		credential.SecretCiphertext, credential.CiphertextNonce, credential.KEKVersion = envelope.Ciphertext, envelope.Nonce, envelope.KeyVersion
	}
	return credential, issued.Plaintext, nil
}

func (service *Service) compileAccessBindings(
	namespaceID, keyID string,
	policyIDs []string,
	now time.Time,
) ([]policymanagement.AccessPolicyBinding, []PolicyBindingReceipt, error) {
	bindings := make([]policymanagement.AccessPolicyBinding, 0, len(policyIDs))
	receipts := make([]PolicyBindingReceipt, 0, len(policyIDs))
	for _, policyID := range policyIDs {
		bindingID, err := service.nextID()
		if err != nil {
			return nil, nil, err
		}
		bindings = append(bindings, policymanagement.AccessPolicyBinding{
			ID: bindingID, NamespaceID: namespaceID, PolicyID: policyID,
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: keyID},
			Status:  accesscontrol.BindingStatusActive, Revision: 1,
			CreatedAt: now, UpdatedAt: now,
		})
		receipts = append(receipts, PolicyBindingReceipt{PolicyID: policyID, BindingID: bindingID})
	}
	return bindings, receipts, nil
}

func (service *Service) compileRateLimitOverride(
	namespaceID, keyID string,
	input *RateLimitOverrideInput,
	now time.Time,
) (*RateLimitOverrideMutation, *RateLimitOverrideReceipt, error) {
	if input == nil {
		return nil, nil, nil
	}
	bindingID, err := service.nextID()
	if err != nil {
		return nil, nil, err
	}
	mutation := &RateLimitOverrideMutation{PolicyID: input.PolicyID}
	resolvedPolicyID := input.PolicyID
	created := false
	if input.InlinePolicy != nil {
		policyID, err := service.nextID()
		if err != nil {
			return nil, nil, err
		}
		var idError error
		compiled, err := policymanagement.CompileInlineRateLimitPolicy(policymanagement.InlineRateLimitPolicySpec{
			NamespaceID: namespaceID, PolicyID: policyID,
			Name: input.InlinePolicy.Name, Description: input.InlinePolicy.Description,
			Rules: input.InlinePolicy.Rules, Now: now,
			NewRuleID: func() string {
				if idError != nil {
					return ""
				}
				var value string
				value, idError = service.nextID()
				return value
			},
		})
		if idError != nil {
			return nil, nil, idError
		}
		if err != nil {
			return nil, nil, ErrInvalidRequest
		}
		mutation.InlinePolicy = &compiled
		resolvedPolicyID = policyID
		created = true
	}
	mutation.Binding = policymanagement.RateLimitBinding{
		ID: bindingID, NamespaceID: namespaceID, PolicyID: resolvedPolicyID,
		Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: keyID},
		Mode:    accesscontrol.RateBindingAllocation, Status: accesscontrol.BindingStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	return mutation, &RateLimitOverrideReceipt{
		PolicyID: resolvedPolicyID, BindingID: bindingID, Created: created,
	}, nil
}

func (service *Service) replaySecret(ctx context.Context, command managementcommand.Command, stored StoredSecret, now time.Time) (SecretMutationResult, error) {
	if !now.Before(stored.Secret.ExpiresAt) {
		return SecretMutationResult{}, ErrSecretResultExpired
	}
	plaintext, replaySecretErr := service.responseKEK.Open(accesscredential.Envelope{
		KeyVersion: stored.Secret.KEKVersion,
		Nonce:      stored.Secret.Nonce, Ciphertext: stored.Secret.Ciphertext,
	},
		responseAAD(command.Endpoint, command.Scope.NamespaceID, stored.Result.ResourceID, stored.Result.ResourceRevision))
	if replaySecretErr != nil {
		return SecretMutationResult{}, ErrUnavailable
	}
	defer zero(plaintext)
	var issued IssuedSecret
	if err := json.Unmarshal(plaintext, &issued); err != nil || issued.Data.ID != stored.Result.ResourceID || issued.Secret == "" {
		return SecretMutationResult{}, ErrUnavailable
	}
	key, replaySecretErr := service.repository.GetKey(ctx, command.Scope.NamespaceID, stored.Result.ResourceID)
	if replaySecretErr != nil {
		return SecretMutationResult{}, replaySecretErr
	}
	if key.Status != accesscontrol.APIKeyStatusActive ||
		(key.ExpiresAt != nil && !now.Before(*key.ExpiresAt)) ||
		key.Owner.Kind != issued.Data.Owner.Kind || string(key.Owner.ID) != issued.Data.Owner.ID ||
		string(key.ContextTeamID) != issued.Data.ContextTeamID {
		return SecretMutationResult{}, ErrCredentialUnavailable
	}
	snapshot, replaySecretErr := service.repository.GetCredential(ctx, command.Scope.NamespaceID,
		stored.Result.ResourceID, issued.Credential.ID)
	if replaySecretErr != nil {
		return SecretMutationResult{}, replaySecretErr
	}
	credential := snapshot.Credential
	deliverable := (credential.Status == accesscontrol.CredentialStatusActive ||
		credential.Status == accesscontrol.CredentialStatusRetiring) && !now.Before(credential.NotBefore) &&
		(credential.ExpiresAt == nil || now.Before(*credential.ExpiresAt))
	if !deliverable {
		return SecretMutationResult{}, ErrCredentialUnavailable
	}
	result := secretMutation(key, accesscontrol.CredentialVersion{
		ID:       accesscontrol.CredentialVersionID(issued.Credential.ID),
		APIKeyID: accesscontrol.APIKeyID(issued.Credential.KeyID), KID: issued.Credential.KID,
		Status: issued.Credential.Status, NotBefore: issued.Credential.NotBefore,
		ExpiresAt: issued.Credential.ExpiresAt, RevokedAt: issued.Credential.RevokedAt,
		CreatedAt: issued.Credential.CreatedAt,
	}, issued.Secret, issued.AccessPolicyBindings,
		issued.RateLimitOverride, append([]byte(nil), plaintext...), true)
	result.ResponseRevision = stored.Result.ResourceRevision
	if err := service.waitAPIKeyActive(ctx, command.Scope.NamespaceID, stored.Result.ResourceID, issued.Credential.KID); err != nil {
		return SecretMutationResult{}, err
	}
	return result, nil
}

func (service *Service) waitAPIKeyActive(ctx context.Context, namespaceID, keyID, publicID string) error {
	if service == nil || service.waiter == nil {
		return ErrUnavailable
	}
	waitContext, cancel := context.WithTimeout(ctx, service.publicationTimeout)
	defer cancel()
	if err := service.waiter.WaitAPIKeyActive(waitContext, namespaceID, keyID, publicID); err != nil {
		return fmt.Errorf("%w: API key publication: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) bindCommand(namespaceID, principalID, endpoint, key string, body any, now time.Time) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID), principalID,
		endpoint, key, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		if errors.Is(err, managementcommand.ErrConflict) {
			return managementcommand.Command{}, err
		}
		return managementcommand.Command{}, ErrInvalidRequest
	}
	return command, nil
}

func (service *Service) twoIDs() (string, string, error) {
	first, err := service.nextID()
	if err != nil {
		return "", "", err
	}
	second, err := service.nextID()
	return first, second, err
}

func (service *Service) nextID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", ErrUnavailable
	}
	return value, nil
}

func (service *Service) timeNow() time.Time { return service.now().UTC() }

func marshalIssuedSecret(
	key accesscontrol.APIKey,
	credential accesscontrol.CredentialVersion,
	plaintext string,
	accessBindings []PolicyBindingReceipt,
	rateLimitOverride *RateLimitOverrideReceipt,
	deliveryExpiry time.Time,
) ([]byte, error) {
	return json.Marshal(IssuedSecret{
		Data: keyMetadata(key), Credential: credentialMetadata(credential, key.UpdatedAt),
		Secret: plaintext, AccessPolicyBindings: clonePolicyBindingReceipts(accessBindings),
		RateLimitOverride: cloneRateLimitOverrideReceipt(rateLimitOverride), DeliveryExpiresAt: deliveryExpiry,
	})
}

func secretMutation(
	key accesscontrol.APIKey,
	credential accesscontrol.CredentialVersion,
	secret string,
	accessBindings []PolicyBindingReceipt,
	rateLimitOverride *RateLimitOverrideReceipt,
	body []byte,
	replayed bool,
) SecretMutationResult {
	return SecretMutationResult{
		Key: key, Credential: credentialMetadata(credential, key.UpdatedAt), Secret: secret,
		AccessPolicyBindings: clonePolicyBindingReceipts(accessBindings),
		RateLimitOverride:    cloneRateLimitOverrideReceipt(rateLimitOverride),
		CanonicalJSON:        append([]byte(nil), body...), ResponseRevision: uint64(key.Revision), Replayed: replayed,
	}
}

func credentialMetadata(value accesscontrol.CredentialVersion, now time.Time) CredentialMetadata {
	status := value.Status
	if value.ExpiresAt != nil && !now.Before(*value.ExpiresAt) && status != accesscontrol.CredentialStatusRevoked {
		status = accesscontrol.CredentialStatusExpired
	}
	return CredentialMetadata{
		ID: string(value.ID), KeyID: string(value.APIKeyID), KID: value.KID,
		Status: status, Revealable: len(value.SecretCiphertext) > 0, NotBefore: value.NotBefore,
		ExpiresAt: cloneTime(value.ExpiresAt), RevokedAt: cloneTime(value.RevokedAt), CreatedAt: value.CreatedAt,
	}
}

func responseAAD(endpoint, namespaceID, keyID string, revision uint64) []byte {
	return []byte(fmt.Sprintf("vllm-sr/api-key-secret-response/v1\x00%s\x00%s\x00%s\x00%d", endpoint, namespaceID, keyID, revision))
}

func revealAAD(namespaceID, keyID, credentialID, kid string) []byte {
	return []byte("vllm-sr/api-key-reveal/v1\x00" + namespaceID + "\x00" + keyID + "\x00" + credentialID + "\x00" + kid)
}

func validatePage(namespaceID string, pageSize int) (int, error) {
	if !canonicalUUID(namespaceID) {
		return 0, ErrInvalidRequest
	}
	if pageSize == 0 {
		return defaultPageSize, nil
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return 0, ErrInvalidRequest
	}
	return pageSize, nil
}

func validateActor(namespaceID string, actor Actor) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) || strings.TrimSpace(actor.RequestID) == "" || len(actor.ActorChain) > 32 {
		return ErrInvalidRequest
	}
	for _, id := range actor.ActorChain {
		if !canonicalUUID(id) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateName(value string) error {
	if value == "" || len(value) > 200 || strings.TrimSpace(value) != value {
		return ErrInvalidRequest
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateOwner(namespaceID string, owner Owner, contextTeamID string) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(owner.ID) ||
		(owner.Kind != accesscontrol.SubjectKindUser && owner.Kind != accesscontrol.SubjectKindTeam) {
		return ErrInvalidRequest
	}
	if owner.Kind == accesscontrol.SubjectKindTeam && contextTeamID != "" {
		return ErrInvalidRequest
	}
	if contextTeamID != "" && !canonicalUUID(contextTeamID) {
		return ErrInvalidRequest
	}
	return nil
}

func canonicalAccessPolicyIDs(input []string) ([]string, error) {
	if len(input) > maximumAccessOverrides {
		return nil, ErrInvalidRequest
	}
	result := append([]string(nil), input...)
	for _, policyID := range result {
		if !canonicalUUID(policyID) {
			return nil, ErrInvalidRequest
		}
	}
	slices.Sort(result)
	for index := 1; index < len(result); index++ {
		if result[index] == result[index-1] {
			return nil, ErrInvalidRequest
		}
	}
	return result, nil
}

func canonicalRateLimitOverride(input *RateLimitOverrideInput) (*RateLimitOverrideInput, error) {
	if input == nil {
		return nil, nil
	}
	hasPolicy := input.PolicyID != ""
	hasInline := input.InlinePolicy != nil
	if hasPolicy == hasInline || (hasPolicy && !canonicalUUID(input.PolicyID)) {
		return nil, ErrInvalidRequest
	}
	result := &RateLimitOverrideInput{PolicyID: input.PolicyID}
	if !hasInline {
		return result, nil
	}
	rules := append([]policymanagement.RateLimitRule(nil), input.InlinePolicy.Rules...)
	for index := range rules {
		rules[index].Ordinal = 0
		if rules[index].GCRABurstTolerance != nil {
			value := *rules[index].GCRABurstTolerance
			rules[index].GCRABurstTolerance = &value
		}
	}
	result.InlinePolicy = &InlineRateLimitPolicyInput{
		Name:        strings.TrimSpace(input.InlinePolicy.Name),
		Description: strings.TrimSpace(input.InlinePolicy.Description), Rules: rules,
	}
	if result.InlinePolicy.Name == "" || len(rules) == 0 {
		return nil, ErrInvalidRequest
	}
	return result, nil
}

func validateFutureExpiry(value *time.Time, now time.Time) error {
	if value != nil && !value.After(now) {
		return ErrInvalidRequest
	}
	return nil
}

func validKeyStatusFilter(value accesscontrol.APIKeyStatus) bool {
	return value == "" || value == accesscontrol.APIKeyStatusActive || value == accesscontrol.APIKeyStatusDisabled || value == accesscontrol.APIKeyStatusDeleted
}

func validCredentialStatusFilter(value accesscontrol.CredentialStatus) bool {
	return value == "" || value.Valid()
}

func validOwnerFilter(kind accesscontrol.SubjectKind, id string) bool {
	if kind == "" && id == "" {
		return true
	}
	return (kind == accesscontrol.SubjectKindUser || kind == accesscontrol.SubjectKindTeam) && canonicalUUID(id)
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func canonicalTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	canonical := value.UTC()
	return &canonical
}

func clonePolicyBindingReceipts(input []PolicyBindingReceipt) []PolicyBindingReceipt {
	return append([]PolicyBindingReceipt(nil), input...)
}

func cloneRateLimitOverrideReceipt(input *RateLimitOverrideReceipt) *RateLimitOverrideReceipt {
	if input == nil {
		return nil
	}
	result := *input
	return &result
}

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
