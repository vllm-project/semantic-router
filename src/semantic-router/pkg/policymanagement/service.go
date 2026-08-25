package policymanagement

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize = 50
	maximumPageSize = 200
	maximumGrants   = 512
	maximumRules    = 128
)

type Options struct {
	Repository     Repository
	CommandCodec   *managementcommand.Codec
	CursorKeyring  securitykeyring.Symmetric
	IdempotencyTTL time.Duration
	Now            func() time.Time
	NewID          func() string
}

type Service struct {
	repository     Repository
	commands       *managementcommand.Codec
	cursors        cursorCodec
	idempotencyTTL time.Duration
	now            func() time.Time
	newID          func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.CommandCodec == nil ||
		options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour {
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
	return &Service{
		repository: options.Repository, commands: options.CommandCodec,
		cursors: cursors, idempotencyTTL: options.IdempotencyTTL, now: now, newID: newID,
	}, nil
}

func (service *Service) Close() { service.cursors.close() }

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) GetAccessPolicy(ctx context.Context, namespaceID, policyID string) (AccessPolicy, error) {
	if service == nil || validateIDs(namespaceID, policyID) != nil {
		return AccessPolicy{}, ErrInvalidRequest
	}
	return service.repository.GetAccessPolicy(ctx, namespaceID, policyID)
}

func (service *Service) ListAccessPolicies(ctx context.Context, request ListPoliciesRequest) (Page[AccessPolicy], error) {
	query, pageSize, err := service.policyQuery(request, "access_policies")
	if err != nil {
		return Page[AccessPolicy]{}, err
	}
	request.Search = query.Search
	if policyResultScopeEmpty(request.Scope, accesscontrol.ScopeResourceAccessPolicy) {
		return Page[AccessPolicy]{Items: []AccessPolicy{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListAccessPolicies(ctx, query)
	if err != nil {
		return Page[AccessPolicy]{}, err
	}
	return policyPage(service.cursors, request, "access_policies", page.Items, page.HasMore, pageSize)
}

func (service *Service) CreateAccessPolicy(ctx context.Context, request CreateAccessPolicyRequest) (MutationResult, error) {
	normalizePolicyText(&request.Name, &request.Description)
	request.Status = defaultPolicyStatus(request.Status)
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		validatePolicyMetadata(request.Name, request.Description, request.Status) != nil ||
		len(request.Grants) > maximumGrants ||
		validateGrants("00000000-0000-4000-8000-000000000000", request.Grants) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Name        string                     `json:"name"`
		Description string                     `json:"description"`
		Status      accesscontrol.PolicyStatus `json:"status"`
		Grants      []AccessGrant              `json:"grants"`
	}{request.Name, request.Description, request.Status, request.Grants}
	command, createAccessPolicyErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/access-policies", request.IdempotencyKey, canonical)
	if createAccessPolicyErr != nil {
		return MutationResult{}, createAccessPolicyErr
	}
	if replay, found, err := service.replay(ctx, command, "access_policy"); err != nil || found {
		return replay, err
	}
	policyID, createAccessPolicyErr := service.nextID()
	if createAccessPolicyErr != nil {
		return MutationResult{}, createAccessPolicyErr
	}
	now := service.timeNow()
	policy := AccessPolicy{
		ID: policyID, NamespaceID: request.NamespaceID, Name: request.Name,
		Description: request.Description, Status: request.Status, Revision: 1,
		Grants: cloneGrants(request.Grants), CreatedAt: now, UpdatedAt: now,
	}
	return service.repository.CreateAccessPolicy(ctx, CreateAccessPolicyMutation{
		Policy: policy, Command: command, Actor: request.Actor,
	})
}

func (service *Service) UpdateAccessPolicy(ctx context.Context, request UpdateAccessPolicyRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor) != nil ||
		(request.Name == nil && request.Description == nil && request.Status == nil && request.Grants == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	policy, err := service.repository.GetAccessPolicy(ctx, request.NamespaceID, request.PolicyID)
	if err != nil {
		return MutationResult{}, err
	}
	if policy.Revision != request.ExpectedRevision {
		return MutationResult{}, ErrRevisionConflict
	}
	if request.Name != nil {
		policy.Name = strings.TrimSpace(*request.Name)
	}
	if request.Description != nil {
		policy.Description = strings.TrimSpace(*request.Description)
	}
	if request.Status != nil {
		policy.Status = *request.Status
	}
	if request.Grants != nil {
		policy.Grants = cloneGrants(*request.Grants)
	}
	if validatePolicyMetadata(policy.Name, policy.Description, policy.Status) != nil ||
		len(policy.Grants) > maximumGrants ||
		validateGrants(policy.ID, policy.Grants) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.UpdateAccessPolicy(ctx, policy, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteAccessPolicy(ctx context.Context, request DeletePolicyRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteAccessPolicy(ctx, request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor)
}

func (service *Service) GetRateLimitPolicy(ctx context.Context, namespaceID, policyID string) (RateLimitPolicy, error) {
	if service == nil || validateIDs(namespaceID, policyID) != nil {
		return RateLimitPolicy{}, ErrInvalidRequest
	}
	return service.repository.GetRateLimitPolicy(ctx, namespaceID, policyID)
}

func (service *Service) ListRateLimitPolicies(ctx context.Context, request ListPoliciesRequest) (Page[RateLimitPolicy], error) {
	query, pageSize, err := service.policyQuery(request, "rate_limit_policies")
	if err != nil {
		return Page[RateLimitPolicy]{}, err
	}
	request.Search = query.Search
	if policyResultScopeEmpty(request.Scope, accesscontrol.ScopeResourceRateLimitPolicy) {
		return Page[RateLimitPolicy]{Items: []RateLimitPolicy{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListRateLimitPolicies(ctx, query)
	if err != nil {
		return Page[RateLimitPolicy]{}, err
	}
	return policyPage(service.cursors, request, "rate_limit_policies", page.Items, page.HasMore, pageSize)
}

func (service *Service) CreateRateLimitPolicy(ctx context.Context, request CreateRateLimitPolicyRequest) (MutationResult, error) {
	normalizePolicyText(&request.Name, &request.Description)
	request.Status = defaultPolicyStatus(request.Status)
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		validatePolicyMetadata(request.Name, request.Description, request.Status) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Name        string                     `json:"name"`
		Description string                     `json:"description"`
		Status      accesscontrol.PolicyStatus `json:"status"`
		Rules       []RateLimitRule            `json:"rules"`
	}{request.Name, request.Description, request.Status, request.Rules}
	command, createRateLimitPolicyErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/rate-limit-policies", request.IdempotencyKey, canonical)
	if createRateLimitPolicyErr != nil {
		return MutationResult{}, createRateLimitPolicyErr
	}
	if replay, found, err := service.replay(ctx, command, "rate_limit_policy"); err != nil || found {
		return replay, err
	}
	policyID, createRateLimitPolicyErr := service.nextID()
	if createRateLimitPolicyErr != nil {
		return MutationResult{}, createRateLimitPolicyErr
	}
	rules, createRateLimitPolicyErr := service.prepareRules(policyID, request.Rules)
	if createRateLimitPolicyErr != nil {
		return MutationResult{}, createRateLimitPolicyErr
	}
	now := service.timeNow()
	policy := RateLimitPolicy{
		ID: policyID, NamespaceID: request.NamespaceID,
		Name: request.Name, Description: request.Description, Status: request.Status,
		Revision: 1, Rules: rules, CreatedAt: now, UpdatedAt: now,
	}
	return service.repository.CreateRateLimitPolicy(ctx, CreateRateLimitPolicyMutation{
		Policy: policy, Command: command, Actor: request.Actor,
	})
}

func (service *Service) UpdateRateLimitPolicy(ctx context.Context, request UpdateRateLimitPolicyRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor) != nil ||
		(request.Name == nil && request.Description == nil && request.Status == nil && request.Rules == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	policy, err := service.repository.GetRateLimitPolicy(ctx, request.NamespaceID, request.PolicyID)
	if err != nil {
		return MutationResult{}, err
	}
	if policy.Revision != request.ExpectedRevision {
		return MutationResult{}, ErrRevisionConflict
	}
	if request.Name != nil {
		policy.Name = strings.TrimSpace(*request.Name)
	}
	if request.Description != nil {
		policy.Description = strings.TrimSpace(*request.Description)
	}
	if request.Status != nil {
		policy.Status = *request.Status
	}
	if request.Rules != nil {
		policy.Rules, err = service.prepareRules(policy.ID, *request.Rules)
		if err != nil {
			return MutationResult{}, err
		}
	}
	if validatePolicyMetadata(policy.Name, policy.Description, policy.Status) != nil ||
		validateRules(policy.ID, policy.Rules) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.UpdateRateLimitPolicy(ctx, policy, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteRateLimitPolicy(ctx context.Context, request DeletePolicyRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteRateLimitPolicy(ctx, request.NamespaceID, request.PolicyID,
		request.ExpectedRevision, request.Actor)
}

func (service *Service) GetAccessBinding(ctx context.Context, namespaceID, bindingID string) (AccessPolicyBinding, error) {
	if service == nil || validateIDs(namespaceID, bindingID) != nil {
		return AccessPolicyBinding{}, ErrInvalidRequest
	}
	return service.repository.GetAccessBinding(ctx, namespaceID, bindingID)
}

func (service *Service) ListAccessBindings(ctx context.Context, request ListBindingsRequest) (Page[AccessPolicyBinding], error) {
	query, pageSize, err := service.bindingQuery(request, "access_policy_bindings", false)
	if err != nil {
		return Page[AccessPolicyBinding]{}, err
	}
	if policyResultScopeEmpty(request.Scope, accesscontrol.ScopeResourceAccessPolicy) {
		return emptyBindingPage[AccessPolicyBinding](pageSize, request.IncludeTotal), nil
	}
	page, err := service.repository.ListAccessBindings(ctx, query)
	if err != nil {
		return Page[AccessPolicyBinding]{}, err
	}
	return bindingPage(service.cursors, request, "access_policy_bindings", page.Items, page.HasMore, pageSize, page.TotalCount)
}

func (service *Service) CreateAccessBinding(ctx context.Context, request CreateAccessBindingRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.PolicyID) || validateSubject(request.NamespaceID, request.Subject) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		PolicyID string  `json:"policyId"`
		Subject  Subject `json:"subject"`
	}{request.PolicyID, request.Subject}
	command, createAccessBindingErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/access-policy-bindings", request.IdempotencyKey, canonical)
	if createAccessBindingErr != nil {
		return MutationResult{}, createAccessBindingErr
	}
	if replay, found, err := service.replay(ctx, command, "access_policy_binding"); err != nil || found {
		return replay, err
	}
	bindingID, createAccessBindingErr := service.nextID()
	if createAccessBindingErr != nil {
		return MutationResult{}, createAccessBindingErr
	}
	now := service.timeNow()
	binding := AccessPolicyBinding{
		ID: bindingID, NamespaceID: request.NamespaceID,
		PolicyID: request.PolicyID, Subject: request.Subject, Status: accesscontrol.BindingStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	return service.repository.CreateAccessBinding(ctx, CreateAccessBindingMutation{
		Binding: binding, Command: command, Actor: request.Actor,
	})
}

func (service *Service) UpdateAccessBinding(ctx context.Context, request UpdateBindingRequest) (MutationResult, error) {
	if service == nil || validateBindingMutation(request) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.UpdateAccessBinding(ctx, request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Status, request.Actor)
}

func (service *Service) DeleteAccessBinding(ctx context.Context, request DeleteBindingRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteAccessBinding(ctx, request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Actor)
}

func (service *Service) GetRateBinding(ctx context.Context, namespaceID, bindingID string) (RateLimitBinding, error) {
	if service == nil || validateIDs(namespaceID, bindingID) != nil {
		return RateLimitBinding{}, ErrInvalidRequest
	}
	return service.repository.GetRateBinding(ctx, namespaceID, bindingID)
}

func (service *Service) ListRateBindings(ctx context.Context, request ListBindingsRequest) (Page[RateLimitBinding], error) {
	query, pageSize, err := service.bindingQuery(request, "rate_limit_bindings", true)
	if err != nil {
		return Page[RateLimitBinding]{}, err
	}
	if policyResultScopeEmpty(request.Scope, accesscontrol.ScopeResourceRateLimitPolicy) {
		return emptyBindingPage[RateLimitBinding](pageSize, request.IncludeTotal), nil
	}
	page, err := service.repository.ListRateBindings(ctx, query)
	if err != nil {
		return Page[RateLimitBinding]{}, err
	}
	return bindingPage(service.cursors, request, "rate_limit_bindings", page.Items, page.HasMore, pageSize, page.TotalCount)
}

func emptyBindingPage[T AccessPolicyBinding | RateLimitBinding](pageSize int, includeTotal bool) Page[T] {
	page := Page[T]{Items: []T{}, PageSize: pageSize}
	if includeTotal {
		count := uint64(0)
		page.TotalCount = &count
	}
	return page
}

func (service *Service) CreateRateBinding(ctx context.Context, request CreateRateBindingRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.PolicyID) || validateSubject(request.NamespaceID, request.Subject) != nil || !request.Mode.Valid() {
		return MutationResult{}, ErrInvalidRequest
	}
	canonical := struct {
		PolicyID string                        `json:"policyId"`
		Subject  Subject                       `json:"subject"`
		Mode     accesscontrol.RateBindingMode `json:"mode"`
	}{request.PolicyID, request.Subject, request.Mode}
	command, createRateBindingErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/rate-limit-bindings", request.IdempotencyKey, canonical)
	if createRateBindingErr != nil {
		return MutationResult{}, createRateBindingErr
	}
	if replay, found, err := service.replay(ctx, command, "rate_limit_binding"); err != nil || found {
		return replay, err
	}
	bindingID, createRateBindingErr := service.nextID()
	if createRateBindingErr != nil {
		return MutationResult{}, createRateBindingErr
	}
	now := service.timeNow()
	binding := RateLimitBinding{
		ID: bindingID, NamespaceID: request.NamespaceID,
		PolicyID: request.PolicyID, Subject: request.Subject, Mode: request.Mode,
		Status: accesscontrol.BindingStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	return service.repository.CreateRateBinding(ctx, CreateRateBindingMutation{
		Binding: binding, Command: command, Actor: request.Actor,
	})
}

// CreateInlineRateBinding turns an inline Budget into the same ordinary,
// reusable policy and binding resources used by every other caller. There is
// no hidden inline-policy persistence mode.
func (service *Service) CreateInlineRateBinding(ctx context.Context, request CreateInlineRateBindingRequest) (InlineRateBindingResult, error) {
	normalizePolicyText(&request.Name, &request.Description)
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		validatePolicyMetadata(request.Name, request.Description, accesscontrol.PolicyStatusActive) != nil ||
		validateSubject(request.NamespaceID, request.Subject) != nil || !request.Mode.Valid() {
		return InlineRateBindingResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Name        string                        `json:"name"`
		Description string                        `json:"description"`
		Rules       []RateLimitRule               `json:"rules"`
		Subject     Subject                       `json:"subject"`
		Mode        accesscontrol.RateBindingMode `json:"mode"`
	}{request.Name, request.Description, request.Rules, request.Subject, request.Mode}
	command, createInlineRateBindingErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/rate-limit-bindings", request.IdempotencyKey, canonical)
	if createInlineRateBindingErr != nil {
		return InlineRateBindingResult{}, createInlineRateBindingErr
	}
	if replay, found, err := service.replay(ctx, command, "rate_limit_binding"); err != nil {
		return InlineRateBindingResult{}, err
	} else if found {
		binding, getErr := service.repository.GetRateBinding(ctx, request.NamespaceID, replay.ID)
		if getErr != nil {
			return InlineRateBindingResult{}, getErr
		}
		policy, getErr := service.repository.GetRateLimitPolicy(ctx, request.NamespaceID, binding.PolicyID)
		return InlineRateBindingResult{Policy: policy, Binding: binding, Created: true, MutationResult: replay}, getErr
	}
	policyID, createInlineRateBindingErr := service.nextID()
	if createInlineRateBindingErr != nil {
		return InlineRateBindingResult{}, createInlineRateBindingErr
	}
	bindingID, createInlineRateBindingErr := service.nextID()
	if createInlineRateBindingErr != nil {
		return InlineRateBindingResult{}, createInlineRateBindingErr
	}
	rules, createInlineRateBindingErr := service.prepareRules(policyID, request.Rules)
	if createInlineRateBindingErr != nil {
		return InlineRateBindingResult{}, createInlineRateBindingErr
	}
	now := service.timeNow()
	policy := RateLimitPolicy{
		ID: policyID, NamespaceID: request.NamespaceID, Name: request.Name,
		Description: request.Description, Status: accesscontrol.PolicyStatusActive,
		Revision: 1, Rules: rules, CreatedAt: now, UpdatedAt: now,
	}
	binding := RateLimitBinding{
		ID: bindingID, NamespaceID: request.NamespaceID,
		PolicyID: policyID, Subject: request.Subject, Mode: request.Mode,
		Status: accesscontrol.BindingStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	return service.repository.CreateInlineRateBinding(ctx, CreateInlineRateBindingMutation{
		Policy: policy, Binding: binding, Command: command, Actor: request.Actor,
	})
}

func (service *Service) UpdateRateBinding(ctx context.Context, request UpdateBindingRequest) (MutationResult, error) {
	if service == nil || validateBindingMutation(request) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.UpdateRateBinding(ctx, request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Status, request.Actor)
}

func (service *Service) DeleteRateBinding(ctx context.Context, request DeleteBindingRequest) (MutationResult, error) {
	if service == nil || validateMutation(request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteRateBinding(ctx, request.NamespaceID, request.BindingID,
		request.ExpectedRevision, request.Actor)
}
