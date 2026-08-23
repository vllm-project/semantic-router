package policymanagement

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
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
		return Page[AccessPolicyBinding]{Items: []AccessPolicyBinding{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListAccessBindings(ctx, query)
	if err != nil {
		return Page[AccessPolicyBinding]{}, err
	}
	return bindingPage(service.cursors, request, "access_policy_bindings", page.Items, page.HasMore, pageSize)
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
		return Page[RateLimitBinding]{Items: []RateLimitBinding{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListRateBindings(ctx, query)
	if err != nil {
		return Page[RateLimitBinding]{}, err
	}
	return bindingPage(service.cursors, request, "rate_limit_bindings", page.Items, page.HasMore, pageSize)
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

func (service *Service) bindCommand(namespaceID, principalID, endpoint, key string, body any) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	now := service.timeNow()
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID),
		principalID, endpoint, key, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	return command, nil
}

func (service *Service) replay(ctx context.Context, command managementcommand.Command, kind string) (MutationResult, bool, error) {
	replay, found, err := service.repository.Replay(ctx, command)
	if err != nil || !found {
		return replay, found, err
	}
	if replay.Kind != kind {
		return MutationResult{}, false, ErrUnavailable
	}
	return replay, true, nil
}

func (service *Service) prepareRules(policyID string, input []RateLimitRule) ([]RateLimitRule, error) {
	var sourceErr error
	rules, err := compileRateLimitRules(policyID, input, func() string {
		if sourceErr != nil {
			return ""
		}
		id, err := service.nextID()
		if err != nil {
			sourceErr = err
			return ""
		}
		return id
	})
	if sourceErr != nil {
		return nil, sourceErr
	}
	return rules, err
}

func (service *Service) nextID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", ErrUnavailable
	}
	return value, nil
}

func (service *Service) timeNow() time.Time { return service.now().UTC().Truncate(time.Microsecond) }

func (service *Service) policyQuery(request ListPoliciesRequest, kind string) (PolicyQuery, int, error) {
	pageSize, err := validatePage(request.NamespaceID, request.PageSize)
	if service == nil || err != nil || (request.Status != "" && !request.Status.Valid()) {
		return PolicyQuery{}, 0, ErrInvalidRequest
	}
	search, err := managementsearch.Normalize(request.Search)
	if err != nil {
		return PolicyQuery{}, 0, ErrInvalidRequest
	}
	request.Search = search
	scopeDigest, err := request.Scope.Digest()
	if err != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return PolicyQuery{}, 0, ErrInvalidRequest
	}
	query := PolicyQuery{
		NamespaceID: request.NamespaceID, Status: request.Status, Search: search,
		Scope: request.Scope, Limit: pageSize,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != kind || cursor.NamespaceID != request.NamespaceID ||
			cursor.Status != string(request.Status) || cursor.PolicyID != "" || cursor.SubjectType != "" ||
			cursor.SubjectID != "" || cursor.Mode != "" || cursor.Search != search || cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return PolicyQuery{}, 0, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	return query, pageSize, nil
}

func (service *Service) bindingQuery(request ListBindingsRequest, kind string, rate bool) (BindingQuery, int, error) {
	pageSize, err := validatePage(request.NamespaceID, request.PageSize)
	if service == nil || err != nil || (request.PolicyID != "" && !canonicalUUID(request.PolicyID)) ||
		(request.Status != "" && !request.Status.Valid()) || (!rate && request.Mode != "") ||
		(rate && request.Mode != "" && !request.Mode.Valid()) {
		return BindingQuery{}, 0, ErrInvalidRequest
	}
	if request.Subject != nil && validateSubject(request.NamespaceID, *request.Subject) != nil {
		return BindingQuery{}, 0, ErrInvalidRequest
	}
	scopeDigest, err := request.Scope.Digest()
	if err != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return BindingQuery{}, 0, ErrInvalidRequest
	}
	query := BindingQuery{
		NamespaceID: request.NamespaceID, PolicyID: request.PolicyID,
		Subject: request.Subject, Status: request.Status, Mode: request.Mode, Scope: request.Scope, Limit: pageSize,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		subjectType, subjectID := "", ""
		if request.Subject != nil {
			subjectType, subjectID = string(request.Subject.Type), request.Subject.ID
		}
		if err != nil || cursor.Kind != kind || cursor.NamespaceID != request.NamespaceID ||
			cursor.PolicyID != request.PolicyID || cursor.SubjectType != subjectType || cursor.SubjectID != subjectID ||
			cursor.Status != string(request.Status) || cursor.Mode != string(request.Mode) ||
			cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return BindingQuery{}, 0, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	return query, pageSize, nil
}

func policyResultScopeEmpty(scope accesscontrol.ResultScope, resourceType accesscontrol.ScopeResourceType) bool {
	return !scope.All && len(scope.IDs(resourceType)) == 0
}

func policyPage[T AccessPolicy | RateLimitPolicy](
	cursors cursorCodec,
	request ListPoliciesRequest,
	kind string,
	items []T,
	hasMore bool,
	pageSize int,
) (Page[T], error) {
	result := Page[T]{Items: items, HasMore: hasMore, PageSize: pageSize}
	if !hasMore {
		return result, nil
	}
	if len(items) == 0 {
		return Page[T]{}, ErrUnavailable
	}
	createdAt, id := policyPosition(any(items[len(items)-1]))
	scopeDigest, err := request.Scope.Digest()
	if err != nil {
		return Page[T]{}, ErrInvalidRequest
	}
	nextCursor, err := cursors.encode(cursorPayload{
		Kind:        kind,
		NamespaceID: request.NamespaceID, Status: string(request.Status), Search: request.Search,
		ScopeDigest: scopeDigest,
		CreatedAt:   createdAt, ID: id,
	})
	if err != nil || nextCursor == "" {
		return Page[T]{}, ErrUnavailable
	}
	result.NextCursor = nextCursor
	return result, nil
}

func bindingPage[T AccessPolicyBinding | RateLimitBinding](
	cursors cursorCodec,
	request ListBindingsRequest,
	kind string,
	items []T,
	hasMore bool,
	pageSize int,
) (Page[T], error) {
	result := Page[T]{Items: items, HasMore: hasMore, PageSize: pageSize}
	if !hasMore {
		return result, nil
	}
	if len(items) == 0 {
		return Page[T]{}, ErrUnavailable
	}
	createdAt, id := bindingPosition(any(items[len(items)-1]))
	subjectType, subjectID := "", ""
	if request.Subject != nil {
		subjectType, subjectID = string(request.Subject.Type), request.Subject.ID
	}
	scopeDigest, err := request.Scope.Digest()
	if err != nil {
		return Page[T]{}, ErrInvalidRequest
	}
	nextCursor, err := cursors.encode(cursorPayload{
		Kind: kind, NamespaceID: request.NamespaceID,
		PolicyID: request.PolicyID, SubjectType: subjectType, SubjectID: subjectID,
		Status: string(request.Status), Mode: string(request.Mode), ScopeDigest: scopeDigest,
		CreatedAt: createdAt, ID: id,
	})
	if err != nil || nextCursor == "" {
		return Page[T]{}, ErrUnavailable
	}
	result.NextCursor = nextCursor
	return result, nil
}

func policyPosition(value any) (time.Time, string) {
	switch item := value.(type) {
	case AccessPolicy:
		return item.CreatedAt, item.ID
	case RateLimitPolicy:
		return item.CreatedAt, item.ID
	default:
		return time.Time{}, ""
	}
}

func bindingPosition(value any) (time.Time, string) {
	switch item := value.(type) {
	case AccessPolicyBinding:
		return item.CreatedAt, item.ID
	case RateLimitBinding:
		return item.CreatedAt, item.ID
	default:
		return time.Time{}, ""
	}
}

func validateMutation(namespaceID, resourceID string, revision uint64, actor Actor) error {
	if validateActor(namespaceID, actor) != nil || validateIDs(namespaceID, resourceID) != nil || revision == 0 {
		return ErrInvalidRequest
	}
	return nil
}

func validateBindingMutation(request UpdateBindingRequest) error {
	if validateMutation(request.NamespaceID, request.BindingID, request.ExpectedRevision, request.Actor) != nil ||
		!request.Status.Valid() {
		return ErrInvalidRequest
	}
	return nil
}

func validatePage(namespaceID string, pageSize int) (int, error) {
	if !canonicalUUID(namespaceID) {
		return 0, ErrInvalidRequest
	}
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return 0, ErrInvalidRequest
	}
	return pageSize, nil
}

func validateActor(namespaceID string, actor Actor) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) || strings.TrimSpace(actor.RequestID) == "" {
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

func validateSubject(namespaceID string, subject Subject) error {
	if !subject.Type.Valid() || !canonicalUUID(subject.ID) || !canonicalUUID(namespaceID) {
		return ErrInvalidRequest
	}
	return nil
}

func validatePolicyMetadata(name, description string, status accesscontrol.PolicyStatus) error {
	if validateText(name, 200, false) != nil || validateText(description, 1000, true) != nil || !status.Valid() {
		return ErrInvalidRequest
	}
	return nil
}

func validateGrants(policyID string, grants []AccessGrant) error {
	seen := make(map[AccessGrant]struct{}, len(grants))
	for _, grant := range grants {
		domain := accesscontrol.AccessPolicyGrant{
			PolicyID:   accesscontrol.AccessPolicyID(policyID),
			Resource:   accesscontrol.GrantResource{Type: grant.ResourceType, ID: accesscontrol.ResourceID(grant.ResourceID)},
			Permission: grant.Permission, Effect: grant.Effect,
		}
		if domain.Validate() != nil {
			return ErrInvalidRequest
		}
		if _, duplicate := seen[grant]; duplicate {
			return ErrInvalidRequest
		}
		seen[grant] = struct{}{}
	}
	return nil
}

func validateRules(policyID string, rules []RateLimitRule) error {
	seen := make(map[string]struct{}, len(rules))
	for index, rule := range rules {
		if !canonicalUUID(rule.ID) || rule.Ordinal != uint32(index) {
			return ErrInvalidRequest
		}
		if _, duplicate := seen[rule.ID]; duplicate {
			return ErrInvalidRequest
		}
		seen[rule.ID] = struct{}{}
		if rateRuleDomain(policyID, rule).Validate() != nil || !costLimitFitsStorage(rule) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func costLimitFitsStorage(rule RateLimitRule) bool {
	if rule.Metric != accesscontrol.RateMetricCost {
		return true
	}
	parts := strings.Split(string(rule.Limit), ".")
	if len(parts) < 1 || len(parts) > 2 {
		return false
	}
	fraction := ""
	if len(parts) == 2 {
		fraction = parts[1]
	}
	if len(fraction) > 15 {
		return false
	}
	scaled := strings.TrimLeft(parts[0]+fraction+strings.Repeat("0", 15-len(fraction)), "0")
	return len(scaled) <= 42
}

func rateRuleDomain(policyID string, rule RateLimitRule) accesscontrol.RateLimitRule {
	return accesscontrol.RateLimitRule{
		ID:       accesscontrol.RateLimitRuleID(rule.ID),
		PolicyID: accesscontrol.RateLimitPolicyID(policyID), Metric: rule.Metric,
		Algorithm: rule.Algorithm, Limit: rule.Limit, Window: rule.Window.Duration(),
		CalendarPeriod: rule.CalendarPeriod, Timezone: rule.Timezone,
		BucketCapacity: rule.BucketCapacity, RefillAmount: rule.RefillAmount,
		RefillPeriod: rule.RefillPeriod.Duration(), GCRAEmissionInterval: rule.GCRAEmissionInterval.Duration(),
		GCRABurstTolerance: cloneInt64(rule.GCRABurstTolerance), Accounting: rule.Accounting,
		Enforcement: rule.Enforcement, Ordinal: rule.Ordinal,
	}
}

func normalizePolicyText(name, description *string) {
	*name = strings.TrimSpace(*name)
	*description = strings.TrimSpace(*description)
}

func defaultPolicyStatus(status accesscontrol.PolicyStatus) accesscontrol.PolicyStatus {
	if status == "" {
		return accesscontrol.PolicyStatusDraft
	}
	return status
}

func validateText(value string, maximum int, optional bool) error {
	if (!optional && value == "") || len(value) > maximum {
		return ErrInvalidRequest
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateIDs(namespaceID, resourceID string) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(resourceID) {
		return ErrInvalidRequest
	}
	return nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func cloneGrants(input []AccessGrant) []AccessGrant { return append([]AccessGrant(nil), input...) }

func cloneRules(input []RateLimitRule) []RateLimitRule {
	result := append([]RateLimitRule(nil), input...)
	for index := range result {
		result[index].GCRABurstTolerance = cloneInt64(result[index].GCRABurstTolerance)
	}
	return result
}

func cloneInt64(input *int64) *int64 {
	if input == nil {
		return nil
	}
	value := *input
	return &value
}
