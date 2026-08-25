package policymanagement

import (
	"context"
	"encoding/json"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

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
	if len(rules) > maximumRules {
		return ErrInvalidRequest
	}
	seen := make(map[string]struct{}, len(rules))
	for index, rule := range rules {
		// #nosec G115 -- the rule count is bounded to maximumRules (128) above.
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
