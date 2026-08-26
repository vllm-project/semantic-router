package accessmanagement

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type Service struct {
	repository Repository
	applied    AppliedPolicyReader
	routing    RoutingPublicationReader
	meters     MeterReader
	waiter     PublicationWaiter
}

type ServiceOptions struct {
	Repository Repository
	Applied    AppliedPolicyReader
	Routing    RoutingPublicationReader
	Meters     MeterReader
	Waiter     PublicationWaiter
}

func NewService(options ServiceOptions) (*Service, error) {
	if options.Repository == nil || options.Applied == nil || options.Routing == nil || options.Meters == nil || options.Waiter == nil {
		return nil, fmt.Errorf("access management requires desired state, applied projection, routing snapshot, meter, and publication dependencies")
	}
	return &Service{
		repository: options.Repository, applied: options.Applied, routing: options.Routing,
		meters: options.Meters, waiter: options.Waiter,
	}, nil
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil {
		return ErrUnavailable
	}
	return service.repository.Ready(ctx)
}

func (service *Service) Inspect(ctx context.Context, namespaceID string, subject Subject) (AuthorizationContext, error) {
	snapshot, err := service.load(ctx, namespaceID, subject)
	if err != nil {
		return AuthorizationContext{}, err
	}
	result := AuthorizationContext{Subject: subject}
	if subject.Kind == accesscontrol.SubjectKindAPIKey {
		if snapshot.LayerSubjects.User != nil {
			result.Ancestors = append(result.Ancestors, *snapshot.LayerSubjects.User)
		} else if snapshot.LayerSubjects.Team != nil {
			result.Ancestors = append(result.Ancestors, *snapshot.LayerSubjects.Team)
		}
	}
	for _, binding := range snapshot.Projection.RateBindings {
		result.RateBindings = append(result.RateBindings, BindingAuthorizationContext{
			BindingID: binding.BindingID, Subject: snapshot.LayerSubjects.Source(binding.Source),
		})
	}
	sort.Slice(result.RateBindings, func(i, j int) bool {
		return result.RateBindings[i].BindingID < result.RateBindings[j].BindingID
	})
	return result, nil
}

func (service *Service) GetEffectivePolicy(ctx context.Context, namespaceID string, subject Subject) (EffectivePolicy, error) {
	snapshot, err := service.load(ctx, namespaceID, subject)
	if err != nil {
		return EffectivePolicy{}, err
	}
	projection, appliedRevision, err := service.appliedProjection(ctx, snapshot)
	if err != nil {
		return EffectivePolicy{}, err
	}
	quotaView, err := service.readQuota(ctx, snapshot, projection)
	if err != nil {
		return EffectivePolicy{}, err
	}
	result := EffectivePolicy{
		Subject: subject, DesiredRevision: snapshot.DesiredRevision,
		AppliedRevision: appliedRevision, Quota: quotaView,
	}
	for _, grant := range projection.Grants {
		result.Access = append(result.Access, GrantView{Grant: grant, Source: snapshot.LayerSubjects.Source(grant.Source)})
	}
	return result, nil
}

func (service *Service) GetQuota(ctx context.Context, namespaceID string, subject Subject) (EffectiveQuota, error) {
	snapshot, err := service.load(ctx, namespaceID, subject)
	if err != nil {
		return EffectiveQuota{}, err
	}
	projection, _, err := service.appliedProjection(ctx, snapshot)
	if err != nil {
		return EffectiveQuota{}, err
	}
	return service.readQuota(ctx, snapshot, projection)
}

func (service *Service) GetRoutingContext(ctx context.Context, namespaceID string, subject Subject) (RoutingContext, error) {
	snapshot, err := service.load(ctx, namespaceID, subject)
	if err != nil {
		return RoutingContext{}, err
	}
	return snapshot.Context, nil
}

func (service *Service) UpdateRoutingContext(ctx context.Context, request UpdateRoutingContextRequest) (RoutingContext, error) {
	if request.ExpectedRevision == 0 {
		return RoutingContext{}, ErrInvalidRequest
	}
	snapshot, updateRoutingContextErr := service.load(ctx, request.NamespaceID, request.Subject)
	if updateRoutingContextErr != nil {
		return RoutingContext{}, updateRoutingContextErr
	}
	if snapshot.SubjectRevision != request.ExpectedRevision {
		return RoutingContext{}, ErrRevisionConflict
	}
	if err := ValidateContextValues(snapshot.Schema, request.Values); err != nil {
		return RoutingContext{}, err
	}
	mutation, updateRoutingContextErr := service.repository.UpdateRoutingContext(ctx, request)
	if updateRoutingContextErr != nil {
		return RoutingContext{}, updateRoutingContextErr
	}
	if err := service.waiter.WaitApplied(ctx, request.NamespaceID, mutation.QuotaPartition, mutation.DesiredRevision); err != nil {
		return RoutingContext{}, fmt.Errorf("%w: wait for routing-context publication: %w", ErrUnavailable, err)
	}
	updated, updateRoutingContextErr := service.load(ctx, request.NamespaceID, request.Subject)
	if updateRoutingContextErr != nil {
		return RoutingContext{}, updateRoutingContextErr
	}
	return updated.Context, nil
}

func (service *Service) Check(ctx context.Context, request AccessCheckRequest) (AccessCheckResult, error) {
	if request.NamespaceID == "" || request.Subject.Validate() != nil || request.Resource.Validate() != nil ||
		!request.Permission.Valid() || !validSimulationPath(request.Path) {
		return AccessCheckResult{}, ErrInvalidRequest
	}
	snapshot, err := service.load(ctx, request.NamespaceID, request.Subject)
	if err != nil {
		return AccessCheckResult{}, err
	}
	exists, err := service.repository.ResourceExists(ctx, request.NamespaceID, request.Resource)
	if err != nil {
		return AccessCheckResult{}, err
	}
	if !exists {
		return AccessCheckResult{}, ErrNotFound
	}
	projection, appliedRevision, err := service.appliedProjection(ctx, snapshot)
	if err != nil {
		return AccessCheckResult{}, err
	}
	contextValues := append([]EffectiveClaim(nil), snapshot.Context.Effective...)
	if request.OverridePresent {
		if err := ValidateContextValues(snapshot.Schema, request.Override); err != nil {
			return AccessCheckResult{}, err
		}
		contextValues = applySimulationOverride(contextValues, request.Subject, request.Override)
	}
	result := AccessCheckResult{
		Subject: request.Subject, Resource: request.Resource,
		Permission: request.Permission, Decision: projection.Evaluate(request.Resource.Type, string(request.Resource.ID), request.Permission),
		RoutingContext: contextValues, Simulation: request.OverridePresent,
		DesiredRevision: snapshot.DesiredRevision, AppliedRevision: appliedRevision,
	}
	for _, grant := range projection.Grants {
		if grant.ResourceType == request.Resource.Type && grant.ResourceID == string(request.Resource.ID) &&
			grant.Permission == request.Permission {
			result.Matched = append(result.Matched, GrantView{Grant: grant, Source: snapshot.LayerSubjects.Source(grant.Source)})
		}
	}
	return result, nil
}

func (service *Service) load(ctx context.Context, namespaceID string, subject Subject) (PolicySnapshot, error) {
	if service == nil || service.repository == nil || strings.TrimSpace(namespaceID) == "" || subject.Validate() != nil {
		return PolicySnapshot{}, ErrInvalidRequest
	}
	snapshot, err := service.repository.LoadPolicySnapshot(ctx, namespaceID, subject)
	if err != nil {
		return PolicySnapshot{}, normalizeRepositoryError(err)
	}
	return snapshot, nil
}

func (service *Service) appliedProjection(ctx context.Context, snapshot PolicySnapshot) (accessprojection.Projection, uint64, error) {
	if snapshot.Subject.Kind == accesscontrol.SubjectKindAPIKey {
		applied, err := service.appliedKeyPolicy(ctx, snapshot)
		if err != nil {
			return accessprojection.Projection{}, 0, err
		}
		return applied.Projection, applied.Active.Revision, nil
	}
	if snapshot.AppliedRevision < snapshot.DesiredRevision {
		return accessprojection.Projection{}, 0, fmt.Errorf("%w: applied subject policy is behind desired state", ErrUnavailable)
	}
	return snapshot.Projection, snapshot.AppliedRevision, nil
}

func (service *Service) readQuota(ctx context.Context, snapshot PolicySnapshot, projection accessprojection.Projection) (EffectiveQuota, error) {
	rules := accessruntime.CompileRuleBindings(projection)
	read, err := service.meters.ReadMeters(ctx, quotaruntime.MeterReadRequest{Partition: projection.QuotaPartition, Rules: rules})
	if err != nil {
		return EffectiveQuota{}, fmt.Errorf("%w: read live quota meters: %w", ErrUnavailable, err)
	}
	if len(read.Meters) != len(rules) {
		return EffectiveQuota{}, fmt.Errorf("%w: quota meter plan changed", ErrUnavailable)
	}
	meters := make(map[quota.CounterIdentity]quotaruntime.Meter, len(read.Meters))
	for _, meter := range read.Meters {
		identity, identityErr := quota.NewCounterIdentity(meter.BindingID, meter.RuleID)
		if identityErr != nil {
			return EffectiveQuota{}, fmt.Errorf("%w: quota meter identity is invalid", ErrUnavailable)
		}
		if _, exists := meters[identity]; exists {
			return EffectiveQuota{}, fmt.Errorf("%w: quota meter identity is duplicated", ErrUnavailable)
		}
		meters[identity] = meter
	}
	result := EffectiveQuota{AsOf: read.AsOf}
	for _, binding := range projection.RateBindings {
		for _, rule := range binding.Rules {
			identity, identityErr := quota.NewCounterIdentity(binding.BindingID, rule.Rule.ID)
			if identityErr != nil {
				return EffectiveQuota{}, fmt.Errorf("%w: projected quota identity is invalid", ErrUnavailable)
			}
			meter, exists := meters[identity]
			if !exists {
				return EffectiveQuota{}, fmt.Errorf("%w: quota meter identity mismatch", ErrUnavailable)
			}
			delete(meters, identity)
			result.Meters = append(result.Meters, QuotaMeterView{
				Binding: binding, Rule: rule,
				Source: snapshot.LayerSubjects.Source(binding.Source), Meter: meter,
			})
			result.FenceIDs = append(result.FenceIDs, meter.ActiveFenceIDs...)
			if result.LimitingRuleID == "" && meter.Enforcement == quota.EnforcementEnforce &&
				(meter.CapacityState == quota.CapacityFenced || meter.CapacityState == quota.CapacityOverLimit ||
					meter.CapacityState == quota.CapacityExhausted) {
				result.LimitingRuleID = meter.RuleID
			}
		}
	}
	if len(meters) != 0 {
		return EffectiveQuota{}, fmt.Errorf("%w: quota meter plan contains unknown counters", ErrUnavailable)
	}
	sort.Strings(result.FenceIDs)
	result.FenceIDs = uniqueStrings(result.FenceIDs)
	return result, nil
}

func applySimulationOverride(current []EffectiveClaim, subject Subject, values map[string]routingsnapshot.ClaimValue) []EffectiveClaim {
	byName := make(map[string]EffectiveClaim, len(current)+len(values))
	for _, claim := range current {
		byName[claim.Name] = claim
	}
	for name, value := range values {
		byName[name] = EffectiveClaim{StoredClaim: StoredClaim{Name: name, Value: value}, Source: subject}
	}
	names := make([]string, 0, len(byName))
	for name := range byName {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]EffectiveClaim, 0, len(names))
	for _, name := range names {
		result = append(result, byName[name])
	}
	return result
}

func validSimulationPath(path string) bool {
	return path == "" || (len(path) <= 2048 && strings.HasPrefix(path, "/") && !strings.ContainsAny(path, "?#\x00"))
}

func uniqueStrings(values []string) []string {
	if len(values) < 2 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}

func normalizeRepositoryError(err error) error {
	switch {
	case errors.Is(err, ErrInvalidRequest), errors.Is(err, ErrNotFound),
		errors.Is(err, ErrRevisionConflict), errors.Is(err, ErrUnavailable):
		return err
	default:
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
}
