package accessruntime

import (
	"context"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Authorize evaluates a target against the authenticated immutable projection
// and atomically rechecks every publication, policy, key, and resource guard.
func (r *Runtime) Authorize(ctx context.Context, request AuthorizationRequest) (Authorization, error) {
	state, preconditions, result, err := r.prepareTarget(ctx, request.Session, request.Target)
	if err != nil || result.Disposition != quotaruntime.AdmissionAllowed {
		return Authorization{Result: accessResult(result), Target: request.Target}, err
	}
	checked, err := r.engine.CheckAccess(ctx, quotaruntime.AccessCheckRequest{
		Partition: state.tenant.QuotaPartition, Preconditions: preconditions,
	})
	if err != nil {
		return Authorization{
			Result: quotaruntime.AccessCheckResult{
				Disposition: quotaruntime.AdmissionUnavailable,
				Reason:      "atomic_access_check_failed",
			},
			Target: request.Target,
		}, err
	}
	if !checked.Allowed() {
		return Authorization{Result: checked, Target: request.Target}, nil
	}
	return Authorization{
		Result: checked, Tenant: cloneTenantContext(state.tenant), Target: request.Target,
	}, nil
}

// Discover performs one atomic guard and returns only immutable resources
// allowed by the verified projection. It is the data source for /v1/models.
func (r *Runtime) Discover(ctx context.Context, request DiscoveryRequest) (Discovery, error) {
	catalog, err := r.DiscoverCatalog(ctx, CatalogDiscoveryRequest{
		Session: request.Session,
		Queries: []DiscoveryQuery{{
			ResourceType: request.ResourceType,
			Permission:   request.Permission,
		}},
	})
	return Discovery{
		Result: catalog.Result, Tenant: catalog.Tenant,
		ResourceIDs: catalog.Resources[request.ResourceType],
	}, err
}

// DiscoverCatalog returns all requested resource classes from one verified,
// immutable projection and one atomic policy guard.
func (r *Runtime) DiscoverCatalog(ctx context.Context, request CatalogDiscoveryRequest) (CatalogDiscovery, error) {
	if len(request.Queries) == 0 {
		return CatalogDiscovery{}, fmt.Errorf("at least one discovery query is required")
	}
	queries := make(map[accesscontrol.GrantResourceType]accesscontrol.GrantPermission, len(request.Queries))
	for _, query := range request.Queries {
		if !query.ResourceType.Valid() || !query.Permission.Valid() {
			return CatalogDiscovery{}, fmt.Errorf("discovery resource type or permission is invalid")
		}
		if permission, exists := queries[query.ResourceType]; exists && permission != query.Permission {
			return CatalogDiscovery{}, fmt.Errorf("resource type %q has conflicting discovery permissions", query.ResourceType)
		}
		queries[query.ResourceType] = query.Permission
	}
	state, err := r.sessionState(request.Session)
	if err != nil {
		return CatalogDiscovery{
			Result: quotaruntime.AccessCheckResult{
				Disposition: quotaruntime.AdmissionUnavailable,
				Reason:      "invalid_session",
			},
		}, err
	}
	barrierResult, err := r.checkDelegationBarriers(ctx, state.delegation)
	if err != nil || barrierResult.Disposition != quotaruntime.AdmissionAllowed {
		return CatalogDiscovery{Result: accessResult(barrierResult)}, err
	}
	preconditions := append([]quotaruntime.AdmissionPrecondition(nil), state.preconditions...)
	resources := make(map[accesscontrol.GrantResourceType][]string, len(queries))
	seen := make(map[accesscontrol.GrantResourceType]map[string]struct{}, len(queries))
	guarded := make(map[string]struct{})
	for resourceType := range queries {
		resources[resourceType] = []string{}
		seen[resourceType] = make(map[string]struct{})
	}
	for _, grant := range state.grants {
		permission, requested := queries[grant.ResourceType]
		if !requested || grant.Permission != permission || grant.Effect != accesscontrol.GrantEffectAllow {
			continue
		}
		if evaluateGrants(state.grants, Target{
			ResourceType: grant.ResourceType,
			ResourceID:   accesscontrol.ResourceID(grant.ResourceID),
			Permission:   grant.Permission,
		}) != accesscontrol.AccessDecisionAllow {
			continue
		}
		if _, exists := seen[grant.ResourceType][grant.ResourceID]; exists {
			continue
		}
		seen[grant.ResourceType][grant.ResourceID] = struct{}{}
		resources[grant.ResourceType] = append(resources[grant.ResourceType], grant.ResourceID)
		guardKey := string(grant.ResourceType) + "\x00" + grant.ResourceID
		if _, exists := guarded[guardKey]; !exists {
			precondition, guardErr := resourceDenyPrecondition(r.keyPrefix, state.tenant.QuotaPartition, Target{
				ResourceType: grant.ResourceType, ResourceID: accesscontrol.ResourceID(grant.ResourceID), Permission: grant.Permission,
			})
			if guardErr != nil {
				return CatalogDiscovery{Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionUnavailable, Reason: "resource_precondition_invalid"}}, guardErr
			}
			preconditions = append(preconditions, precondition)
			guarded[guardKey] = struct{}{}
		}
	}
	checked, err := r.engine.CheckAccess(ctx, quotaruntime.AccessCheckRequest{
		Partition: state.tenant.QuotaPartition, Preconditions: preconditions,
	})
	if err != nil {
		return CatalogDiscovery{Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionUnavailable, Reason: "atomic_access_check_failed"}}, err
	}
	if !checked.Allowed() {
		return CatalogDiscovery{Result: checked}, nil
	}
	for resourceType := range resources {
		sort.Strings(resources[resourceType])
	}
	return CatalogDiscovery{
		Result: checked, Tenant: cloneTenantContext(state.tenant), Resources: resources,
	}, nil
}

func (r *Runtime) prepareTarget(
	ctx context.Context,
	session Session,
	target Target,
) (*sessionState, []quotaruntime.AdmissionPrecondition, quotaruntime.AdmissionResult, error) {
	if err := target.validate(); err != nil {
		return nil, nil, quotaruntime.AdmissionResult{}, fmt.Errorf("target: %w", err)
	}
	state, err := r.sessionState(session)
	if err != nil {
		return nil, nil, unavailable("invalid_session"), err
	}
	barrierResult, err := r.checkDelegationBarriers(ctx, state.delegation)
	if err != nil || barrierResult.Disposition != quotaruntime.AdmissionAllowed {
		return nil, nil, barrierResult, err
	}
	if evaluateGrants(state.grants, target) != accesscontrol.AccessDecisionAllow {
		return nil, nil, forbidden("resource_not_found"), nil
	}
	resourceGuard, err := resourceDenyPrecondition(r.keyPrefix, state.tenant.QuotaPartition, target)
	if err != nil {
		return nil, nil, unavailable("resource_precondition_invalid"), fmt.Errorf("%w: %w", ErrRuntimeCorrupt, err)
	}
	preconditions := make([]quotaruntime.AdmissionPrecondition, 0, len(state.preconditions)+1)
	preconditions = append(preconditions, state.preconditions...)
	preconditions = append(preconditions, resourceGuard)
	return state, preconditions, quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed}, nil
}

func resourceDenyPrecondition(
	keyPrefix string,
	partition string,
	target Target,
) (quotaruntime.AdmissionPrecondition, error) {
	keys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(keyPrefix, partition)
	if err != nil {
		return quotaruntime.AdmissionPrecondition{}, err
	}
	return quotaruntime.AdmissionPrecondition{
		Key:     keys.Deny(string(target.ResourceType), string(target.ResourceID)),
		Kind:    quotaruntime.AdmissionCheckKeyAbsent,
		Failure: quotaruntime.AdmissionForbidden,
		Reason:  "resource_denied",
	}, nil
}

func evaluateGrants(grants []accessprojection.Grant, target Target) accesscontrol.AccessDecision {
	decision := accesscontrol.AccessDecisionDeny
	for _, grant := range grants {
		if grant.ResourceType != target.ResourceType ||
			grant.ResourceID != string(target.ResourceID) ||
			grant.Permission != target.Permission {
			continue
		}
		if grant.Effect == accesscontrol.GrantEffectDeny {
			return accesscontrol.AccessDecisionDeny
		}
		if grant.Effect == accesscontrol.GrantEffectAllow {
			decision = accesscontrol.AccessDecisionAllow
		}
	}
	return decision
}
