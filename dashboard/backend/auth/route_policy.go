package auth

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

type Sensitivity string

const (
	SensitivityPublic      Sensitivity = "public"
	SensitivityOperational Sensitivity = "operational"
	SensitivitySensitive   Sensitivity = "sensitive"
	SensitivitySecret      Sensitivity = "secret"
)

type ResourceOwner string

const (
	ResourceOwnerPublic        ResourceOwner = "public"
	ResourceOwnerAuth          ResourceOwner = "auth"
	ResourceOwnerConfig        ResourceOwner = "config"
	ResourceOwnerEvaluation    ResourceOwner = "evaluation"
	ResourceOwnerInference     ResourceOwner = "inference"
	ResourceOwnerObservability ResourceOwner = "observability"
	ResourceOwnerReplay        ResourceOwner = "replay"
	ResourceOwnerFeedback      ResourceOwner = "feedback"
	ResourceOwnerTools         ResourceOwner = "tools"
	ResourceOwnerOpenClaw      ResourceOwner = "openclaw"
	ResourceOwnerML            ResourceOwner = "ml"
	ResourceOwnerWorkflow      ResourceOwner = "workflow"
	ResourceOwnerTenantGrants  ResourceOwner = "tenant_grants"
	ResourceOwnerTenantQuotas  ResourceOwner = "tenant_quotas"
	ResourceOwnerVirtualKeys   ResourceOwner = "virtual_keys"
	ResourceOwnerAuditPolicy   ResourceOwner = "audit_policy"
	ResourceOwnerBreakGlass    ResourceOwner = "breakglass"
)

type AuditMode string

const (
	AuditNone      AuditMode = "none"
	AuditRequired  AuditMode = "required"
	AuditDelegated AuditMode = "delegated"
)

const (
	NoBodyLimit int64 = 0
)

type RoutePolicy struct {
	Method        string
	Permission    string
	AuditMode     AuditMode
	AuditAction   string
	Sensitivity   Sensitivity
	ResourceOwner ResourceOwner
	Public        bool
	Revalidate    bool
	MaxBodyBytes  int64
	ProxyUpstream bool
	MaxAuthAge    time.Duration
}

type RouteContract struct {
	Pattern  string
	Policies []RoutePolicy
}

type RouteLookup int

const (
	RouteNotFound RouteLookup = iota
	RouteMethodNotAllowed
	RouteFound
)

type RoutePolicyResolver interface {
	LookupRoutePolicy(method, path string) (RoutePolicy, RouteLookup)
}

func PublicRoute(pattern string, methods ...string) RouteContract {
	return RouteContract{
		Pattern:  pattern,
		Policies: policiesForMethods("", AuditNone, "", SensitivityPublic, ResourceOwnerPublic, true, false, NoBodyLimit, false, methods...),
	}
}

func Route(pattern string, policies ...RoutePolicy) RouteContract {
	return RouteContract{Pattern: pattern, Policies: policies}
}

func PublicPolicy(method string) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		AuditMode:     AuditNone,
		Sensitivity:   SensitivityPublic,
		ResourceOwner: ResourceOwnerPublic,
		Public:        true,
	}
}

func ReadPolicy(
	method string,
	permission string,
	sensitivity Sensitivity,
	owner ResourceOwner,
) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		Permission:    permission,
		AuditMode:     AuditNone,
		Sensitivity:   sensitivity,
		ResourceOwner: owner,
	}
}

func MutationPolicy(
	method string,
	permission string,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		Permission:    permission,
		AuditMode:     AuditRequired,
		AuditAction:   auditAction,
		Sensitivity:   sensitivity,
		ResourceOwner: owner,
		Revalidate:    true,
		MaxBodyBytes:  maxBodyBytes,
	}
}

func ProtectedRoute(
	pattern string,
	permission string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	methods ...string,
) RouteContract {
	return RouteContract{
		Pattern:  pattern,
		Policies: policiesForMethods(permission, AuditNone, "", sensitivity, owner, false, false, NoBodyLimit, false, methods...),
	}
}

func ProtectedBoundedRoute(
	pattern string,
	permission string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	contract := ProtectedRoute(pattern, permission, sensitivity, owner, methods...)
	for index := range contract.Policies {
		contract.Policies[index].MaxBodyBytes = maxBodyBytes
	}
	return contract
}

func ProtectedMutationRoute(
	pattern string,
	permission string,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	return RouteContract{
		Pattern:  pattern,
		Policies: policiesForMethods(permission, AuditRequired, auditAction, sensitivity, owner, false, true, maxBodyBytes, false, methods...),
	}
}

func ProtectedDelegatedAuditRoute(
	pattern string,
	permission string,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	return RouteContract{
		Pattern: pattern,
		Policies: policiesForMethods(
			permission,
			AuditDelegated,
			auditAction,
			sensitivity,
			owner,
			false,
			true,
			maxBodyBytes,
			false,
			methods...,
		),
	}
}

func BreakGlassMutationRoute(
	pattern string,
	auditAction string,
	maxBodyBytes int64,
	maxAuthAge time.Duration,
	methods ...string,
) RouteContract {
	contract := ProtectedMutationRoute(
		pattern,
		PermBreakGlass,
		auditAction,
		SensitivitySecret,
		ResourceOwnerBreakGlass,
		maxBodyBytes,
		methods...,
	)
	for index := range contract.Policies {
		contract.Policies[index].MaxAuthAge = maxAuthAge
	}
	return contract
}

func ProxyRoute(
	pattern string,
	permission string,
	auditMode AuditMode,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	methods ...string,
) RouteContract {
	return RouteContract{
		Pattern:  pattern,
		Policies: policiesForMethods(permission, auditMode, auditAction, sensitivity, owner, false, false, NoBodyLimit, true, methods...),
	}
}

func ProxyMutationRoute(
	pattern string,
	permission string,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	contract := ProtectedMutationRoute(
		pattern,
		permission,
		auditAction,
		sensitivity,
		owner,
		maxBodyBytes,
		methods...,
	)
	for index := range contract.Policies {
		contract.Policies[index].ProxyUpstream = true
	}
	return contract
}

func policiesForMethods(
	permission string,
	auditMode AuditMode,
	auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	public bool,
	revalidate bool,
	maxBodyBytes int64,
	proxy bool,
	methods ...string,
) []RoutePolicy {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, RoutePolicy{
			Method:        method,
			Permission:    permission,
			AuditMode:     auditMode,
			AuditAction:   auditAction,
			Sensitivity:   sensitivity,
			ResourceOwner: owner,
			Public:        public,
			Revalidate:    revalidate,
			MaxBodyBytes:  maxBodyBytes,
			ProxyUpstream: proxy,
		})
	}
	return policies
}

func mustValidateRouteContract(contract RouteContract) RouteContract {
	contract.Pattern = normalizeRoutePattern(contract.Pattern)
	if err := ValidateRouteContract(contract); err != nil {
		panic(err)
	}
	for index := range contract.Policies {
		contract.Policies[index].Method = strings.ToUpper(strings.TrimSpace(contract.Policies[index].Method))
		contract.Policies[index].Permission = strings.TrimSpace(contract.Policies[index].Permission)
		contract.Policies[index].AuditAction = strings.TrimSpace(contract.Policies[index].AuditAction)
	}
	return contract
}

func ValidateRouteContract(contract RouteContract) error {
	pattern := normalizeRoutePattern(contract.Pattern)
	if pattern == "" || pattern[0] != '/' {
		return errors.New("route pattern must be an absolute path")
	}
	if len(contract.Policies) == 0 {
		return fmt.Errorf("route %q has no method policies", pattern)
	}

	seen := map[string]struct{}{}
	for _, policy := range contract.Policies {
		method := strings.ToUpper(strings.TrimSpace(policy.Method))
		if method == "" {
			return fmt.Errorf("route %q has an empty method", pattern)
		}
		if _, exists := seen[method]; exists {
			return fmt.Errorf("route %q has duplicate policy for %s", pattern, method)
		}
		seen[method] = struct{}{}

		if policy.Public {
			if strings.TrimSpace(policy.Permission) != "" {
				return fmt.Errorf("public route %q %s must not require a permission", pattern, method)
			}
		} else if strings.TrimSpace(policy.Permission) == "" {
			return fmt.Errorf("protected route %q %s has no permission", pattern, method)
		}
		if policy.AuditMode != AuditNone && policy.AuditMode != AuditRequired && policy.AuditMode != AuditDelegated {
			return fmt.Errorf("route %q %s has invalid audit mode %q", pattern, method, policy.AuditMode)
		}
		if policy.AuditMode != AuditNone && strings.TrimSpace(policy.AuditAction) == "" {
			return fmt.Errorf("route %q %s requires an audit action", pattern, method)
		}
		if _, ok := knownSensitivities[policy.Sensitivity]; !ok {
			return fmt.Errorf("route %q %s has invalid sensitivity %q", pattern, method, policy.Sensitivity)
		}
		if _, ok := knownResourceOwners[policy.ResourceOwner]; !ok {
			return fmt.Errorf("route %q %s has invalid resource owner %q", pattern, method, policy.ResourceOwner)
		}
		if policy.Revalidate && policy.Public {
			return fmt.Errorf("public route %q %s cannot request live revalidation", pattern, method)
		}
		if policy.MaxBodyBytes < 0 {
			return fmt.Errorf("route %q %s has a negative body limit", pattern, method)
		}
		if requiredPermission, sensitiveOwner := isolatedOwnerPermissions[policy.ResourceOwner]; sensitiveOwner &&
			policy.Permission != requiredPermission {
			return fmt.Errorf(
				"route %q %s for %s requires isolated permission %q",
				pattern,
				method,
				policy.ResourceOwner,
				requiredPermission,
			)
		}
		if policy.ResourceOwner == ResourceOwnerBreakGlass {
			if policy.AuditMode != AuditRequired || !policy.Revalidate {
				return fmt.Errorf("route %q %s break-glass mutation must be revalidated and audited", pattern, method)
			}
			if policy.MaxAuthAge <= 0 || policy.MaxAuthAge > 15*time.Minute {
				return fmt.Errorf("route %q %s break-glass authorization must expire within 15 minutes", pattern, method)
			}
		} else if policy.MaxAuthAge < 0 {
			return fmt.Errorf("route %q %s has a negative authorization age", pattern, method)
		}
	}
	return nil
}

var isolatedOwnerPermissions = map[ResourceOwner]string{
	ResourceOwnerTenantGrants: PermGrantPublish,
	ResourceOwnerTenantQuotas: PermQuotaPublish,
	ResourceOwnerVirtualKeys:  PermVirtualKeys,
	ResourceOwnerAuditPolicy:  PermAuditPolicy,
	ResourceOwnerBreakGlass:   PermBreakGlass,
}

var knownSensitivities = map[Sensitivity]struct{}{
	SensitivityPublic:      {},
	SensitivityOperational: {},
	SensitivitySensitive:   {},
	SensitivitySecret:      {},
}

var knownResourceOwners = map[ResourceOwner]struct{}{
	ResourceOwnerPublic:        {},
	ResourceOwnerAuth:          {},
	ResourceOwnerConfig:        {},
	ResourceOwnerEvaluation:    {},
	ResourceOwnerInference:     {},
	ResourceOwnerObservability: {},
	ResourceOwnerReplay:        {},
	ResourceOwnerFeedback:      {},
	ResourceOwnerTools:         {},
	ResourceOwnerOpenClaw:      {},
	ResourceOwnerML:            {},
	ResourceOwnerWorkflow:      {},
	ResourceOwnerTenantGrants:  {},
	ResourceOwnerTenantQuotas:  {},
	ResourceOwnerVirtualKeys:   {},
	ResourceOwnerAuditPolicy:   {},
	ResourceOwnerBreakGlass:    {},
}

func normalizeRoutePattern(pattern string) string {
	pattern = strings.TrimSpace(pattern)
	if pattern == "/" {
		return pattern
	}
	if strings.HasSuffix(pattern, "/") {
		return strings.TrimRight(pattern, "/") + "/"
	}
	return strings.TrimRight(pattern, "/")
}

func normalizePolicyPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "/"
	}
	return path
}

func isProtectedNamespace(path string) bool {
	path = normalizePolicyPath(path)
	return strings.HasPrefix(path, "/api/") ||
		path == "/api" ||
		strings.HasPrefix(path, "/embedded/") ||
		path == "/embedded"
}
