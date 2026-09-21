package auth

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Sensitivity classifies what a route exposes or changes. It is declared with
// the route so reviewers and inventory tests can see the classification next
// to the permission that guards it.
type Sensitivity string

const (
	SensitivityPublic      Sensitivity = "public"
	SensitivityOperational Sensitivity = "operational"
	SensitivitySensitive   Sensitivity = "sensitive"
	SensitivitySecret      Sensitivity = "secret"
)

// ResourceOwner names the security domain a route belongs to. Independent
// domains keep their permissions independently grantable and revocable.
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
)

// AuditMode says who writes the audit row for a route. AuditRequired routes are
// audited by the authentication middleware from the declared action;
// AuditDelegated routes are audited by their handler with request-specific
// detail; AuditNone routes are reads or probes that leave no audit trail.
type AuditMode string

const (
	AuditNone      AuditMode = "none"
	AuditRequired  AuditMode = "required"
	AuditDelegated AuditMode = "delegated"
)

// NoBodyLimit leaves the request body unbounded by the authorization layer.
const NoBodyLimit int64 = 0

// RoutePolicy is the authorization declaration for one method of one route.
type RoutePolicy struct {
	Method        string
	Permissions   []string
	AuditMode     AuditMode
	AuditAction   string
	Sensitivity   Sensitivity
	ResourceOwner ResourceOwner
	// Public routes need no session. SessionOnly routes need a live session but
	// no specific permission. Every other route needs every listed permission.
	Public      bool
	SessionOnly bool
	// Revalidate re-resolves the session and permissions after the bounded
	// body has been read and exposes a revalidator for the handler to call
	// again immediately before its side effect.
	Revalidate   bool
	MaxBodyBytes int64
	// StreamBody keeps a revalidated mutation's body streaming through a
	// bounded reader instead of buffering it, for uploads too large to hold.
	// The handler must call RejectRevokedMutation before its side effect.
	StreamBody bool
}

// RouteContract binds one ServeMux pattern to the policies of every method it
// serves. A pattern without a policy for a method is not served for that
// method.
type RouteContract struct {
	Pattern  string
	Policies []RoutePolicy
}

// RouteLookup is the result of resolving a request against the registry.
type RouteLookup int

const (
	RouteNotFound RouteLookup = iota
	RouteMethodNotAllowed
	RouteFound
)

// RoutePolicyResolver answers which policy, if any, governs a request.
type RoutePolicyResolver interface {
	LookupRoutePolicy(method, path string) (RoutePolicy, RouteLookup)
}

// PublicRoute serves the listed methods without authentication.
func PublicRoute(pattern string, methods ...string) RouteContract {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, PublicPolicy(method))
	}
	return RouteContract{Pattern: pattern, Policies: policies}
}

// SessionRoute requires a live session but no particular permission.
func SessionRoute(pattern string, sensitivity Sensitivity, owner ResourceOwner, methods ...string) RouteContract {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, RoutePolicy{
			Method:        method,
			AuditMode:     AuditNone,
			Sensitivity:   sensitivity,
			ResourceOwner: owner,
			SessionOnly:   true,
		})
	}
	return RouteContract{Pattern: pattern, Policies: policies}
}

// Route assembles a contract from explicit per-method policies.
func Route(pattern string, policies ...RoutePolicy) RouteContract {
	return RouteContract{Pattern: pattern, Policies: policies}
}

// PublicPolicy is one unauthenticated method.
func PublicPolicy(method string) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		AuditMode:     AuditNone,
		Sensitivity:   SensitivityPublic,
		ResourceOwner: ResourceOwnerPublic,
		Public:        true,
	}
}

// ReadPolicy is a side-effect-free method guarded by one permission.
func ReadPolicy(method, permission string, sensitivity Sensitivity, owner ResourceOwner) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		Permissions:   []string{permission},
		AuditMode:     AuditNone,
		Sensitivity:   sensitivity,
		ResourceOwner: owner,
	}
}

// BoundedPolicy is a side-effect-free method that carries a request body,
// such as a preview or probe. The body is bounded but not audited.
func BoundedPolicy(method, permission string, sensitivity Sensitivity, owner ResourceOwner, maxBodyBytes int64) RoutePolicy {
	policy := ReadPolicy(method, permission, sensitivity, owner)
	policy.MaxBodyBytes = maxBodyBytes
	return policy
}

// MutationPolicy is a method with a privileged side effect. It is bounded,
// revalidated after the body is read, and audited from the declared action.
func MutationPolicy(
	method, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
) RoutePolicy {
	return RoutePolicy{
		Method:        method,
		Permissions:   []string{permission},
		AuditMode:     AuditRequired,
		AuditAction:   auditAction,
		Sensitivity:   sensitivity,
		ResourceOwner: owner,
		Revalidate:    true,
		MaxBodyBytes:  maxBodyBytes,
	}
}

// StreamingMutationPolicy is a MutationPolicy for large uploads: the body is
// bounded but streamed, so the commit-time revalidation in the handler is the
// barrier rather than a post-body recheck in the middleware.
func StreamingMutationPolicy(
	method, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
) RoutePolicy {
	policy := MutationPolicy(method, permission, auditAction, sensitivity, owner, maxBodyBytes)
	policy.StreamBody = true
	return policy
}

// DelegatedMutationPolicy is a MutationPolicy whose handler writes its own
// audit row with request-specific detail.
func DelegatedMutationPolicy(
	method, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
) RoutePolicy {
	policy := MutationPolicy(method, permission, auditAction, sensitivity, owner, maxBodyBytes)
	policy.AuditMode = AuditDelegated
	return policy
}

// ProtectedRoute guards every listed method with the same read permission.
func ProtectedRoute(pattern, permission string, sensitivity Sensitivity, owner ResourceOwner, methods ...string) RouteContract {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, ReadPolicy(method, permission, sensitivity, owner))
	}
	return RouteContract{Pattern: pattern, Policies: policies}
}

// ProtectedBoundedRoute guards body-carrying, side-effect-free methods.
func ProtectedBoundedRoute(
	pattern, permission string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, BoundedPolicy(method, permission, sensitivity, owner, maxBodyBytes))
	}
	return RouteContract{Pattern: pattern, Policies: policies}
}

// ProtectedMutationRoute guards privileged side effects on every listed method.
func ProtectedMutationRoute(
	pattern, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	policies := make([]RoutePolicy, 0, len(methods))
	for _, method := range methods {
		policies = append(policies, MutationPolicy(method, permission, auditAction, sensitivity, owner, maxBodyBytes))
	}
	return RouteContract{Pattern: pattern, Policies: policies}
}

// ProtectedStreamingMutationRoute is ProtectedMutationRoute for bounded
// uploads that must not be buffered by the authorization layer.
func ProtectedStreamingMutationRoute(
	pattern, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	contract := ProtectedMutationRoute(pattern, permission, auditAction, sensitivity, owner, maxBodyBytes, methods...)
	for index := range contract.Policies {
		contract.Policies[index].StreamBody = true
	}
	return contract
}

// ProtectedDelegatedMutationRoute is ProtectedMutationRoute for handlers that
// write their own audit rows.
func ProtectedDelegatedMutationRoute(
	pattern, permission, auditAction string,
	sensitivity Sensitivity,
	owner ResourceOwner,
	maxBodyBytes int64,
	methods ...string,
) RouteContract {
	contract := ProtectedMutationRoute(pattern, permission, auditAction, sensitivity, owner, maxBodyBytes, methods...)
	for index := range contract.Policies {
		contract.Policies[index].AuditMode = AuditDelegated
	}
	return contract
}

// AlsoRequiring adds permissions the method must hold in addition to its
// primary permission.
func (p RoutePolicy) AlsoRequiring(permissions ...string) RoutePolicy {
	p.Permissions = append(append([]string(nil), p.Permissions...), permissions...)
	return p
}

func mustValidateRouteContract(contract RouteContract) RouteContract {
	contract.Pattern = normalizeRoutePattern(contract.Pattern)
	if err := ValidateRouteContract(contract); err != nil {
		panic(err)
	}
	for index := range contract.Policies {
		policy := &contract.Policies[index]
		policy.Method = strings.ToUpper(strings.TrimSpace(policy.Method))
		policy.AuditAction = strings.TrimSpace(policy.AuditAction)
		for permissionIndex := range policy.Permissions {
			policy.Permissions[permissionIndex] = strings.TrimSpace(policy.Permissions[permissionIndex])
		}
	}
	return contract
}

// ValidateRouteContract rejects contracts that could be served without a
// complete authorization declaration.
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
		if err := validateRoutePolicy(pattern, method, policy); err != nil {
			return err
		}
	}
	return nil
}

func validateRoutePolicy(pattern, method string, policy RoutePolicy) error {
	if _, ok := knownSensitivities[policy.Sensitivity]; !ok {
		return fmt.Errorf("route %q %s has invalid sensitivity %q", pattern, method, policy.Sensitivity)
	}
	if _, ok := knownResourceOwners[policy.ResourceOwner]; !ok {
		return fmt.Errorf("route %q %s has invalid resource owner %q", pattern, method, policy.ResourceOwner)
	}
	if policy.AuditMode != AuditNone && policy.AuditMode != AuditRequired && policy.AuditMode != AuditDelegated {
		return fmt.Errorf("route %q %s has invalid audit mode %q", pattern, method, policy.AuditMode)
	}
	if policy.AuditMode != AuditNone && strings.TrimSpace(policy.AuditAction) == "" {
		return fmt.Errorf("route %q %s requires an audit action", pattern, method)
	}
	if policy.MaxBodyBytes < 0 {
		return fmt.Errorf("route %q %s has a negative body limit", pattern, method)
	}
	if policy.StreamBody && (!policy.Revalidate || policy.MaxBodyBytes == 0) {
		return fmt.Errorf("route %q %s streams a body without a revalidated bound", pattern, method)
	}
	for _, permission := range policy.Permissions {
		if strings.TrimSpace(permission) == "" {
			return fmt.Errorf("route %q %s has an empty permission", pattern, method)
		}
	}
	switch {
	case policy.Public && policy.SessionOnly:
		return fmt.Errorf("route %q %s cannot be both public and session-only", pattern, method)
	case policy.Public || policy.SessionOnly:
		if len(policy.Permissions) != 0 {
			return fmt.Errorf("route %q %s must not require a permission", pattern, method)
		}
		if policy.Revalidate || policy.AuditMode != AuditNone {
			return fmt.Errorf("route %q %s cannot request revalidation or audit without a permission", pattern, method)
		}
	case len(policy.Permissions) == 0:
		return fmt.Errorf("protected route %q %s has no permission", pattern, method)
	}
	return nil
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
}

func normalizeRoutePattern(pattern string) string {
	pattern = strings.TrimSpace(pattern)
	if pattern == "/" {
		return pattern
	}
	if strings.HasSuffix(pattern, "/") {
		return strings.TrimRight(pattern, "/") + "/"
	}
	return pattern
}

func normalizePolicyPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "/"
	}
	return path
}

// isProtectedNamespace marks the API and embedded-service namespaces where an
// unregistered route is denied rather than served by a fallback handler.
func isProtectedNamespace(path string) bool {
	path = normalizePolicyPath(path)
	return path == "/api" || strings.HasPrefix(path, "/api/") ||
		path == "/embedded" || strings.HasPrefix(path, "/embedded/")
}

// optionsPolicy lets an authenticated preflight reach handlers that answer it
// themselves without granting anything to an anonymous caller.
func optionsPolicy(owner ResourceOwner) RoutePolicy {
	return RoutePolicy{
		Method:        http.MethodOptions,
		AuditMode:     AuditNone,
		Sensitivity:   SensitivityOperational,
		ResourceOwner: owner,
		SessionOnly:   true,
	}
}
