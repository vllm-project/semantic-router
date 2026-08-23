package managementserver

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const accessCheckPath = managementapi.BasePath + "/access:check"

var routingContextETagPattern = regexp.MustCompile(`^"routing-context:([1-9][0-9]*)"$`)

type AccessReadRoutes struct {
	service       AccessReadService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewAccessReadRoutes(options AccessReadRoutesOptions) (*AccessReadRoutes, error) {
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil {
		return nil, errors.New("access read routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &AccessReadRoutes{
		service: options.Service, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range accessReadHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("access read operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[accessReadOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *AccessReadRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Access read routes and mux are required")
	}
	for _, contract := range accessReadHTTPContracts() {
		mux.Handle(string(contract.method)+" "+contract.path, routes)
	}
}

func (routes *AccessReadRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return accessmanagement.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *AccessReadRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.Method == http.MethodPost && request.URL.Path == accessCheckPath {
		routes.check(response, request, requestID)
		return
	}
	subject, action, operationPath, ok := accessReadSubjectRequest(request)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	inspection, err := routes.service.Inspect(request.Context(), namespaceID, subject)
	if err != nil {
		writeAccessReadError(response, err, requestID)
		return
	}
	operation := routes.operation(managementapi.HTTPMethod(request.Method), operationPath)
	targets, err := accessReadTargets(namespaceID, inspection, action)
	if err != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID, operation, targets, nil) {
		return
	}
	switch action {
	case "effective-policy":
		policy, err := routes.service.GetEffectivePolicy(request.Context(), namespaceID, subject)
		if err != nil {
			writeAccessReadError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, effectivePolicyDTO(policy), requestID)
	case "quota":
		quota, err := routes.service.GetQuota(request.Context(), namespaceID, subject)
		if err != nil {
			writeAccessReadError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, effectiveQuotaDTO(quota), requestID)
	case "routing-context":
		if request.Method == http.MethodGet {
			value, err := routes.service.GetRoutingContext(request.Context(), namespaceID, subject)
			if err != nil {
				writeAccessReadError(response, err, requestID)
				return
			}
			writeRoutingContext(response, value, requestID)
			return
		}
		routes.putRoutingContext(response, request, requestID, namespaceID, session, subject)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AccessReadRoutes) putRoutingContext(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
	session managementauth.AuthenticatedSession,
	subject accessmanagement.Subject,
) {
	revision, ok := requireRoutingContextRevision(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingContextPutRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	if body.Values == nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing context values must be an object.", requestID)
		return
	}
	value, err := routes.service.UpdateRoutingContext(request.Context(), accessmanagement.UpdateRoutingContextRequest{
		NamespaceID: namespaceID, Subject: subject, ExpectedRevision: revision, Values: routingClaimValues(body.Values),
		Actor: accessmanagement.Actor{
			PrincipalID: session.Session.PrincipalID,
			ActorChain:  []string{session.Session.PrincipalID}, RequestID: requestID, SourceIP: directRequestIP(request),
		},
	})
	if err != nil {
		writeAccessReadError(response, err, requestID)
		return
	}
	writeRoutingContext(response, value, requestID)
}

func (routes *AccessReadRoutes) check(response http.ResponseWriter, request *http.Request, requestID string) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.AccessCheckRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	subject := accessmanagement.Subject{Kind: accesscontrol.SubjectKind(body.Subject.Type), ID: body.Subject.ID}
	if subject.Validate() != nil || !canonicalUUID(subject.ID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "A valid typed subject is required.", requestID)
		return
	}
	inspection, err := routes.service.Inspect(request.Context(), namespaceID, subject)
	if err != nil {
		writeAccessReadError(response, err, requestID)
		return
	}
	subjectTarget, err := accessReadSubjectTarget(namespaceID, inspection)
	if err != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	resourceKind := accesscontrol.ScopeResourceModel
	resourceType := accesscontrol.GrantResourceType(body.Resource.Type)
	permission := accesscontrol.GrantPermission(body.Permission)
	if resourceType == accesscontrol.GrantResourceEntrypoint {
		resourceKind = accesscontrol.ScopeResourceEntrypoint
	} else if resourceType != accesscontrol.GrantResourceModel {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "A valid access-check resource is required.", requestID)
		return
	}
	if (accesscontrol.GrantResource{Type: resourceType, ID: accesscontrol.ResourceID(body.Resource.ID)}).Validate() != nil ||
		!permission.Valid() {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "A valid access-check resource and permission are required.", requestID)
		return
	}
	resourceTarget := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), resourceKind, accesscontrol.ResourceID(body.Resource.ID))}
	operation := routes.operation(managementapi.MethodPOST, accessCheckPath)
	conditions := map[string]bool{
		"routing_context_override_requested": body.RoutingContextOverride != nil,
		"entrypoint_topology_requested":      false, "internal_usage_dimensions_requested": false,
	}
	if !routes.authorize(response, request, requestID, session, namespaceID, operation,
		map[string][]accesscontrol.ScopedTarget{"subject": {subjectTarget}, "resource": {resourceTarget}}, conditions) {
		return
	}
	override := map[string]routingsnapshot.ClaimValue(nil)
	if body.RoutingContextOverride != nil {
		if *body.RoutingContextOverride == nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing context override must be an object.", requestID)
			return
		}
		override = routingClaimValues(*body.RoutingContextOverride)
	}
	result, err := routes.service.Check(request.Context(), accessmanagement.AccessCheckRequest{
		NamespaceID: namespaceID, Subject: subject,
		Resource:   accesscontrol.GrantResource{Type: resourceType, ID: accesscontrol.ResourceID(body.Resource.ID)},
		Permission: permission, Path: body.Path, Override: override,
		OverridePresent: body.RoutingContextOverride != nil,
	})
	if err != nil {
		writeAccessReadError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, accessCheckDTO(result), requestID)
}

func (routes *AccessReadRoutes) authenticate(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (string, managementauth.AuthenticatedSession, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != namespaceID || !canonicalUUID(session.Session.PrincipalID) {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *AccessReadRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget,
	conditions map[string]bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets, Conditions: conditions,
	})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	} else {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}

func (routes *AccessReadRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[accessReadOperationKey(method, path)]
}

type accessReadHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func accessReadHTTPContracts() []accessReadHTTPContract {
	contracts := []accessReadHTTPContract{{method: managementapi.MethodPOST, path: accessCheckPath}}
	for _, item := range []struct {
		base, parameter string
	}{
		{managementapi.BasePath + "/users", "userId"},
		{managementapi.BasePath + "/teams", "teamId"},
		{managementapi.BasePath + "/api-keys", "keyId"},
	} {
		for _, action := range []string{"effective-policy", "routing-context", "quota"} {
			path := item.base + "/{" + item.parameter + "}/" + action
			contracts = append(contracts, accessReadHTTPContract{method: managementapi.MethodGET, path: path})
			if action == "routing-context" {
				contracts = append(contracts, accessReadHTTPContract{method: managementapi.MethodPUT, path: path})
			}
		}
	}
	return contracts
}

func accessReadOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + "\x00" + path
}

func accessReadSubjectRequest(request *http.Request) (accessmanagement.Subject, string, string, bool) {
	var kind accesscontrol.SubjectKind
	var parameter, base string
	switch {
	case request.PathValue("userId") != "":
		kind, parameter, base = accesscontrol.SubjectKindUser, "userId", managementapi.BasePath+"/users"
	case request.PathValue("teamId") != "":
		kind, parameter, base = accesscontrol.SubjectKindTeam, "teamId", managementapi.BasePath+"/teams"
	case request.PathValue("keyId") != "":
		kind, parameter, base = accesscontrol.SubjectKindAPIKey, "keyId", managementapi.BasePath+"/api-keys"
	default:
		return accessmanagement.Subject{}, "", "", false
	}
	id := request.PathValue(parameter)
	if !canonicalUUID(id) {
		return accessmanagement.Subject{}, "", "", false
	}
	action := request.URL.Path[strings.LastIndexByte(request.URL.Path, '/')+1:]
	if action != "effective-policy" && action != "routing-context" && action != "quota" {
		return accessmanagement.Subject{}, "", "", false
	}
	return accessmanagement.Subject{Kind: kind, ID: id}, action, base + "/{" + parameter + "}/" + action, true
}

func accessReadTargets(
	namespaceID string,
	inspection accessmanagement.AuthorizationContext,
	action string,
) (map[string][]accesscontrol.ScopedTarget, error) {
	target, err := accessReadSubjectTarget(namespaceID, inspection)
	if err != nil {
		return nil, err
	}
	operand := map[accesscontrol.SubjectKind]string{
		accesscontrol.SubjectKindUser: "user",
		accesscontrol.SubjectKindTeam: "team", accesscontrol.SubjectKindAPIKey: "key",
	}[inspection.Subject.Kind]
	if operand == "" {
		return nil, accessmanagement.ErrInvalidRequest
	}
	result := map[string][]accesscontrol.ScopedTarget{operand: {target}}
	if action != "quota" {
		return result, nil
	}
	bindingTargets := make([]accesscontrol.ScopedTarget, 0, len(inspection.RateBindings))
	for _, binding := range inspection.RateBindings {
		source, sourceErr := accessReadPlainSubjectTarget(namespaceID, binding.Subject)
		if sourceErr != nil {
			return nil, sourceErr
		}
		bindingTargets = append(bindingTargets, accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceRateLimitBinding,
			accesscontrol.ResourceID(binding.BindingID)), Ancestors: []accesscontrol.Scope{source.Scope}})
	}
	if len(bindingTargets) == 0 {
		bindingTargets = append(bindingTargets, target)
	}
	result["all_returned_bindings"] = bindingTargets
	return result, nil
}

func accessReadSubjectTarget(namespaceID string, inspection accessmanagement.AuthorizationContext) (accesscontrol.ScopedTarget, error) {
	target, err := accessReadPlainSubjectTarget(namespaceID, inspection.Subject)
	if err != nil {
		return accesscontrol.ScopedTarget{}, err
	}
	if inspection.Subject.Kind != accesscontrol.SubjectKindAPIKey {
		return target, nil
	}
	for _, ancestor := range inspection.Ancestors {
		value, ancestorErr := accessReadPlainSubjectTarget(namespaceID, ancestor)
		if ancestorErr != nil {
			return accesscontrol.ScopedTarget{}, ancestorErr
		}
		target.Ancestors = append(target.Ancestors, value.Scope)
	}
	return target, target.Validate()
}

func accessReadPlainSubjectTarget(namespaceID string, subject accessmanagement.Subject) (accesscontrol.ScopedTarget, error) {
	namespace := accesscontrol.NamespaceID(namespaceID)
	switch subject.Kind {
	case accesscontrol.SubjectKindUser:
		return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(namespace, accesscontrol.UserID(subject.ID))}, nil
	case accesscontrol.SubjectKindTeam:
		return accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(namespace, accesscontrol.TeamID(subject.ID))}, nil
	case accesscontrol.SubjectKindAPIKey:
		return accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(namespace,
			accesscontrol.ScopeResourceAPIKey, accesscontrol.ResourceID(subject.ID))}, nil
	default:
		return accesscontrol.ScopedTarget{}, accessmanagement.ErrInvalidRequest
	}
}

func requireRoutingContextRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	match := routingContextETagPattern.FindStringSubmatch(request.Header.Get(managementapi.HeaderIfMatch))
	if match == nil {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "A routing-context ETag is required.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_revision", "The routing-context ETag is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func writeRoutingContext(response http.ResponseWriter, context accessmanagement.RoutingContext, requestID string) {
	response.Header().Set(managementapi.HeaderETag, fmt.Sprintf(`"routing-context:%d"`, context.Revision))
	writeProviderJSON(response, http.StatusOK, routingContextDTO(context), requestID)
}

func writeAccessReadError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, accessmanagement.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "The access request is invalid.", requestID)
	case errors.Is(err, accessmanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case errors.Is(err, accessmanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "Resource changed. Refresh and retry.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "access_state_unavailable", "Access state is unavailable.", requestID)
	}
}

func effectivePolicyDTO(value accessmanagement.EffectivePolicy) managementapi.EffectivePolicy {
	return managementapi.EffectivePolicy{
		Subject:  policySubjectDTO(value.Subject),
		Revision: safeRevision(value.DesiredRevision), AppliedRevision: safeRevision(value.AppliedRevision),
		Access: managementapi.EffectiveAccess{Grants: effectiveGrantDTOs(value.Access)}, Quota: effectiveQuotaDTO(value.Quota),
	}
}

func effectiveGrantDTOs(values []accessmanagement.GrantView) []managementapi.EffectiveGrant {
	type key struct{ resourceType, resourceID, effect, sourceType, sourceID, bindingID string }
	grouped := make(map[key]map[string]struct{})
	for _, value := range values {
		item := key{
			resourceType: string(value.Grant.ResourceType), resourceID: value.Grant.ResourceID,
			effect: string(value.Grant.Effect), sourceType: string(value.Source.Kind), sourceID: value.Source.ID,
			bindingID: value.Grant.BindingID,
		}
		if grouped[item] == nil {
			grouped[item] = make(map[string]struct{})
		}
		grouped[item][string(value.Grant.Permission)] = struct{}{}
	}
	keys := make([]key, 0, len(grouped))
	for item := range grouped {
		keys = append(keys, item)
	}
	sort.Slice(keys, func(i, j int) bool {
		left, right := keys[i], keys[j]
		return left.resourceType+"\x00"+left.resourceID+"\x00"+left.effect+"\x00"+left.bindingID <
			right.resourceType+"\x00"+right.resourceID+"\x00"+right.effect+"\x00"+right.bindingID
	})
	result := make([]managementapi.EffectiveGrant, 0, len(keys))
	for _, item := range keys {
		permissions := make([]string, 0, len(grouped[item]))
		for permission := range grouped[item] {
			permissions = append(permissions, permission)
		}
		sort.Strings(permissions)
		result = append(result, managementapi.EffectiveGrant{
			ResourceType: item.resourceType, ResourceID: item.resourceID,
			Permissions: permissions, Effect: item.effect, Source: managementapi.GrantSource{
				SubjectType: item.sourceType, SubjectID: item.sourceID, BindingID: item.bindingID,
			},
		})
	}
	return result
}

func effectiveQuotaDTO(value accessmanagement.EffectiveQuota) managementapi.EffectiveQuota {
	result := managementapi.EffectiveQuota{
		Meters:         make([]managementapi.QuotaMeter, 0, len(value.Meters)),
		LimitingRuleID: value.LimitingRuleID, UnknownUsageFences: append([]string(nil), value.FenceIDs...), AsOf: value.AsOf,
	}
	for _, view := range value.Meters {
		meter := view.Meter
		remaining := decimalPointer(meter.Remaining)
		overage := decimalPointer(meter.Overage)
		window := ""
		if view.Rule.Rule.Window > 0 {
			window = policymanagement.ISODuration(view.Rule.Rule.Window).String()
		} else if view.Rule.Rule.CalendarPeriod != "" {
			window = string(view.Rule.Rule.CalendarPeriod)
		}
		result.Meters = append(result.Meters, managementapi.QuotaMeter{
			PolicyID: view.Binding.PolicyID, RuleID: meter.RuleID, BindingID: meter.BindingID,
			Source:       managementapi.GrantSource{SubjectType: string(view.Source.Kind), SubjectID: view.Source.ID, BindingID: meter.BindingID},
			CounterOwner: view.Source.ID, Metric: string(meter.Metric), Algorithm: string(meter.Algorithm),
			Accounting: string(meter.Accounting), Enforcement: string(meter.Enforcement), Window: window,
			Currency: meter.Currency, Limit: managementapi.DecimalQuantity(meter.Limit), Used: managementapi.DecimalQuantity(meter.Used),
			Remaining: remaining, Overage: overage, ResetAt: meter.ResetAt, Completeness: string(meter.Completeness),
			KnownDispatches:      managementapi.WholeQuantity(meter.KnownDispatches),
			IncompleteDispatches: managementapi.WholeQuantity(meter.IncompleteDispatches),
			CapacityState:        string(meter.CapacityState), ActiveFenceIDs: append([]string(nil), meter.ActiveFenceIDs...),
			Freshness: managementapi.MeterFreshness{Source: "valkey", AsOf: value.AsOf},
		})
	}
	return result
}

func decimalPointer(value *string) *managementapi.DecimalQuantity {
	if value == nil {
		return nil
	}
	converted := managementapi.DecimalQuantity(*value)
	return &converted
}

func routingContextDTO(value accessmanagement.RoutingContext) managementapi.RoutingContext {
	result := managementapi.RoutingContext{
		Subject:  policySubjectDTO(value.Subject),
		Revision: safeRevision(value.Revision), SchemaRevision: safeRevisionAllowZero(value.SchemaRevision),
		Stored:    make([]managementapi.RoutingContextStoredValue, 0, len(value.Stored)),
		Effective: make([]managementapi.RoutingContextEffectiveValue, 0, len(value.Effective)),
	}
	for _, claim := range value.Stored {
		result.Stored = append(result.Stored, managementapi.RoutingContextStoredValue{
			Name: claim.Name, Value: routingClaimValueDTO(claim.Value),
			Revision: safeRevision(claim.Revision), UpdatedAt: claim.UpdatedAt,
		})
	}
	for _, claim := range value.Effective {
		updated := claim.UpdatedAt
		var updatedAt *time.Time
		if !updated.IsZero() {
			updatedAt = &updated
		}
		result.Effective = append(result.Effective, managementapi.RoutingContextEffectiveValue{
			Name:  claim.Name,
			Value: routingClaimValueDTO(claim.Value), Source: managementapi.RoutingContextSource{SubjectType: string(claim.Source.Kind), SubjectID: claim.Source.ID},
			Revision: safeRevisionAllowZero(claim.Revision), UpdatedAt: updatedAt,
		})
	}
	return result
}

func accessCheckDTO(value accessmanagement.AccessCheckResult) managementapi.AccessCheckResponse {
	context := routingContextDTO(accessmanagement.RoutingContext{Effective: value.RoutingContext}).Effective
	return managementapi.AccessCheckResponse{
		Subject:    policySubjectDTO(value.Subject),
		Resource:   managementapi.AccessCheckResource{Type: string(value.Resource.Type), ID: string(value.Resource.ID)},
		Permission: string(value.Permission), Decision: string(value.Decision), MatchedGrants: effectiveGrantDTOs(value.Matched),
		RoutingContext: context, Simulation: value.Simulation,
		Revision: safeRevision(value.DesiredRevision), AppliedRevision: safeRevision(value.AppliedRevision),
	}
}

func policySubjectDTO(value accessmanagement.Subject) managementapi.PolicySubject {
	return managementapi.PolicySubject{Type: string(value.Kind), ID: value.ID}
}

func routingClaimValues(values map[string]managementapi.RoutingClaimValue) map[string]routingsnapshot.ClaimValue {
	result := make(map[string]routingsnapshot.ClaimValue, len(values))
	for name, value := range values {
		result[name] = routingsnapshot.ClaimValue{
			Kind: value.Kind, String: value.String,
			Boolean: value.Boolean, Integer: value.Integer,
		}
	}
	return result
}

func routingClaimValueDTO(value routingsnapshot.ClaimValue) managementapi.RoutingClaimValue {
	return managementapi.RoutingClaimValue{
		Kind: value.Kind, String: value.String,
		Boolean: value.Boolean, Integer: value.Integer,
	}
}

func safeRevision(value uint64) int64 {
	if value == 0 || value > math.MaxInt64 {
		return 0
	}
	return int64(value)
}

func safeRevisionAllowZero(value uint64) int64 {
	if value > math.MaxInt64 {
		return 0
	}
	return int64(value)
}
