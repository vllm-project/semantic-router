package managementserver

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	providerPath              = managementapi.BasePath + "/providers"
	maximumProviderQueryBytes = 16 << 10
	maximumDiscoveryBodyBytes = 256 << 10
	maximumBearerTokenBytes   = 16 << 10
)

type ProviderRoutes struct {
	catalog           ProviderCatalog
	discovery         ProviderDiscovery
	namespaces        NamespaceResolver
	sessions          SessionAuthenticator
	authorization     Authorizer
	now               func() time.Time
	listOperation     managementapi.OperationContract
	getOperation      managementapi.OperationContract
	discoverOperation managementapi.OperationContract
}

func NewProviderRoutes(options ProviderRoutesOptions) (*ProviderRoutes, error) {
	if options.Catalog == nil || options.Discovery == nil || options.Namespaces == nil ||
		options.Sessions == nil || options.Authorization == nil {
		return nil, fmt.Errorf("provider Management routes require catalog, discovery, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	listOperation, listFound := managementapi.LookupOperation(managementapi.MethodGET, providerPath)
	getOperation, getFound := managementapi.LookupOperation(managementapi.MethodGET, providerPath+"/{providerId}")
	discoverOperation, discoverFound := managementapi.LookupOperation(
		managementapi.MethodPOST, providerPath+"/{providerId}:discover-models",
	)
	if !listFound || !getFound || !discoverFound {
		return nil, fmt.Errorf("provider Management operation contracts are unavailable")
	}
	return &ProviderRoutes{
		catalog: options.Catalog, discovery: options.Discovery, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, now: now,
		listOperation: listOperation, getOperation: getOperation, discoverOperation: discoverOperation,
	}, nil
}

func (routes *ProviderRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Provider Management routes and mux are required")
	}
	mux.Handle("GET "+providerPath, routes)
	mux.Handle("GET "+providerPath+"/", routes)
	mux.Handle("POST "+providerPath+"/", routes)
}

func (routes *ProviderRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil {
		writeProviderError(response, http.StatusServiceUnavailable, "service_unavailable", "Provider catalog is unavailable.", requestID)
		return
	}
	if request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && request.URL.Path == providerPath:
		routes.list(response, request, requestID)
	case request.Method == http.MethodGet && strings.HasPrefix(request.URL.Path, providerPath+"/"):
		routes.detail(response, request, requestID)
	case request.Method == http.MethodPost && strings.HasPrefix(request.URL.Path, providerPath+"/"):
		routes.discover(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *ProviderRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "search": true, "category": true, "capability": true,
	})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider query is invalid.", requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	_, _, ok := routes.authorize(response, request, requestID, routes.listOperation, "")
	if !ok {
		return
	}
	result, err := routes.catalog.List(request.Context(), providercatalog.ListRequest{
		PageSize: pageSize, Cursor: query.Get("cursor"), Search: query.Get("search"),
		Category: query.Get("category"), Capability: query.Get("capability"),
	})
	if err != nil {
		writeCatalogDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, providerCatalogPageDTO(result), requestID)
}

func (routes *ProviderRoutes) detail(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider detail does not accept query parameters.", requestID)
		return
	}
	providerID, action, ok := providerPathValue(request.URL.Path)
	if !ok || action != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if _, _, ok := routes.authorize(response, request, requestID, routes.getOperation, ""); !ok {
		return
	}
	result, err := routes.catalog.Get(request.Context(), providerID)
	if err != nil {
		writeCatalogDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, providerCatalogDetailDTO(result), requestID)
}

func (routes *ProviderRoutes) discover(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Discovery pagination belongs in the JSON body.", requestID)
		return
	}
	providerID, action, ok := providerPathValue(request.URL.Path)
	if !ok || action != "discover-models" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.DiscoverModelsRequest
	if err := decodeStrictProviderJSON(response, request, &body); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, errProviderBodyTooLarge) {
			status = http.StatusRequestEntityTooLarge
		}
		writeProviderError(response, status, "invalid_request", "Discovery request body is invalid.", requestID)
		return
	}
	decision, ok := routes.authorizeAuthenticated(
		response, request, requestID, session, namespaceID, routes.discoverOperation, body.CredentialID,
	)
	if !ok {
		return
	}
	if !validAuthorityDigest(decision.AuthorityDigest) {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	catalogRequest, err := providerDiscoveryRequest(body, namespaceID)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Discovery request is invalid.", requestID)
		return
	}
	plan, err := routes.catalog.PrepareDiscovery(request.Context(), providerID, catalogRequest)
	if err != nil {
		writeCatalogDomainError(response, err, requestID)
		return
	}
	result, err := routes.discovery.Execute(request.Context(), providerdiscovery.ExecuteRequest{
		Plan: plan, AuthorityDigest: decision.AuthorityDigest,
	})
	if err != nil {
		writeDiscoveryDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, discoveredModelsPageDTO(result, plan.PageSize), requestID)
}

func (routes *ProviderRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	operation managementapi.OperationContract,
	credentialID string,
) (string, AuthorizationDecision, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return "", AuthorizationDecision{}, false
	}
	decision, ok := routes.authorizeAuthenticated(
		response, request, requestID, session, namespaceID, operation, credentialID,
	)
	return namespaceID, decision, ok
}

func (routes *ProviderRoutes) authenticate(
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
	if session.NamespaceID != namespaceID || session.Session.PrincipalID == "" {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *ProviderRoutes) authorizeAuthenticated(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	operation managementapi.OperationContract,
	credentialID string,
) (AuthorizationDecision, bool) {
	targets := make(map[string][]accesscontrol.ScopedTarget)
	if credentialID != "" {
		targets["credential"] = []accesscontrol.ScopedTarget{{
			Scope: accesscontrol.ResourceScope(
				accesscontrol.NamespaceID(namespaceID),
				accesscontrol.ScopeResourceProviderCredential,
				accesscontrol.ResourceID(credentialID),
			),
		}}
	}
	decision, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: targets,
		Conditions: map[string]bool{
			"provider_credential_supplied":    credentialID != "",
			"no_provider_credential_supplied": credentialID == "",
		},
	})
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable."
		if errors.Is(err, managementauthorization.ErrDenied) {
			status, code, message = http.StatusForbidden, "forbidden", "Permission denied."
		}
		writeProviderError(response, status, code, message, requestID)
		return AuthorizationDecision{}, false
	}
	return decision, true
}

func providerPathValue(path string) (string, string, bool) {
	value := strings.TrimPrefix(path, providerPath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", "", false
	}
	providerID, action, hasAction := strings.Cut(value, ":")
	if !providerIDPattern.MatchString(providerID) || (hasAction && action == "") {
		return "", "", false
	}
	return providerID, action, true
}

func strictProviderQuery(raw string, allowed map[string]bool) (url.Values, error) {
	if len(raw) > maximumProviderQueryBytes {
		return nil, fmt.Errorf("query is too large")
	}
	values, err := url.ParseQuery(raw)
	if err != nil {
		return nil, err
	}
	for name, entries := range values {
		if !allowed[name] || len(entries) != 1 {
			return nil, fmt.Errorf("query parameter %q is unknown or repeated", name)
		}
	}
	return values, nil
}

func parseOptionalPageSize(raw string) (int, error) {
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 1 || value > 200 || strconv.Itoa(value) != raw {
		return 0, fmt.Errorf("page size is invalid")
	}
	return value, nil
}

var errProviderBodyTooLarge = errors.New("provider Management request body is too large")

func decodeStrictProviderJSON(response http.ResponseWriter, request *http.Request, target any) error {
	if request.ContentLength > maximumDiscoveryBodyBytes {
		return errProviderBodyTooLarge
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		return fmt.Errorf("content type is invalid")
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumDiscoveryBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			return errProviderBodyTooLarge
		}
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("JSON body contains trailing values")
	}
	return nil
}

func bearerToken(request *http.Request) (string, bool) {
	values := request.Header.Values("Authorization")
	if len(values) != 1 || len(values[0]) > maximumBearerTokenBytes {
		return "", false
	}
	const prefix = "Bearer "
	if !strings.HasPrefix(values[0], prefix) {
		return "", false
	}
	token := strings.TrimPrefix(values[0], prefix)
	return token, token != "" && strings.TrimSpace(token) == token && !strings.ContainsAny(token, "\r\n\t ")
}

func managementRequestID(request *http.Request) string {
	if request != nil {
		values := request.Header.Values(managementapi.HeaderRequestID)
		if len(values) != 1 {
			return uuid.NewString()
		}
		value := values[0]
		if len(value) >= 1 && len(value) <= 128 && strings.TrimSpace(value) == value &&
			!strings.ContainsAny(value, "\x00\r\n\t") {
			return value
		}
	}
	return uuid.NewString()
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func validAuthorityDigest(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	decoded, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}

var providerIDPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
