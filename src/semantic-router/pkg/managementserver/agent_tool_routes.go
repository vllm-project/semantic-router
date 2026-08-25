package managementserver

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const agentCredentialRetirementGrace = 5 * time.Minute

type agentSourcePatchWire struct {
	Name         *string                       `json:"name,omitempty"`
	Description  *string                       `json:"description,omitempty"`
	Transport    *string                       `json:"transport,omitempty"`
	Endpoint     *string                       `json:"endpoint,omitempty"`
	CredentialID json.RawMessage               `json:"credentialId,omitempty"`
	EgressPolicy *agentmanagement.EgressPolicy `json:"egressPolicy,omitempty"`
	Status       *agentmanagement.Status       `json:"status,omitempty"`
}

func (routes *AgentRoutes) tools(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.Method != http.MethodGet {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	pageRequest, toolsErr := parseAgentToolListQuery(request)
	if toolsErr != nil {
		writeAgentDomainError(response, toolsErr, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, agentToolsPath)
	if _, err := routes.accessContext(request.Context(), authenticated, operation); err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, registryRevision, toolsErr := routes.service.ListTools(
		request.Context(), authenticated.NamespaceID, pageRequest,
	)
	if toolsErr != nil {
		writeAgentDomainError(response, toolsErr, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, agentToolPage{
		Data: page.Items,
		Page: agentPageInfo{
			NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageRequest.PageSize,
		},
		RegistryRevision: registryRevision,
	}, requestID)
}

func (routes *AgentRoutes) credentials(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		pageRequest, err := parseAgentSearchListQuery(request)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		authenticated, ok := routes.authenticate(response, request, requestID)
		if !ok {
			return
		}
		access, err := routes.accessContext(
			request.Context(), authenticated,
			routes.operation(managementapi.MethodGET, agentCredentialsPath),
		)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		pageRequest.Scope = access.Scope
		page, err := routes.service.ListToolCredentials(request.Context(), authenticated.NamespaceID, pageRequest)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
	case http.MethodPost:
		routes.createCredential(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) createCredential(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if _, err := routes.authorize(
		request.Context(), authenticated,
		routes.operation(managementapi.MethodPOST, agentCredentialsPath),
		agentNamespaceTarget(authenticated.NamespaceID),
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var body agentCredentialCreateWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	secret := []byte(body.Secret)
	defer clear(secret)
	body.Secret = ""
	result, err := routes.service.CreateToolCredential(
		request.Context(), authenticated.NamespaceID, idempotencyKey,
		agentmanagement.ToolCredentialInput{Name: body.Name, Secret: secret},
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	response.Header().Set("Location", agentCredentialsPath+"/"+result.ResourceID)
	writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
		"agent_tool_credential", result.ResourceID, publicRevision(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func (routes *AgentRoutes) credential(response http.ResponseWriter, request *http.Request, requestID string) {
	id, action, ok := agentResourcePathValue(agentCredentialsPath, request.URL.Path)
	if !ok || request.URL.RawQuery != "" || (action != "" && action != "rotate") {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	contractPath := agentCredentialsPath + "/{credential}"
	if action == "rotate" {
		contractPath += ":rotate"
	}
	operation := routes.operation(managementapi.HTTPMethod(request.Method), contractPath)
	if operation.Path == "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if _, err := routes.authorize(
		request.Context(), authenticated, operation,
		agentTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceAgentToolCredential, id),
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && action == "":
		credential, err := routes.service.GetToolCredential(request.Context(), authenticated.NamespaceID, id)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, credential.Revision)
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.ToolCredential]{Data: credential}, requestID)
	case request.Method == http.MethodPatch && action == "":
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		var patch agentmanagement.ToolCredentialPatch
		if !decodeAgentBody(response, request, requestID, &patch) {
			return
		}
		credential, err := routes.service.PatchToolCredential(
			request.Context(), authenticated.NamespaceID, id, revision, patch,
			agentMutation(request, authenticated, requestID),
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, credential.Revision)
		writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
			"agent_tool_credential", credential.ID, publicRevision(credential.Revision), nil,
		), requestID)
	case request.Method == http.MethodDelete && action == "":
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		result, err := routes.service.DeleteToolCredential(
			request.Context(), authenticated.NamespaceID, id, revision,
			agentMutation(request, authenticated, requestID),
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result)
		response.WriteHeader(http.StatusNoContent)
	case request.Method == http.MethodPost && action == "rotate":
		routes.rotateCredential(response, request, requestID, authenticated, id)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) rotateCredential(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, id string,
) {
	revision, ok := requireAgentRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var body agentCredentialRotateWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	secret := []byte(body.Secret)
	defer clear(secret)
	body.Secret = ""
	result, err := routes.service.RotateToolCredential(
		request.Context(), authenticated.NamespaceID, id, idempotencyKey, revision, secret,
		agentCredentialRetirementGrace, agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"agent_tool_credential", result.ResourceID, publicRevision(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func (routes *AgentRoutes) sources(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		pageRequest, err := parseAgentSearchListQuery(request)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		authenticated, ok := routes.authenticate(response, request, requestID)
		if !ok {
			return
		}
		access, err := routes.accessContext(
			request.Context(), authenticated,
			routes.operation(managementapi.MethodGET, agentSourcesPath),
		)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		pageRequest.Scope = access.Scope
		page, err := routes.service.ListToolSources(request.Context(), authenticated.NamespaceID, pageRequest)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
	case http.MethodPost:
		routes.createSource(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) createSource(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if _, err := routes.authorize(
		request.Context(), authenticated,
		routes.operation(managementapi.MethodPOST, agentSourcesPath),
		agentNamespaceTarget(authenticated.NamespaceID),
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var input agentmanagement.ToolSourceInput
	if !decodeAgentBody(response, request, requestID, &input) {
		return
	}
	result, err := routes.service.CreateToolSource(
		request.Context(), authenticated.NamespaceID, idempotencyKey, input,
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	response.Header().Set("Location", agentSourcesPath+"/"+result.ResourceID)
	writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
		"agent_tool_source", result.ResourceID, publicRevision(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func (routes *AgentRoutes) source(response http.ResponseWriter, request *http.Request, requestID string) {
	id, action, ok := agentResourcePathValue(agentSourcesPath, request.URL.Path)
	if !ok || request.URL.RawQuery != "" || (action != "" && action != "test" && action != "approve") {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	contractPath := agentSourcesPath + "/{source}"
	if action != "" {
		contractPath += ":" + action
	}
	operation := routes.operation(managementapi.HTTPMethod(request.Method), contractPath)
	if operation.Path == "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if _, err := routes.authorize(
		request.Context(), authenticated, operation,
		agentTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceAgentToolSource, id),
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && action == "":
		source, err := routes.service.GetToolSource(request.Context(), authenticated.NamespaceID, id)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, source.Revision)
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.ToolSource]{Data: source}, requestID)
	case request.Method == http.MethodPatch && action == "":
		routes.patchSource(response, request, requestID, authenticated, id)
	case request.Method == http.MethodDelete && action == "":
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		result, err := routes.service.DeleteToolSource(
			request.Context(), authenticated.NamespaceID, id, revision,
			agentMutation(request, authenticated, requestID),
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result)
		response.WriteHeader(http.StatusNoContent)
	case request.Method == http.MethodPost && action == "test":
		routes.testSource(response, request, requestID, authenticated, id)
	case request.Method == http.MethodPost && action == "approve":
		routes.approveSource(response, request, requestID, authenticated, id)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) patchSource(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, id string,
) {
	revision, ok := requireAgentRevision(response, request, requestID)
	if !ok {
		return
	}
	var body agentSourcePatchWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	patch := agentmanagement.ToolSourcePatch{
		Name: body.Name, Description: body.Description, Transport: body.Transport,
		Endpoint: body.Endpoint, EgressPolicy: body.EgressPolicy, Status: body.Status,
	}
	if body.CredentialID != nil {
		patch.CredentialID.Present = true
		if !bytes.Equal(bytes.TrimSpace(body.CredentialID), []byte("null")) {
			var credentialID string
			decoder := json.NewDecoder(bytes.NewReader(body.CredentialID))
			if err := decoder.Decode(&credentialID); err != nil {
				writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
				return
			}
			patch.CredentialID.Value = &credentialID
		}
	}
	source, err := routes.service.PatchToolSource(
		request.Context(), authenticated.NamespaceID, id, revision, patch,
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, source.Revision)
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"agent_tool_source", source.ID, publicRevision(source.Revision), nil,
	), requestID)
}

func (routes *AgentRoutes) testSource(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, id string,
) {
	if !emptyAgentBody(response, request, requestID) {
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.TestToolSource(
		request.Context(), authenticated.NamespaceID, id, idempotencyKey,
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"agent_tool_source", result.ResourceID, publicRevision(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func (routes *AgentRoutes) approveSource(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, id string,
) {
	revision, ok := requireAgentRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var body agentSourceApprovalWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.ApproveToolSource(
		request.Context(), authenticated.NamespaceID, id, idempotencyKey, revision, body.DiscoveryDigest,
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"agent_tool_source", result.ResourceID, publicRevision(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func emptyAgentBody(response http.ResponseWriter, request *http.Request, requestID string) bool {
	const maximumEmptyActionBodyBytes = 1024
	request.Body = http.MaxBytesReader(response, request.Body, maximumEmptyActionBodyBytes+1)
	payload, err := io.ReadAll(request.Body)
	if err != nil || len(payload) > maximumEmptyActionBodyBytes || len(bytes.TrimSpace(payload)) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body must be empty.", requestID)
		return false
	}
	return true
}
