package managementserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

type agentProfilePatchWire struct {
	Name                      *string                           `json:"name,omitempty"`
	Description               *string                           `json:"description,omitempty"`
	DefaultTarget             json.RawMessage                   `json:"defaultTarget,omitempty"`
	MinimumTargetCapabilities *[]string                         `json:"minimumTargetCapabilities,omitempty"`
	SupportedModes            *[]agentmanagement.SessionMode    `json:"supportedModes,omitempty"`
	DefaultForModes           *[]agentmanagement.SessionMode    `json:"defaultForModes,omitempty"`
	Skills                    *[]agentmanagement.SkillReference `json:"skills,omitempty"`
	ToolPolicy                *agentmanagement.ToolPolicy       `json:"toolPolicy,omitempty"`
	ApprovalPolicy            *string                           `json:"approvalPolicy,omitempty"`
	MaximumTurnSeconds        *int64                            `json:"maximumTurnSeconds,omitempty"`
	MaximumToolSteps          *int                              `json:"maximumToolSteps,omitempty"`
	ContextTokenBudget        *int64                            `json:"contextTokenBudget,omitempty"`
}

func (routes *AgentRoutes) profiles(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		routes.listProfiles(response, request, requestID)
	case http.MethodPost:
		routes.createProfile(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) listProfiles(response http.ResponseWriter, request *http.Request, requestID string) {
	pageRequest, err := parseAgentSearchListQuery(request)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, agentProfilesPath)
	access, err := routes.accessContext(request.Context(), authenticated, operation)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	pageRequest.Scope = access.Scope
	page, err := routes.service.ListProfiles(request.Context(), authenticated.NamespaceID, pageRequest, access)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
}

func (routes *AgentRoutes) createProfile(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if _, err := routes.authorize(request.Context(), authenticated,
		routes.operation(managementapi.MethodPOST, agentProfilesPath),
		agentNamespaceTarget(authenticated.NamespaceID)); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var input agentmanagement.ProfileInput
	if !decodeAgentBody(response, request, requestID, &input) {
		return
	}
	result, err := routes.service.CreateProfile(
		request.Context(), authenticated.NamespaceID, idempotencyKey, input,
		agentMutation(request, authenticated, requestID),
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setAgentETag(response, result.ResourceRevision)
	setIdempotencyReplayHeader(response, result.Replayed)
	response.Header().Set("Location", agentProfilesPath+"/"+result.ResourceID)
	writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
		"agent_profile", result.ResourceID, uint64(result.ResourceRevision), &result.Replayed,
	), requestID)
}

func (routes *AgentRoutes) profile(response http.ResponseWriter, request *http.Request, requestID string) {
	id, action, ok := agentResourcePathValue(agentProfilesPath, request.URL.Path)
	if !ok || action != "" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	method := managementapi.HTTPMethod(request.Method)
	contractPath := agentProfilesPath + "/{profile}"
	operation := routes.operation(method, contractPath)
	if operation.Path == "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if _, err := routes.authorize(request.Context(), authenticated, operation,
		agentTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceAgentProfile, id)); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	switch request.Method {
	case http.MethodGet:
		access, err := routes.accessContext(request.Context(), authenticated, operation)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		profile, err := routes.service.GetProfile(request.Context(), authenticated.NamespaceID, id, access)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, profile.Revision)
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.Profile]{Data: profile}, requestID)
	case http.MethodPatch:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		var body agentProfilePatchWire
		if !decodeAgentBody(response, request, requestID, &body) {
			return
		}
		patch, err := decodeProfilePatch(body)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		profile, err := routes.service.PatchProfile(request.Context(), authenticated.NamespaceID, id,
			revision, patch, agentMutation(request, authenticated, requestID))
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, profile.Revision)
		writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
			"agent_profile", profile.ID, uint64(profile.Revision), nil,
		), requestID)
	case http.MethodDelete:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		result, err := routes.service.DeleteProfile(request.Context(), authenticated.NamespaceID, id,
			revision, agentMutation(request, authenticated, requestID))
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result)
		response.WriteHeader(http.StatusNoContent)
	}
}

func decodeProfilePatch(body agentProfilePatchWire) (agentmanagement.ProfilePatch, error) {
	patch := agentmanagement.ProfilePatch{
		Name: body.Name, Description: body.Description,
		MinimumTargetCapabilities: body.MinimumTargetCapabilities,
		SupportedModes:            body.SupportedModes, DefaultForModes: body.DefaultForModes,
		Skills: body.Skills, ToolPolicy: body.ToolPolicy, ApprovalPolicy: body.ApprovalPolicy,
		MaximumTurnSeconds: body.MaximumTurnSeconds, MaximumToolSteps: body.MaximumToolSteps,
		ContextTokenBudget: body.ContextTokenBudget,
	}
	if body.DefaultTarget == nil {
		return patch, nil
	}
	patch.DefaultTarget.Present = true
	if bytes.Equal(bytes.TrimSpace(body.DefaultTarget), []byte("null")) {
		return patch, nil
	}
	var target agentmanagement.Target
	decoder := json.NewDecoder(bytes.NewReader(body.DefaultTarget))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&target); err != nil {
		return agentmanagement.ProfilePatch{}, agentmanagement.ErrInvalid
	}
	patch.DefaultTarget.Value = &target
	return patch, nil
}

func (routes *AgentRoutes) skills(response http.ResponseWriter, request *http.Request, requestID string) {
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
		operation := routes.operation(managementapi.MethodGET, agentSkillsPath)
		access, err := routes.accessContext(request.Context(), authenticated, operation)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		pageRequest.Scope = access.Scope
		page, err := routes.service.ListSkills(request.Context(), authenticated.NamespaceID, pageRequest)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
	case http.MethodPost:
		authenticated, ok := routes.authenticate(response, request, requestID)
		if !ok {
			return
		}
		if _, err := routes.authorize(request.Context(), authenticated,
			routes.operation(managementapi.MethodPOST, agentSkillsPath), agentNamespaceTarget(authenticated.NamespaceID)); err != nil {
			writeAgentAuthorizationError(response, err, requestID)
			return
		}
		idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
		if !ok {
			return
		}
		var input agentmanagement.SkillInput
		if !decodeAgentBody(response, request, requestID, &input) {
			return
		}
		result, err := routes.service.CreateSkill(
			request.Context(), authenticated.NamespaceID, idempotencyKey, input,
			agentMutation(request, authenticated, requestID),
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result.ResourceRevision)
		setIdempotencyReplayHeader(response, result.Replayed)
		response.Header().Set("Location", agentSkillsPath+"/"+result.ResourceID)
		writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
			"agent_skill", result.ResourceID, uint64(result.ResourceRevision), &result.Replayed,
		), requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) skill(response http.ResponseWriter, request *http.Request, requestID string) {
	id, action, ok := agentResourcePathValue(agentSkillsPath, request.URL.Path)
	if !ok || action != "" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.HTTPMethod(request.Method), agentSkillsPath+"/{skill}")
	if operation.Path == "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if _, err := routes.authorize(request.Context(), authenticated, operation,
		agentTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceAgentSkill, id)); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	switch request.Method {
	case http.MethodGet:
		skill, err := routes.service.GetSkill(request.Context(), authenticated.NamespaceID, id)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, skill.Revision)
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.Skill]{Data: skill}, requestID)
	case http.MethodPatch:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		var patch agentmanagement.SkillPatch
		if !decodeAgentBody(response, request, requestID, &patch) {
			return
		}
		skill, err := routes.service.PatchSkill(request.Context(), authenticated.NamespaceID, id,
			revision, patch, agentMutation(request, authenticated, requestID))
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, skill.Revision)
		writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
			"agent_skill", skill.ID, uint64(skill.Revision), nil,
		), requestID)
	case http.MethodDelete:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		result, err := routes.service.DeleteSkill(request.Context(), authenticated.NamespaceID, id,
			revision, agentMutation(request, authenticated, requestID))
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result)
		response.WriteHeader(http.StatusNoContent)
	}
}

func agentResourcePathValue(basePath, path string) (string, string, bool) {
	value := strings.TrimPrefix(path, basePath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", "", false
	}
	id, action, hasAction := strings.Cut(value, ":")
	if uuid.Validate(id) != nil || (hasAction && action == "") {
		return "", "", false
	}
	return id, action, true
}
