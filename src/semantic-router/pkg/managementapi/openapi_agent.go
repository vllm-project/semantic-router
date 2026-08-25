package managementapi

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name:            "agent",
		Schemas:         agentSchemas,
		RequestSchema:   agentRequestSchema,
		ResponseSchema:  agentResponseSchema,
		ExtraParameters: agentParameters,
		AmendResponses:  amendAgentResponses,
	})
}

func agentRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-profiles":
		return "AgentProfileCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/agent-profiles/{profile}":
		return "AgentProfilePatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-skills":
		return "AgentSkillCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/agent-skills/{skill}":
		return "AgentSkillPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-tool-credentials":
		return "AgentToolCredentialCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/agent-tool-credentials/{credential}":
		return "AgentToolCredentialPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-tool-credentials/{credential}:rotate":
		return "AgentToolCredentialRotateRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-tool-sources":
		return "AgentToolSourceCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/agent-tool-sources/{source}":
		return "AgentToolSourcePatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-tool-sources/{source}:approve":
		return "AgentToolSourceApprovalRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-sessions":
		return "AgentSessionCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/agent-sessions/{session}":
		return "AgentSessionPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/agent-sessions/{session}/turns":
		return "AgentTurnCreateRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/publication-plans/{plan}:commit":
		return "AgentPublicationCommitRequest", true
	default:
		return "", false
	}
}

func agentResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-profiles":
		return refSchema("AgentProfilePage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-profiles/{profile}":
		return refSchema("AgentProfileDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-skills":
		return refSchema("AgentSkillPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-skills/{skill}":
		return refSchema("AgentSkillDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-tools":
		return refSchema("AgentToolPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-tool-credentials":
		return refSchema("AgentToolCredentialPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-tool-credentials/{credential}":
		return refSchema("AgentToolCredentialDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-tool-sources":
		return refSchema("AgentToolSourcePage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-tool-sources/{source}":
		return refSchema("AgentToolSourceDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions":
		return refSchema("AgentSessionPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions/{session}":
		return refSchema("AgentSessionDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions/{session}/turns":
		return refSchema("AgentTurnPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions/{session}/events":
		return refSchema("AgentEventPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-artifacts/{artifact}":
		return refSchema("AgentArtifactDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/agent-artifacts/{artifact}/content":
		return refSchema("AgentArtifactContentDetail"), true
	case isAgentMutation(contract):
		return refSchema("MutationReceipt"), true
	default:
		return JSONSchema{}, false
	}
}

func isAgentMutation(contract OperationContract) bool {
	if contract.Method != MethodPOST && contract.Method != MethodPATCH {
		return false
	}
	return strings.HasPrefix(contract.Path, BasePath+"/agent-") ||
		contract.Path == BasePath+"/publication-plans/{plan}:commit"
}

func agentParameters(contract OperationContract) []OpenAPIParameter {
	if agentSearchableCollection(contract) {
		return []OpenAPIParameter{{
			Name: "search", In: "query",
			Description: "Case-insensitive literal prefix matched against public resource names and descriptions.",
			Schema:      JSONSchema{Type: "string", MaxLength: intPointer(managementsearch.MaximumRunes)},
		}}
	}
	if contract.Method != MethodGET || contract.Path != BasePath+"/agent-sessions/{session}/events" {
		return nil
	}
	return []OpenAPIParameter{
		{
			Name: "afterSequence", In: "query",
			Description: "Resume an event stream strictly after this sequence. Mutually exclusive with Last-Event-ID.",
			Schema:      boundedIntegerSchema(0, 1<<62),
		},
		{
			Name: "Last-Event-ID", In: "header",
			Description: "SSE sequence to resume after. Mutually exclusive with afterSequence.",
			Schema:      JSONSchema{Type: "string", Pattern: `^(0|[1-9][0-9]{0,18})$`},
		},
	}
}

func agentSearchableCollection(contract OperationContract) bool {
	if contract.Method != MethodGET {
		return false
	}
	switch contract.Path {
	case BasePath + "/agent-profiles",
		BasePath + "/agent-skills",
		BasePath + "/agent-tools",
		BasePath + "/agent-tool-credentials",
		BasePath + "/agent-tool-sources",
		BasePath + "/agent-sessions":
		return true
	default:
		return false
	}
}

func amendAgentResponses(contract OperationContract, responses map[string]OpenAPIResponse) {
	if contract.Method != MethodGET || contract.Path != BasePath+"/agent-sessions/{session}/events" {
		return
	}
	success := responses["200"]
	if success.Content == nil {
		success.Content = make(map[string]OpenAPIMedia)
	}
	success.Content[EventStreamMediaType] = OpenAPIMedia{Schema: refSchema("AgentEventStream")}
	responses["200"] = success
	responses["410"] = OpenAPIResponse{
		Description: "Requested event history has expired; recover from the supplied checkpoint.",
		Content: map[string]OpenAPIMedia{
			JSONMediaType: {Schema: refSchema("AgentEventHistoryExpiredError")},
		},
	}
}

var (
	revision                   = JSONSchema{Type: "integer", Format: "int64", Minimum: intPointer(1)}
	digest                     = JSONSchema{Type: "string", Pattern: `^sha256:[a-f0-9]{64}$`}
	status                     = JSONSchema{Type: "string", Enum: []string{"active", "disabled", "deleted"}}
	openObject                 = openObjectSchema
	resourceProperties         = agentResourceProperties()
	resourceRequired           = []string{"id", "name", "status", "revision", "createdAt", "updatedAt"}
	namespacedResourceRequired = append(append([]string(nil), resourceRequired...), "namespaceId")
	profileProperties          = agentProfileProperties()
	profileRequired            = append(append([]string(nil), namespacedResourceRequired...), "contentRevision",
		"minimumTargetCapabilities", "supportedModes", "defaultForModes", "skills", "toolPolicy", "approvalPolicy",
		"maximumTurnSeconds", "maximumToolSteps", "contextTokenBudget")
	skillProperties = agentSkillProperties()
	skillRequired   = append(append([]string(nil), resourceRequired...), "contentRevision",
		"builtin", "requiredTools", "minimumCapabilities", "contentDigest")
	toolSourceProperties = agentToolSourceProperties()
	toolSourceRequired   = append(append([]string(nil), namespacedResourceRequired...), "contentRevision",
		"kind", "transport", "endpoint", "egressPolicy", "discoveredTools", "availability")
)

func agentResourceProperties() map[string]JSONSchema {
	return map[string]JSONSchema{
		"id": uuid, "namespaceId": uuid, "name": stringSchema, "description": stringSchema,
		"status": status, "revision": revision, "createdAt": timestamp, "updatedAt": timestamp,
	}
}

func agentProfileProperties() map[string]JSONSchema {
	properties := cloneSchemas(resourceProperties)
	properties["contentRevision"] = revision
	properties["defaultTarget"] = refSchema("AgentTarget")
	properties["minimumTargetCapabilities"] = stringArray
	properties["supportedModes"] = arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}})
	properties["defaultForModes"] = arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}})
	properties["skills"] = arraySchema(refSchema("AgentSkillReference"))
	properties["toolPolicy"] = refSchema("AgentToolPolicy")
	properties["approvalPolicy"] = JSONSchema{Type: "string", Enum: []string{"required"}}
	properties["maximumTurnSeconds"] = boundedIntegerSchema(10, 86400)
	properties["maximumToolSteps"] = boundedIntegerSchema(1, 256)
	properties["contextTokenBudget"] = boundedIntegerSchema(1024, 1048576)
	return properties
}

func agentSkillProperties() map[string]JSONSchema {
	properties := cloneSchemas(resourceProperties)
	properties["contentRevision"] = revision
	properties["builtin"] = JSONSchema{Type: "boolean"}
	properties["instructions"] = stringSchema
	properties["requiredTools"] = stringArray
	properties["minimumCapabilities"] = stringArray
	properties["contentDigest"] = digest
	return properties
}

func agentToolSourceProperties() map[string]JSONSchema {
	properties := cloneSchemas(resourceProperties)
	properties["contentRevision"] = revision
	properties["kind"] = JSONSchema{Type: "string", Enum: []string{"remote"}}
	properties["transport"] = JSONSchema{Type: "string", Enum: []string{"streamable_http"}}
	properties["endpoint"] = JSONSchema{Type: "string", Format: "uri"}
	properties["credentialId"] = uuid
	properties["egressPolicy"] = refSchema("AgentEgressPolicy")
	properties["discoveredTools"] = arraySchema(refSchema("AgentToolDefinition"))
	properties["discoveryDigest"] = digest
	properties["approvedDiscoveryDigest"] = digest
	properties["availability"] = JSONSchema{Type: "string", Enum: []string{
		"undiscovered", "pending_approval", "ready", "drifted", "disabled",
	}}
	return properties
}

func agentEventSchema(eventType, payloadSchema string) JSONSchema {
	return objectSchema(
		[]string{"sessionId", "sequence", "type", "payload", "createdAt"},
		map[string]JSONSchema{
			"sessionId": uuid,
			"turnId":    uuid,
			"sequence":  revision,
			"type":      {Type: "string", Enum: []string{eventType}},
			"payload":   refSchema(payloadSchema),
			"createdAt": timestamp,
		},
	)
}

var agentSchemaCatalog = map[string]JSONSchema{
	"AgentTarget": objectSchema([]string{"kind", "id"}, map[string]JSONSchema{
		"kind": {Type: "string", Enum: []string{"model", "entrypoint"}},
		"id": {
			Type: "string", MinLength: intPointer(1), MaxLength: intPointer(256),
			Description: "Authorized request-facing identifier returned by /v1/models.",
		},
	}),
	"AgentSkillReference": objectSchema([]string{"id", "revision"}, map[string]JSONSchema{
		"id": uuid, "revision": revision,
	}),
	"AgentToolPolicy": objectSchema([]string{"allow"}, map[string]JSONSchema{
		"allow": stringArray, "deny": stringArray,
	}),
	"AgentProfileCreateRequest": objectSchema([]string{"name", "toolPolicy"}, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema,
		"defaultTarget":             {OneOf: []JSONSchema{refSchema("AgentTarget"), {Type: "null"}}},
		"minimumTargetCapabilities": stringArray, "skills": arraySchema(refSchema("AgentSkillReference")),
		"supportedModes":  arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}}),
		"defaultForModes": arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}}),
		"toolPolicy":      refSchema("AgentToolPolicy"), "approvalPolicy": {Type: "string", Enum: []string{"required"}},
		"maximumTurnSeconds": boundedIntegerSchema(10, 86400), "maximumToolSteps": boundedIntegerSchema(1, 256),
		"contextTokenBudget": boundedIntegerSchema(1024, 1048576),
	}),
	"AgentProfilePatchRequest": objectSchema(nil, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema,
		"defaultTarget":             {OneOf: []JSONSchema{refSchema("AgentTarget"), {Type: "null"}}},
		"minimumTargetCapabilities": stringArray, "skills": arraySchema(refSchema("AgentSkillReference")),
		"supportedModes":  arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}}),
		"defaultForModes": arraySchema(JSONSchema{Type: "string", Enum: []string{"chat", "builder"}}),
		"toolPolicy":      refSchema("AgentToolPolicy"), "approvalPolicy": {Type: "string", Enum: []string{"required"}},
		"maximumTurnSeconds": boundedIntegerSchema(10, 86400),
		"maximumToolSteps":   boundedIntegerSchema(1, 256), "contextTokenBudget": boundedIntegerSchema(1024, 1048576),
	}),
	"AgentProfile":       objectSchema(profileRequired, profileProperties),
	"AgentProfilePage":   objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(refSchema("AgentProfile")), "page": refSchema("PageInfo")}),
	"AgentProfileDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentProfile")}),

	"AgentSkillCreateRequest": objectSchema([]string{"name", "instructions"}, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema, "instructions": stringSchema,
		"requiredTools": stringArray, "minimumCapabilities": stringArray,
	}),
	"AgentSkillPatchRequest": objectSchema(nil, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema, "instructions": stringSchema,
		"requiredTools": stringArray, "minimumCapabilities": stringArray,
	}),
	"AgentSkill":       objectSchema(skillRequired, skillProperties),
	"AgentSkillPage":   objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(refSchema("AgentSkill")), "page": refSchema("PageInfo")}),
	"AgentSkillDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentSkill")}),

	"AgentToolDefinition": objectSchema([]string{
		"name", "description", "inputSchema", "outputSchema", "requiredPermissions", "class", "idempotency", "timeoutMilliseconds",
	}, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema, "inputSchema": openObject,
		"outputSchema": openObject, "requiredPermissions": stringArray,
		"class":               {Type: "string", Enum: []string{"read", "write", "execute"}},
		"idempotency":         {Type: "string", Enum: []string{"none", "invocation"}},
		"timeoutMilliseconds": boundedIntegerSchema(100, 600000),
	}),
	"AgentToolPage": objectSchema([]string{"data", "page", "registryRevision"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("AgentToolDefinition")), "page": refSchema("PageInfo"), "registryRevision": digest,
	}),

	"AgentToolCredentialCreateRequest": objectSchema([]string{"name", "secret"}, map[string]JSONSchema{
		"name": stringSchema, "secret": {Type: "string", Format: "password", Description: "Write-only Tool Source credential."},
	}),
	"AgentToolCredentialPatchRequest": objectSchema(nil, map[string]JSONSchema{
		"name": stringSchema, "status": {Type: "string", Enum: []string{"active", "disabled"}},
	}),
	"AgentToolCredentialRotateRequest": objectSchema([]string{"secret"}, map[string]JSONSchema{
		"secret": {Type: "string", Format: "password", Description: "Write-only replacement credential."},
	}),
	"AgentToolCredential": objectSchema(namespacedResourceRequired, resourceProperties),
	"AgentToolCredentialPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("AgentToolCredential")), "page": refSchema("PageInfo"),
	}),
	"AgentToolCredentialDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentToolCredential")}),

	"AgentEgressPolicy": objectSchema([]string{"allowedHosts"}, map[string]JSONSchema{
		"allowedHosts": stringArray, "allowedPorts": arraySchema(boundedIntegerSchema(1, 65535)),
		"allowedPrivateCidrs": stringArray,
	}),
	"AgentToolSourceCreateRequest": objectSchema([]string{"name", "kind", "transport", "endpoint", "egressPolicy"}, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema, "kind": {Type: "string", Enum: []string{"remote"}},
		"transport": {Type: "string", Enum: []string{"streamable_http"}},
		"endpoint":  {Type: "string", Format: "uri"}, "credentialId": uuid, "egressPolicy": refSchema("AgentEgressPolicy"),
	}),
	"AgentToolSourcePatchRequest": objectSchema(nil, map[string]JSONSchema{
		"name": stringSchema, "description": stringSchema,
		"transport":    {Type: "string", Enum: []string{"streamable_http"}},
		"endpoint":     {Type: "string", Format: "uri"},
		"credentialId": {OneOf: []JSONSchema{uuid, {Type: "null"}}},
		"egressPolicy": refSchema("AgentEgressPolicy"), "status": {Type: "string", Enum: []string{"active", "disabled"}},
	}),
	"AgentToolSourceApprovalRequest": objectSchema([]string{"discoveryDigest"}, map[string]JSONSchema{
		"discoveryDigest": digest,
	}),
	"AgentToolSource":       objectSchema(toolSourceRequired, toolSourceProperties),
	"AgentToolSourcePage":   objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(refSchema("AgentToolSource")), "page": refSchema("PageInfo")}),
	"AgentToolSourceDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentToolSource")}),

	"AgentSessionCreateRequest": objectSchema([]string{"mode", "keyId", "target"}, map[string]JSONSchema{
		"mode": {Type: "string", Enum: []string{"chat", "builder"}}, "profileId": uuid,
		"keyId": uuid, "effectiveTeamId": uuid, "target": refSchema("AgentTarget"), "title": stringSchema,
	}),
	"AgentSessionPatchRequest": objectSchema(nil, map[string]JSONSchema{
		"title": stringSchema, "status": {Type: "string", Enum: []string{"closed"}},
	}),
	"AgentSession": objectSchema([]string{
		"id", "namespaceId", "ownerPrincipalId", "keyId", "profileId", "profileRevision", "target", "mode", "title", "status", "revision", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"id": uuid, "namespaceId": uuid, "ownerPrincipalId": uuid, "effectiveUserId": uuid, "effectiveTeamId": uuid,
		"keyId": uuid, "profileId": uuid, "profileRevision": revision, "target": refSchema("AgentTarget"),
		"mode": {Type: "string", Enum: []string{"chat", "builder"}}, "title": stringSchema,
		"status": {Type: "string", Enum: []string{"active", "closed", "deleted"}}, "revision": revision,
		"createdAt": timestamp, "updatedAt": timestamp,
	}),
	"AgentSessionPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("AgentSession")), "page": refSchema("PageInfo"),
	}),
	"AgentSessionDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentSession")}),

	"AgentTextContent": objectSchema([]string{"type", "text"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"text"}}, "text": stringSchema,
	}),
	"AgentImageURLContent": objectSchema([]string{"type", "url"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"image_url"}}, "url": {Type: "string", Format: "uri"},
		"detail": {Type: "string", Enum: []string{"auto", "low", "high"}},
	}),
	"AgentFileReferenceContent": objectSchema([]string{"type", "fileId"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"file_reference"}}, "fileId": uuid,
	}),
	"AgentContentBlock": {OneOf: []JSONSchema{
		refSchema("AgentTextContent"), refSchema("AgentImageURLContent"), refSchema("AgentFileReferenceContent"),
	}},
	"AgentTurnInput": objectSchema([]string{"content"}, map[string]JSONSchema{
		"content": {Type: "array", Items: schemaPointer(refSchema("AgentContentBlock")), MinItems: intPointer(1), MaxItems: intPointer(64)},
	}),
	"AgentTurnCreateRequest": objectSchema([]string{"input"}, map[string]JSONSchema{"input": refSchema("AgentTurnInput")}),
	"AgentFailure": objectSchema([]string{"code", "message", "retryable"}, map[string]JSONSchema{
		"code": stringSchema, "message": stringSchema, "retryable": {Type: "boolean"},
	}),
	"AgentTurn": objectSchema([]string{
		"id", "sessionId", "ordinal", "status", "input", "revision", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"id": uuid, "sessionId": uuid, "ordinal": revision,
		"status":           {Type: "string", Enum: []string{"queued", "running", "waiting_approval", "completed", "failed", "cancelled"}},
		"registryRevision": digest, "input": refSchema("AgentTurnInput"), "revision": revision,
		"cancelRequestedAt": timestamp, "failure": refSchema("AgentFailure"), "createdAt": timestamp, "updatedAt": timestamp,
	}),
	"AgentTurnPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("AgentTurn")), "page": refSchema("PageInfo"),
	}),

	"AgentUserInputEventPayload": objectSchema([]string{"content"}, map[string]JSONSchema{
		"content": {Type: "array", Items: schemaPointer(refSchema("AgentContentBlock")), MinItems: intPointer(1), MaxItems: intPointer(64)},
	}),
	"AgentAssistantDeltaEventPayload": objectSchema([]string{"modelStepId", "chunkIndex", "delta"}, map[string]JSONSchema{
		"modelStepId": uuid,
		"chunkIndex":  boundedIntegerSchema(0, 1<<31-1),
		"delta": objectSchema([]string{"kind", "text"}, map[string]JSONSchema{
			"kind": {Type: "string", Enum: []string{"text"}},
			"text": {Type: "string", MinLength: intPointer(1)},
		}),
	}),
	"AgentToolRequestEventPayload": objectSchema([]string{"invocationId", "toolName", "arguments", "class"}, map[string]JSONSchema{
		"invocationId": uuid, "toolName": stringSchema, "arguments": openObject,
		"class": {Type: "string", Enum: []string{"read", "write", "execute"}},
	}),
	"AgentToolResultEventPayload": objectSchema([]string{"invocationId", "toolName", "status"}, map[string]JSONSchema{
		"invocationId": uuid, "toolName": stringSchema,
		"status": {Type: "string", Enum: []string{"completed", "failed", "cancelled"}},
		"result": openObject, "artifactId": uuid, "error": refSchema("AgentFailure"),
	}),
	"AgentProgressEventPayload": objectSchema([]string{"phase", "message"}, map[string]JSONSchema{
		"phase": stringSchema, "message": stringSchema,
	}),
	"AgentContextCheckpointEventPayload": objectSchema([]string{"checkpointId", "throughSequence"}, map[string]JSONSchema{
		"checkpointId": uuid, "throughSequence": revision,
	}),
	"AgentApprovalRequestEventPayload": objectSchema([]string{
		"planId", "planDigest", "planRevision", "planEtag", "expiresAt", "summary",
	}, map[string]JSONSchema{
		"planId": uuid, "planDigest": digest, "planRevision": revision,
		"planEtag": stringSchema, "expiresAt": timestamp, "summary": refSchema("AgentPublicationSummary"),
	}),
	"AgentPublicationSummary": objectSchema(nil, map[string]JSONSchema{
		"recipeId": routingResourceID, "recipeName": stringSchema,
		"entrypointId": routingResourceID, "entrypointName": stringSchema,
		"changedResources": stringArray, "warnings": stringArray,
		"topology": openObject, "assignments": openObject, "gateResults": openObject,
	}),
	"AgentApprovalResultEventPayload": objectSchema([]string{"planId", "status"}, map[string]JSONSchema{
		"planId": uuid, "status": {Type: "string", Enum: []string{"committed", "rejected", "expired", "failed"}},
		"operationId": uuid,
	}),
	"AgentCancellationEventPayload": objectSchema([]string{"requestedAt"}, map[string]JSONSchema{
		"requestedAt": timestamp,
	}),
	"AgentTerminalEventPayload": objectSchema([]string{"status"}, map[string]JSONSchema{
		"status": {Type: "string", Enum: []string{"completed", "failed", "cancelled"}},
		"error":  refSchema("AgentFailure"),
	}),
	"AgentEventPayload": {OneOf: []JSONSchema{
		refSchema("AgentUserInputEventPayload"), refSchema("AgentAssistantDeltaEventPayload"),
		refSchema("AgentToolRequestEventPayload"), refSchema("AgentToolResultEventPayload"),
		refSchema("AgentProgressEventPayload"), refSchema("AgentContextCheckpointEventPayload"),
		refSchema("AgentApprovalRequestEventPayload"), refSchema("AgentApprovalResultEventPayload"),
		refSchema("AgentCancellationEventPayload"), refSchema("AgentTerminalEventPayload"),
	}},
	"AgentUserInputEvent":         agentEventSchema("user_input", "AgentUserInputEventPayload"),
	"AgentAssistantDeltaEvent":    agentEventSchema("assistant_delta", "AgentAssistantDeltaEventPayload"),
	"AgentToolRequestEvent":       agentEventSchema("tool_request", "AgentToolRequestEventPayload"),
	"AgentToolResultEvent":        agentEventSchema("tool_result", "AgentToolResultEventPayload"),
	"AgentProgressEvent":          agentEventSchema("progress", "AgentProgressEventPayload"),
	"AgentContextCheckpointEvent": agentEventSchema("context_checkpoint", "AgentContextCheckpointEventPayload"),
	"AgentApprovalRequestEvent":   agentEventSchema("approval_request", "AgentApprovalRequestEventPayload"),
	"AgentApprovalResultEvent":    agentEventSchema("approval_result", "AgentApprovalResultEventPayload"),
	"AgentCancellationEvent":      agentEventSchema("cancellation", "AgentCancellationEventPayload"),
	"AgentTerminalEvent":          agentEventSchema("terminal", "AgentTerminalEventPayload"),
	"AgentEvent": {OneOf: []JSONSchema{
		refSchema("AgentUserInputEvent"), refSchema("AgentAssistantDeltaEvent"),
		refSchema("AgentToolRequestEvent"), refSchema("AgentToolResultEvent"),
		refSchema("AgentProgressEvent"), refSchema("AgentContextCheckpointEvent"),
		refSchema("AgentApprovalRequestEvent"), refSchema("AgentApprovalResultEvent"),
		refSchema("AgentCancellationEvent"), refSchema("AgentTerminalEvent"),
	}},
	"AgentEventPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("AgentEvent")), "page": refSchema("PageInfo"),
	}),
	"AgentLiveModelStepEvent": {OneOf: []JSONSchema{
		objectSchema([]string{
			"sessionId", "turnId", "modelStepId", "phase", "ordinal", "delta", "createdAt",
		}, map[string]JSONSchema{
			"sessionId": uuid, "turnId": uuid, "modelStepId": uuid,
			"phase":   {Type: "string", Enum: []string{"delta"}},
			"ordinal": boundedIntegerSchema(1, 1<<31-1),
			"delta": objectSchema([]string{"kind", "text"}, map[string]JSONSchema{
				"kind": {Type: "string", Enum: []string{"text"}},
				"text": {Type: "string", MinLength: intPointer(1)},
			}),
			"createdAt": timestamp,
		}),
		objectSchema([]string{
			"sessionId", "turnId", "modelStepId", "phase", "createdAt",
		}, map[string]JSONSchema{
			"sessionId": uuid, "turnId": uuid, "modelStepId": uuid,
			"phase":   {Type: "string", Enum: []string{"committed", "discarded"}},
			"ordinal": boundedIntegerSchema(0, 1<<31-1), "createdAt": timestamp,
		}),
	}},
	"AgentEventStream": {
		Type:        "string",
		Description: "Durable SSE records use the PostgreSQL sequence as id, the Agent event type as event, and AgentEvent JSON as data. Transient assistant_delta.provisional, model_step.committed, and model_step.discarded records carry AgentLiveModelStepEvent JSON without an id; they are visible only while attached, never replayed, and are replaced or removed by the durable modelStepId outcome.",
	},
	"AgentEventHistoryExpiredError": objectSchema([]string{"error", "recovery"}, map[string]JSONSchema{
		"error": refSchema("APIError"),
		"recovery": objectSchema([]string{"checkpointId", "throughSequence"}, map[string]JSONSchema{
			"checkpointId": uuid, "throughSequence": revision, "eventsHref": stringSchema,
		}),
	}),

	"AgentArtifact": objectSchema([]string{"id", "sessionId", "kind", "mediaType", "digest", "safePreview", "expiresAt", "createdAt"}, map[string]JSONSchema{
		"id": uuid, "sessionId": uuid, "turnId": uuid, "kind": stringSchema, "mediaType": stringSchema,
		"digest": digest, "safePreview": openObject,
		"expiresAt": timestamp, "createdAt": timestamp,
	}),
	"AgentArtifactDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentArtifact")}),
	"AgentArtifactContent": objectSchema([]string{"id", "mediaType", "encoding", "content", "digest"}, map[string]JSONSchema{
		"id": uuid, "mediaType": stringSchema, "encoding": {Type: "string", Enum: []string{"base64"}},
		"content": {Type: "string", Format: "byte"}, "digest": digest,
	}),
	"AgentArtifactContentDetail":    objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("AgentArtifactContent")}),
	"AgentPublicationCommitRequest": objectSchema([]string{"planDigest"}, map[string]JSONSchema{"planDigest": digest}),
}

func agentSchemas() map[string]JSONSchema {
	return cloneSchemas(agentSchemaCatalog)
}

func cloneSchemas(source map[string]JSONSchema) map[string]JSONSchema {
	result := make(map[string]JSONSchema, len(source))
	for name, schema := range source {
		result[name] = schema
	}
	return result
}
