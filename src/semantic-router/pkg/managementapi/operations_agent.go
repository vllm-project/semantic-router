package managementapi

func agentOperations() []OperationContract {
	agentRead := Require("agent.read", "target")
	agentUse := Require("agent.use", "target")
	agentManage := Require("agent.manage", "target")
	toolRead := Require("tool.read", "target")
	toolManage := Require("tool.manage", "target")

	operations := agentResourceCRUD("Agent Profiles", BasePath+"/agent-profiles", "profile", agentRead, agentManage)
	operations = append(operations, agentResourceCRUD(
		"Agent Skills", BasePath+"/agent-skills", "skill", agentRead, agentManage,
	)...)
	operations = append(operations,
		operation(MethodGET, BasePath+"/agent-tools", "Agent Tools", ScopeResultSet, toolRead, paginated()),
		operation(MethodGET, BasePath+"/agent-tool-credentials", "Agent Tool Credentials", ScopeResultSet, toolRead, paginated()),
		operation(MethodPOST, BasePath+"/agent-tool-credentials", "Agent Tool Credentials", ScopeNamespace,
			Require("tool.manage", "request_namespace"), secret(SecretInputBody, SecretOutputNone, true)),
		operation(MethodGET, BasePath+"/agent-tool-credentials/{credential}", "Agent Tool Credentials", ScopeResource, toolRead, operationRevision()),
		operation(MethodPATCH, BasePath+"/agent-tool-credentials/{credential}", "Agent Tool Credentials", ScopeResource,
			toolManage, secret(SecretInputBody, SecretOutputNone, true)),
		operation(MethodDELETE, BasePath+"/agent-tool-credentials/{credential}", "Agent Tool Credentials", ScopeResource,
			toolManage, casRevision()),
		operation(MethodPOST, BasePath+"/agent-tool-credentials/{credential}:rotate", "Agent Tool Credentials", ScopeResource,
			toolManage, secret(SecretInputBody, SecretOutputNone, true), casRevision()),
	)
	operations = append(operations, agentResourceCRUD(
		"Agent Tool Sources", BasePath+"/agent-tool-sources", "source", toolRead, toolManage,
	)...)
	operations = append(operations,
		operation(MethodPOST, BasePath+"/agent-tool-sources/{source}:test", "Agent Tool Sources",
			ScopeResource, RequireAll(toolRead, Require("tool.invoke", "target")), operationRevision()),
		operation(MethodPOST, BasePath+"/agent-tool-sources/{source}:approve", "Agent Tool Sources",
			ScopeResource, toolManage, casRevision()),
	)

	sessionCreate := RequireAll(
		Require("agent.use", "attributed_subject"),
		Require("delegation.use", "attributed_subject"),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/agent-sessions", "Agent Sessions", ScopeResultSet, agentRead, paginated()),
		operation(MethodPOST, BasePath+"/agent-sessions", "Agent Sessions", ScopeCompound, sessionCreate),
		operation(MethodGET, BasePath+"/agent-sessions/{session}", "Agent Sessions", ScopeResource, agentRead, operationRevision()),
		operation(MethodPATCH, BasePath+"/agent-sessions/{session}", "Agent Sessions", ScopeResource, agentUse),
		operation(MethodDELETE, BasePath+"/agent-sessions/{session}", "Agent Sessions", ScopeResource, agentUse, casRevision()),
		operation(MethodPOST, BasePath+"/agent-sessions/{session}/turns", "Agent Turns", ScopeResource, agentUse, noRevision()),
		operation(MethodGET, BasePath+"/agent-sessions/{session}/turns", "Agent Turns", ScopeResource, agentRead, paginated(), noRevision()),
		operation(MethodGET, BasePath+"/agent-sessions/{session}/events", "Agent Events", ScopeResource, agentRead, paginated(), noRevision()),
		operation(MethodPOST, BasePath+"/agent-sessions/{session}/turns/{turn}:cancel", "Agent Turns", ScopeResource,
			agentUse, noRevision()),
		operation(MethodGET, BasePath+"/agent-artifacts/{artifact}", "Agent Artifacts", ScopeResource, agentRead),
		operation(MethodGET, BasePath+"/agent-artifacts/{artifact}/content", "Agent Artifacts", ScopeResource,
			agentRead, sensitiveNoStore(true)),
	)

	operations = append(operations,
		operation(MethodPOST, BasePath+"/publication-plans/{plan}:commit", "Routing Publication",
			ScopeCompound,
			RequireAll(
				Require("routing.publish", "target"),
				Require("routing.read", "all_dependencies"),
			),
			asynchronous(), casRevision()),
	)
	return operations
}

// An Agent collection is namespace-scoped on create, result-scoped on list,
// and resource-scoped only after the Router has resolved a concrete ID. This
// avoids authorizing POST against a client-invented resource scope.
func agentResourceCRUD(
	tag, basePath, idParameter string,
	read, manage PermissionExpression,
) []OperationContract {
	detail := basePath + "/{" + idParameter + "}"
	return []OperationContract{
		operation(MethodGET, basePath, tag, ScopeResultSet, read, paginated()),
		operation(MethodPOST, basePath, tag, ScopeNamespace, manage),
		operation(MethodGET, detail, tag, ScopeResource, read, operationRevision()),
		operation(MethodPATCH, detail, tag, ScopeResource, manage),
		operation(MethodDELETE, detail, tag, ScopeResource, manage, casRevision()),
	}
}
