package managementapi

func accessOperations() []OperationContract {
	operations := accessKeyOperations()

	accessPolicyRead := Require("access_policy.read", "policy")
	accessPolicyManage := Require("access_policy.manage", "policy")
	accessPolicyWrite := RequireAll(
		accessPolicyManage,
		RequireWhen("access_policy_references_routing_resources", Require("routing.read", "all_dependencies")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/access-policies", "Access Policies", ScopeResultSet, accessPolicyRead, paginated()),
		operation(MethodPOST, BasePath+"/access-policies", "Access Policies", ScopeCompound, accessPolicyWrite),
		operation(MethodGET, BasePath+"/access-policies/{policyId}", "Access Policies", ScopeResource, accessPolicyRead),
		operation(MethodPATCH, BasePath+"/access-policies/{policyId}", "Access Policies", ScopeCompound, accessPolicyWrite),
		operation(MethodDELETE, BasePath+"/access-policies/{policyId}", "Access Policies", ScopeResource, accessPolicyManage, casRevision()),
	)

	accessBindingRead := Require("access_policy.read", "policy")
	accessBindingManage := RequireAll(
		Require("access_policy.manage", "policy"),
		subjectManageRequirement(),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/access-policy-bindings", "Access Policies", ScopeResultSet, accessBindingRead, paginated()),
		operation(MethodPOST, BasePath+"/access-policy-bindings", "Access Policies", ScopeCompound, accessBindingManage),
		operation(MethodGET, BasePath+"/access-policy-bindings/{bindingId}", "Access Policies", ScopeResource, accessBindingRead),
		operation(MethodPATCH, BasePath+"/access-policy-bindings/{bindingId}", "Access Policies", ScopeCompound, accessBindingManage),
		operation(MethodDELETE, BasePath+"/access-policy-bindings/{bindingId}", "Access Policies", ScopeCompound, accessBindingManage, casRevision()),
		operation(MethodPOST, BasePath+"/access-policy-bindings:bulk-apply", "Access Policies", ScopeCompound, accessBindingManage, asynchronous(), noRevision()),
	)

	ratePolicyRead := Require("rate_policy.read", "policy")
	ratePolicyManage := Require("rate_policy.manage", "policy")
	operations = append(operations,
		operation(MethodGET, BasePath+"/rate-limit-policies", "Rate Limit Policies", ScopeResultSet, ratePolicyRead, paginated()),
		operation(MethodPOST, BasePath+"/rate-limit-policies", "Rate Limit Policies", ScopeResource, ratePolicyManage),
		operation(MethodGET, BasePath+"/rate-limit-policies/{policyId}", "Rate Limit Policies", ScopeResource, ratePolicyRead),
		operation(MethodPATCH, BasePath+"/rate-limit-policies/{policyId}", "Rate Limit Policies", ScopeResource, ratePolicyManage),
		operation(MethodDELETE, BasePath+"/rate-limit-policies/{policyId}", "Rate Limit Policies", ScopeResource, ratePolicyManage, casRevision()),
	)

	rateBindingManage := RequireAll(ratePolicyManage, subjectManageRequirement())
	operations = append(operations,
		operation(MethodGET, BasePath+"/rate-limit-bindings", "Rate Limit Policies", ScopeResultSet, ratePolicyRead, paginated()),
		operation(MethodPOST, BasePath+"/rate-limit-bindings", "Rate Limit Policies", ScopeCompound, rateBindingManage),
		operation(MethodGET, BasePath+"/rate-limit-bindings/{bindingId}", "Rate Limit Policies", ScopeResource, ratePolicyRead),
		operation(MethodPATCH, BasePath+"/rate-limit-bindings/{bindingId}", "Rate Limit Policies", ScopeCompound, rateBindingManage),
		operation(MethodDELETE, BasePath+"/rate-limit-bindings/{bindingId}", "Rate Limit Policies", ScopeCompound, rateBindingManage, casRevision()),
		operation(MethodPOST, BasePath+"/rate-limit-bindings:bulk-apply", "Rate Limit Policies", ScopeCompound, rateBindingManage, asynchronous(), noRevision()),
	)

	accessCheck := RequireAll(
		Require("access_policy.read", "subject"),
		Require("access_policy.read", "resource"),
		Require("routing_context.read", "subject"),
		RequireWhen("routing_context_override_requested", Require("routing_context.manage", "subject")),
		RequireWhen("entrypoint_topology_requested", Require("routing.read", "all_dependencies")),
		RequireWhen("internal_usage_dimensions_requested", Require("usage.internal_dimensions.read", "all_dependencies")),
	)
	operations = append(operations,
		operation(MethodPOST, BasePath+"/access:check", "Access Decisions", ScopeCompound, accessCheck, noRevision()),
	)

	fenceRead := Require("quota.read", "all_affected_bindings")
	fenceDetail := RequireAll(
		fenceRead,
		RequireWhen("internal_usage_dimensions_requested", Require("usage.internal_dimensions.read", "all_affected_bindings")),
		RequireWhen("fence_payload_evidence_requested", Require("log_payload.read", "attributed_subject")),
		RequireWhen("fence_actor_or_audit_fields_requested", RequireAny(
			Require("audit.read", "all_affected_bindings"),
			Require("quota.reconcile", "all_affected_bindings"),
		)),
	)
	fenceReconcile := RequireAll(
		Require("quota.reconcile", "all_affected_bindings"),
		RequireWhen("fence_actual_reconciliation", Require("usage.internal_dimensions.read", "all_affected_bindings")),
		RequireWhen("fence_payload_evidence_requested", Require("log_payload.read", "attributed_subject")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/unknown-usage-fences", "Quota", ScopeResultSet, fenceRead, paginated()),
		operation(MethodGET, BasePath+"/unknown-usage-fences/{fenceId}", "Quota", ScopeCompound, fenceDetail),
		operation(MethodPOST, BasePath+"/unknown-usage-fences/{fenceId}:reconcile", "Quota", ScopeCompound, fenceReconcile, asynchronous(), casRevision()),
	)

	return operations
}

func accessKeyOperations() []OperationContract {
	keyRead := Require("key.read", "key")
	keyManage := Require("key.manage", "key")
	keyCreate := RequireAll(
		Require("key.manage", "owner"),
		RequireWhen("access_policy_binding_requested", Require("access_policy.manage", "access_policy")),
		RequireWhen("rate_policy_binding_requested", Require("rate_policy.manage", "rate_policy")),
		RequireWhen("inline_rate_policy_requested", Require("rate_policy.manage", "request_namespace")),
	)
	operations := []OperationContract{
		operation(MethodGET, BasePath+"/api-keys", "API Keys", ScopeResultSet, keyRead, paginated()),
		operation(MethodPOST, BasePath+"/api-keys", "API Keys", ScopeCompound, keyCreate,
			secret(SecretInputNone, SecretOutputOneTime, true)),
		operation(MethodGET, BasePath+"/api-keys/{keyId}", "API Keys", ScopeResource, keyRead),
		operation(MethodPATCH, BasePath+"/api-keys/{keyId}", "API Keys", ScopeResource, keyManage),
		operation(MethodDELETE, BasePath+"/api-keys/{keyId}", "API Keys", ScopeResource, keyManage, casRevision()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}:enable", "API Keys", ScopeResource, keyManage, casRevision()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}:disable", "API Keys", ScopeResource, keyManage, casRevision()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}:renew", "API Keys", ScopeResource, keyManage, casRevision()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}:reassign", "API Keys", ScopeCompound,
			RequireAll(
				keyManage,
				RequireWhen("current_user_owner", Require("user.manage", "current_owner")),
				RequireWhen("current_team_owner", Require("team.manage", "current_owner")),
				RequireWhen("target_user_owner", Require("user.manage", "target_owner")),
				RequireWhen("target_team_owner", Require("team.manage", "target_owner")),
			), casRevision()),
	}
	credentialManage := RequireAll(keyRead, keyManage)
	operations = append(operations,
		operation(MethodGET, BasePath+"/api-keys/{keyId}/credentials", "API Key Credentials", ScopeResource, keyRead, paginated()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}/credentials:rotate", "API Key Credentials", ScopeResource, credentialManage,
			secret(SecretInputNone, SecretOutputOneTime, true), casRevision()),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}/credentials/{credentialId}:reveal", "API Key Credentials", ScopeCompound,
			RequireAll(keyRead, Require("key.reveal", "key")),
			secret(SecretInputNone, SecretOutputOneTime, true), noIdempotency(), noRevision()),
		operation(MethodDELETE, BasePath+"/api-keys/{keyId}/credentials/{credentialId}", "API Key Credentials", ScopeResource, credentialManage, casRevision()),
	)
	delegationManage := RequireAll(keyRead, Require("delegation.manage", "key"))
	return append(operations,
		operation(MethodGET, BasePath+"/api-keys/{keyId}/inference-sessions", "Delegation", ScopeResource, delegationManage, paginated()),
		operation(MethodDELETE, BasePath+"/api-keys/{keyId}/inference-sessions/{sessionId}", "Delegation", ScopeResource, delegationManage),
		operation(MethodPOST, BasePath+"/api-keys/{keyId}/inference-sessions:revoke-all", "Delegation", ScopeResource, delegationManage),
		operation(MethodGET, BasePath+"/api-keys/{keyId}/effective-policy", "API Keys", ScopeCompound,
			RequireAll(keyRead, Require("access_policy.read", "key"), Require("rate_policy.read", "key"))),
		operation(MethodGET, BasePath+"/api-keys/{keyId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(keyRead, Require("routing_context.read", "key"))),
		operation(MethodGET, BasePath+"/api-keys/{keyId}/routing-catalog", "Routing Catalog", ScopeCompound,
			RequireAll(keyRead, Require("access_policy.read", "key"), Require("routing_context.read", "key"))),
		operation(MethodPUT, BasePath+"/api-keys/{keyId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(keyManage, Require("routing_context.manage", "key")), casRevision()),
		operation(MethodGET, BasePath+"/api-keys/{keyId}/quota", "Quota", ScopeCompound,
			RequireAll(keyRead, Require("quota.read", "all_returned_bindings"))),
		operation(MethodGET, BasePath+"/api-keys/{keyId}/usage", "Usage", ScopeCompound,
			RequireAll(keyRead, Require("usage.read", "key"))),
	)
}

func subjectManageRequirement() PermissionExpression {
	return RequireAll(
		RequireWhen("user_owner", Require("user.manage", "subject")),
		RequireWhen("team_owner", Require("team.manage", "subject")),
		RequireWhen("key_owner", Require("key.manage", "subject")),
	)
}
