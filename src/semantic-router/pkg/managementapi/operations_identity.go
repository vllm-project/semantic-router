package managementapi

func identityOperations() []OperationContract {
	operations := identityBootstrapOperations()

	issuerRead := Require("identity_issuer.read", "cluster")
	issuerManage := Require("identity_issuer.manage", "cluster")
	issuerOperations := resourceCRUD("Identity Issuers", BasePath+"/trusted-identity-issuers", "issuerId", issuerRead, issuerManage, ScopeCluster)
	for index := range issuerOperations {
		if issuerOperations[index].Method == MethodDELETE {
			casRevision()(&issuerOperations[index])
		}
	}
	operations = append(operations, issuerOperations...)
	operations = append(operations,
		operation(MethodPOST, BasePath+"/trusted-identity-issuers/{issuerId}:refresh-keys", "Identity Issuers", ScopeCluster, issuerManage),
	)

	mappingManage := RequireAll(Require("identity_issuer.manage", "cluster"), Require("principal.manage", "cluster"))
	mappingOperations := resourceCRUD("mTLS Identity", BasePath+"/mtls-identity-mappings", "mappingId", issuerRead, mappingManage, ScopeCluster)
	mappingOperations[len(mappingOperations)-1].Revision = RevisionCAS
	operations = append(operations, mappingOperations...)

	roleRead := Require("management_role.read", "target")
	roleManage := Require("management_role.manage", "target")
	operations = append(operations, resourceCRUD("Management Roles", BasePath+"/management-roles", "roleId", roleRead, roleManage, ScopeResource)...)

	principalRead := Require("principal.read", "cluster")
	principalManage := Require("principal.manage", "cluster")
	operations = append(operations, resourceCRUD("Management Principals", BasePath+"/management-principals", "principalId", principalRead, principalManage, ScopeCluster)...)
	operations = append(operations,
		operation(MethodGET, BasePath+"/management-principals/{principalId}/user-links", "Management Principals", ScopeCluster, principalRead, paginated()),
		operation(MethodGET, BasePath+"/management-principals/{principalId}/management-sessions", "Management Principals", ScopeCluster, principalRead, paginated()),
		operation(MethodPOST, BasePath+"/management-principals/{principalId}/management-sessions:revoke-all", "Management Principals", ScopeCluster, principalManage, noRevision()),
	)

	directoryRead := RequireAny(Require("principal_directory.read", "path_namespace"), Require("principal_link.read", "path_namespace"))
	operations = append(operations,
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/principal-directory", "Principal Links", ScopeNamespace, directoryRead, paginated()),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/principal-directory/{principalId}", "Principal Links", ScopeNamespace, directoryRead),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/principal-user-links", "Principal Links", ScopeNamespace, Require("principal_link.read", "path_namespace"), paginated()),
		operation(MethodPUT, BasePath+"/namespaces/{namespaceId}/principal-user-links/{principalId}", "Principal Links", ScopeCompound,
			RequireAll(
				Require("principal_link.manage", "path_namespace"),
				Require("user.manage", "current_owner"),
				Require("user.manage", "target_owner"),
			)),
		operation(MethodDELETE, BasePath+"/namespaces/{namespaceId}/principal-user-links/{principalId}", "Principal Links", ScopeCompound,
			RequireAll(Require("principal_link.manage", "path_namespace"), Require("user.manage", "current_owner")), casRevision()),
	)

	bindingRead := Require("role_binding.read", "target")
	bindingManage := Require("role_binding.manage", "target")
	operations = append(operations, resourceCRUD("Role Bindings", BasePath+"/role-bindings", "bindingId", bindingRead, bindingManage, ScopeResource)...)

	serviceRead := Require("service_account.read", "target")
	serviceManage := Require("service_account.manage", "target")
	serviceAccountOperations := []OperationContract{
		operation(MethodGET, BasePath+"/service-accounts", "Service Accounts", ScopeResource, serviceRead, paginated()),
		operation(MethodPOST, BasePath+"/service-accounts", "Service Accounts", ScopeResource, serviceManage,
			secret(SecretInputNone, SecretOutputOneTime, true)),
		operation(MethodGET, BasePath+"/service-accounts/{serviceAccountId}", "Service Accounts", ScopeResource, serviceRead),
		operation(MethodPATCH, BasePath+"/service-accounts/{serviceAccountId}", "Service Accounts", ScopeResource, serviceManage),
		operation(MethodDELETE, BasePath+"/service-accounts/{serviceAccountId}", "Service Accounts", ScopeResource, serviceManage, casRevision()),
	}
	operations = append(operations, serviceAccountOperations...)
	operations = append(operations,
		operation(MethodGET, BasePath+"/service-accounts/{serviceAccountId}/credentials", "Service Accounts", ScopeResource, serviceRead, paginated()),
		operation(MethodPOST, BasePath+"/service-accounts/{serviceAccountId}/credentials:rotate", "Service Accounts", ScopeResource, serviceManage,
			secret(SecretInputNone, SecretOutputOneTime, true)),
		operation(MethodDELETE, BasePath+"/service-accounts/{serviceAccountId}/credentials/{credentialId}", "Service Accounts", ScopeResource, serviceManage, casRevision()),
	)

	invitationRead := Require("invitation.read", "target")
	invitationManage := RequireAll(
		Require("invitation.manage", "target"),
		RequireWhen("team_role_requested", Require("membership.manage", "team")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/invitations", "Invitations", ScopeNamespace, invitationRead, paginated()),
		operation(MethodPOST, BasePath+"/invitations", "Invitations", ScopeCompound, invitationManage, secret(SecretInputNone, SecretOutputOneTime, true)),
		operation(MethodGET, BasePath+"/invitations/{invitationId}", "Invitations", ScopeResource, invitationRead),
		operation(MethodDELETE, BasePath+"/invitations/{invitationId}", "Invitations", ScopeCompound, invitationManage, casRevision()),
		operation(MethodPOST, BasePath+"/invitations/{invitationId}:rotate-token", "Invitations", ScopeCompound, invitationManage,
			secret(SecretInputNone, SecretOutputOneTime, true), casRevision()),
	)

	return append(operations, onboardingOperation())
}

func onboardingOperation() OperationContract {
	onboarding := RequireAll(
		Require("onboarding.manage", "request_namespace"),
		Require("principal_link.manage", "request_namespace"),
		Require("user.manage", "user"),
		RequireWhen("role_binding_requested", Require("role_binding.manage", "target")),
		RequireWhen("team_membership_requested", Require("membership.manage", "team")),
		RequireWhen("first_key_requested", Require("key.manage", "owner")),
		RequireWhen("access_binding_requested", Require("access_policy.manage", "policy")),
		RequireWhen("rate_binding_requested", Require("rate_policy.manage", "policy")),
	)
	return operation(MethodPOST, BasePath+"/onboarding", "Onboarding", ScopeCompound, onboarding,
		secret(SecretInputNone, SecretOutputOneTime, true), noRevision())
}

func identityBootstrapOperations() []OperationContract {
	operations := []OperationContract{
		operation(MethodGET, BasePath+"/me", "Identity", ScopeIntrinsicSelf, Require("self.read", "intrinsic_self")),
		operation(MethodPOST, BasePath+"/auth/bootstrap", "Authentication", ScopeAuthentication, RequireSpecial("bootstrap_credential"), secret(SecretInputAuthorization, SecretOutputOneTime, false), noRevision()),
		operation(MethodPOST, BasePath+"/auth/exchange-challenges", "Authentication", ScopeAuthentication, RequireSpecial("exchange_challenge"), sensitiveNoStore(false), noIdempotency(), noRevision()),
		operation(MethodPOST, BasePath+"/auth/token-exchange", "Authentication", ScopeAuthentication, RequireSpecial("subject_token_exchange"), secret(SecretInputBody, SecretOutputAccessToken, false), noIdempotency(), noRevision()),
		operation(MethodPOST, BasePath+"/auth/service-token", "Authentication", ScopeAuthentication, RequireSpecial("service_credential_or_mtls"), secret(SecretInputAuthorization, SecretOutputAccessToken, false), noIdempotency(), noRevision()),
		operation(MethodPOST, BasePath+"/auth/recovery", "Authentication", ScopeAuthentication, RequireSpecial("recovery_credential"), secret(SecretInputAuthorization, SecretOutputNone, false), noRevision()),
		operation(MethodPOST, BasePath+"/auth/backchannel-logout", "Authentication", ScopeAuthentication, RequireSpecial("trusted_issuer_logout_token"), secret(SecretInputBody, SecretOutputNone, false), noIdempotency(), noRevision()),
		operation(MethodGET, BasePath+"/self/management-sessions", "Identity", ScopeIntrinsicSelf, Require("self.read", "intrinsic_self"), paginated()),
		operation(MethodDELETE, BasePath+"/self/management-sessions/{sessionId}", "Identity", ScopeIntrinsicSelf, Require("self.manage", "intrinsic_self"), noRevision()),
		operation(MethodPOST, BasePath+"/management-sessions/{sessionId}:revoke", "Identity", ScopeCluster, Require("principal.manage", "cluster"), noRevision()),
		operation(MethodGET, BasePath+"/self/inference-keys", "Delegation", ScopeIntrinsicSelf,
			RequireAll(Require("self.read", "intrinsic_self"), Require("delegation.use", "user")), paginated()),
		operation(MethodGET, BasePath+"/self/inference-sessions", "Delegation", ScopeIntrinsicSelf,
			RequireAll(Require("self.read", "intrinsic_self"), Require("delegation.use", "user")), paginated()),
		operation(MethodPOST, BasePath+"/self/inference-sessions", "Delegation", ScopeIntrinsicSelf,
			RequireAll(Require("self.manage", "intrinsic_self"), Require("delegation.use", "key")), secret(SecretInputNone, SecretOutputOneTime, true)),
		operation(MethodDELETE, BasePath+"/self/inference-sessions/{sessionId}", "Delegation", ScopeIntrinsicSelf,
			RequireAll(Require("self.manage", "intrinsic_self"), Require("delegation.use", "key"))),
		operation(MethodGET, BasePath+"/management-session-policy", "Identity", ScopeCluster, Require("cluster.read", "cluster")),
		operation(MethodPATCH, BasePath+"/management-session-policy", "Identity", ScopeCluster, Require("cluster.manage", "cluster")),
	}
	return append(operations, namespaceOperations()...)
}

func namespaceOperations() []OperationContract {
	namespaceRead := RequireAny(Require("cluster.read", "cluster"), Require("namespace.read", "target"))
	return []OperationContract{
		operation(MethodGET, BasePath+"/namespaces", "Namespaces", ScopeResultSet,
			RequireAny(Require("cluster.read", "cluster"), RequireWhen("namespace_list_item", Require("namespace.read", "all_returned_resources"))), paginated()),
		operation(MethodPOST, BasePath+"/namespaces", "Namespaces", ScopeCluster, Require("cluster.manage", "cluster")),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}", "Namespaces", ScopeNamespace, namespaceRead),
		operation(MethodPATCH, BasePath+"/namespaces/{namespaceId}", "Namespaces", ScopeNamespace, Require("namespace.manage", "path_namespace")),
		operation(MethodDELETE, BasePath+"/namespaces/{namespaceId}", "Namespaces", ScopeCluster, Require("cluster.manage", "cluster"), casRevision()),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/self-service-policy", "Namespaces", ScopeNamespace, Require("namespace.read", "path_namespace")),
		operation(MethodPATCH, BasePath+"/namespaces/{namespaceId}/self-service-policy", "Namespaces", ScopeCompound,
			RequireAll(
				Require("namespace.manage", "path_namespace"),
				RequireWhen("current_access_policy_default_present", Require("access_policy.manage", "current_access_policy_default")),
				RequireWhen("current_rate_policy_default_present", Require("rate_policy.manage", "current_rate_policy_default")),
				RequireWhen("target_access_policy_default_present", Require("access_policy.manage", "target_access_policy_default")),
				RequireWhen("target_rate_policy_default_present", Require("rate_policy.manage", "target_rate_policy_default")),
			)),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/management-security-policy", "Namespaces", ScopeNamespace, Require("namespace.read", "path_namespace")),
		operation(MethodPATCH, BasePath+"/namespaces/{namespaceId}/management-security-policy", "Namespaces", ScopeNamespace, Require("namespace.manage", "path_namespace")),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/routing-claim-schema", "Routing Context", ScopeNamespace,
			RequireAll(Require("namespace.read", "path_namespace"), Require("routing_context.read", "path_namespace"))),
		operation(MethodPATCH, BasePath+"/namespaces/{namespaceId}/routing-claim-schema", "Routing Context", ScopeCompound,
			RequireAll(Require("namespace.manage", "path_namespace"), Require("routing_context.manage", "path_namespace"))),
	}
}
