package managementapi

func routingOperations() []OperationContract {
	var operations []OperationContract

	routingRead := Require("routing.read", "target")
	routingManage := Require("routing.manage", "target")
	modelWrite := RequireAll(
		routingManage,
		RequireWhen("provider_credential_referenced", Require("provider_credential.use", "credential")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/routing/models", "Models", ScopeResultSet, routingRead, paginated()),
		operation(MethodGET, BasePath+"/routing/model-cards", "Models", ScopeResultSet, routingRead, paginated()),
		operation(MethodPOST, BasePath+"/routing/models", "Models", ScopeCompound, modelWrite),
		operation(MethodPOST, BasePath+"/routing/models:bulk-import", "Models", ScopeCompound,
			RequireAll(
				routingManage,
				Require("provider_catalog.read", "request_namespace"),
				RequireWhen("provider_credential_referenced", Require("provider_credential.use", "credential")),
			), asynchronous(), noRevision()),
		operation(MethodGET, BasePath+"/routing/models/{modelId}", "Models", ScopeResource, routingRead),
		operation(MethodPATCH, BasePath+"/routing/models/{modelId}", "Models", ScopeCompound, modelWrite),
		operation(MethodDELETE, BasePath+"/routing/models/{modelId}", "Models", ScopeResource, routingManage, casRevision()),
		operation(MethodPOST, BasePath+"/routing/models/{modelId}:probe", "Models", ScopeCompound, modelWrite,
			noIdempotency(), noRevision()),
	)

	operations = append(operations,
		operation(MethodGET, BasePath+"/routing/recipes", "Recipes", ScopeResultSet, routingRead, paginated()),
		operation(MethodPOST, BasePath+"/routing/recipes", "Recipes", ScopeResource, routingManage),
		operation(MethodGET, BasePath+"/routing/recipes/{recipeId}", "Recipes", ScopeResource, routingRead),
		operation(MethodPATCH, BasePath+"/routing/recipes/{recipeId}", "Recipes", ScopeResource, routingManage),
		operation(MethodDELETE, BasePath+"/routing/recipes/{recipeId}", "Recipes", ScopeResource, routingManage, casRevision()),
	)

	entrypointWrite := RequireAll(
		Require("routing.manage", "target"),
		Require("routing.read", "all_dependencies"),
	)
	entrypointRead := RequireAll(
		Require("routing.read", "target"),
		RequireWhen("entrypoint_topology_requested", Require("routing.read", "all_dependencies")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/routing/entrypoints", "Entrypoints", ScopeResultSet, entrypointRead, paginated()),
		operation(MethodPOST, BasePath+"/routing/entrypoints", "Entrypoints", ScopeCompound, entrypointWrite),
		operation(MethodGET, BasePath+"/routing/entrypoints/{entrypointId}", "Entrypoints", ScopeCompound, entrypointRead),
		operation(MethodPATCH, BasePath+"/routing/entrypoints/{entrypointId}", "Entrypoints", ScopeCompound, entrypointWrite),
		operation(MethodDELETE, BasePath+"/routing/entrypoints/{entrypointId}", "Entrypoints", ScopeResource,
			Require("routing.manage", "target"), casRevision()),
		operation(MethodPOST, BasePath+"/routing/entrypoints/{entrypointId}:publish", "Entrypoints", ScopeCompound,
			entrypointWrite, asynchronous(), casRevision()),
		operation(MethodPOST, BasePath+"/routing/entrypoints/{entrypointId}:unpublish", "Entrypoints", ScopeResource,
			Require("routing.manage", "target"), asynchronous(), casRevision()),
		operation(MethodPOST, BasePath+"/routing/entrypoints/{entrypointId}:resolve", "Entrypoints", ScopeCompound,
			RequireAll(
				Require("routing.read", "target"),
				RequireWhen("entrypoint_resolution_matched", Require("routing.read", "all_dependencies")),
				RequireWhen("routing_subject_supplied", Require("routing_context.read", "subject")),
				RequireWhen("routing_context_override_requested", Require("routing_context.manage", "subject")),
			), noIdempotency(), noRevision()),
	)

	operations = append(operations,
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/routing/snapshots", "Routing Snapshots", ScopeNamespace,
			Require("routing.read", "path_namespace"), paginated()),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/routing/snapshots/{routingRevision}", "Routing Snapshots", ScopeNamespace,
			Require("routing.read", "path_namespace")),
	)

	providerCatalogRead := Require("provider_catalog.read", "request_namespace")
	operations = append(operations,
		operation(MethodPOST, BasePath+"/provider-catalog:bootstrap", "Providers", ScopeCluster,
			Require("cluster.manage", "cluster"), noIdempotency(), noRevision()),
		operation(MethodPOST, BasePath+"/provider-catalog:activate", "Providers", ScopeCluster,
			Require("cluster.manage", "cluster"), noIdempotency(), noRevision()),
		operation(MethodGET, BasePath+"/providers", "Providers", ScopeNamespace, providerCatalogRead, paginated()),
		operation(MethodGET, BasePath+"/providers/{providerId}", "Providers", ScopeNamespace, providerCatalogRead),
		operation(MethodPOST, BasePath+"/providers/{providerId}:discover-models", "Providers", ScopeCompound,
			RequireAll(
				providerCatalogRead,
				RequireWhen("provider_credential_supplied", RequireAll(
					Require("provider_credential.read", "credential"),
					Require("provider_credential.use", "credential"),
				)),
				RequireWhen("no_provider_credential_supplied", Require("routing.manage", "request_namespace")),
			), providerPaginated(), noIdempotency(), noRevision()),
	)

	credentialRead := Require("provider_credential.read", "credential")
	credentialManage := Require("provider_credential.manage", "credential")
	operations = append(operations,
		operation(MethodGET, BasePath+"/provider-credentials", "Provider Credentials", ScopeResultSet, credentialRead, paginated()),
		operation(MethodPOST, BasePath+"/provider-credentials", "Provider Credentials", ScopeNamespace,
			Require("provider_credential.manage", "request_namespace"),
			secret(SecretInputBody, SecretOutputNone, true)),
		operation(MethodGET, BasePath+"/provider-credentials/{credentialId}", "Provider Credentials", ScopeResource, credentialRead),
		operation(MethodPATCH, BasePath+"/provider-credentials/{credentialId}", "Provider Credentials", ScopeResource, credentialManage,
			secret(SecretInputBody, SecretOutputNone, true)),
		operation(MethodDELETE, BasePath+"/provider-credentials/{credentialId}", "Provider Credentials", ScopeResource, credentialManage,
			casRevision()),
		operation(MethodPOST, BasePath+"/provider-credentials/{credentialId}:rotate", "Provider Credentials", ScopeResource, credentialManage,
			secret(SecretInputBody, SecretOutputNone, true), casRevision()),
	)

	return operations
}
