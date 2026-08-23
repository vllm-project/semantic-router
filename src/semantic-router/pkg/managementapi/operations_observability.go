package managementapi

func observabilityOperations() []OperationContract {
	var operations []OperationContract

	usageRead := RequireAll(
		Require("usage.read", "all_returned_resources"),
		RequireWhen("internal_usage_dimensions_requested", Require("usage.internal_dimensions.read", "all_returned_resources")),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/statistics", "Statistics", ScopeResultSet,
			Require("usage.read", "all_returned_resources")),
		operation(MethodGET, BasePath+"/usage", "Usage", ScopeResultSet, usageRead),
		operation(MethodGET, BasePath+"/usage/series", "Usage", ScopeResultSet, usageRead),
		operation(MethodGET, BasePath+"/usage/breakdowns", "Usage", ScopeResultSet, usageRead),
	)

	operations = append(operations,
		operation(MethodGET, BasePath+"/request-logs", "Request Logs", ScopeResultSet,
			Require("log.read", "all_returned_resources"), paginated()),
		operation(MethodGET, BasePath+"/namespaces/{namespaceId}/request-logs/{admissionId}", "Request Logs", ScopeCompound,
			RequireAll(
				Require("log.read", "attributed_subject"),
				RequireWhen("request_log_payload_requested", Require("log_payload.read", "attributed_subject")),
			)),
		operation(MethodGET, BasePath+"/audit-events", "Audit", ScopeResultSet,
			Require("audit.read", "all_returned_resources"), paginated()),
		operation(MethodGET, BasePath+"/runtime-diagnostics", "Diagnostics", ScopeCluster, Require("health.read", "cluster")),
	)

	operationRead := RequireAll(
		RequireAll(
			RequireWhen("operation_originator", Require("self.read", "intrinsic_self")),
			RequireWhen("cross_actor_operation", Require("operation.read", "operation_targets")),
		),
		RequireRecordedPermission("original_domain_read"),
	)
	operationCancel := RequireAll(
		RequireAll(
			RequireWhen("operation_originator", Require("self.manage", "intrinsic_self")),
			RequireWhen("cross_actor_operation", Require("operation.manage", "operation_targets")),
		),
		RequireRecordedPermission("original_domain_mutation"),
	)
	operations = append(operations,
		operation(MethodGET, BasePath+"/operations", "Operations", ScopeOperation, operationRead, paginated()),
		operation(MethodGET, BasePath+"/operations/{operationId}", "Operations", ScopeOperation, operationRead, operationRevision()),
		operation(MethodPOST, BasePath+"/operations/{operationId}:cancel", "Operations", ScopeOperation, operationCancel, casRevision()),
	)

	return operations
}

func operationRevision() operationOption {
	return func(operation *OperationContract) { operation.Revision = RevisionReturns }
}
