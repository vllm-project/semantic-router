package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "namespaces", Schemas: namespaceSchemas,
		RequestSchema: namespaceRequestSchema, ResponseSchema: namespaceResponseSchema,
		ExtraParameters: namespaceParameters,
	})
}

func namespaceRequestSchema(contract OperationContract) (string, bool) {
	schemas := map[string]string{
		"POST " + BasePath + "/namespaces":                                           "NamespaceCreateRequest",
		"PATCH " + BasePath + "/namespaces/{namespaceId}":                            "NamespacePatchRequest",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/self-service-policy":        "SelfServicePolicyPatchRequest",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/management-security-policy": "NamespaceManagementSecurityPolicyPatchRequest",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/routing-claim-schema":       "RoutingClaimSchemaPatchRequest",
	}
	value, found := schemas[string(contract.Method)+" "+contract.Path]
	return value, found
}

func namespaceResponseSchema(contract OperationContract) (JSONSchema, bool) {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "GET " + BasePath + "/namespaces":
		return refSchema("NamespacePage"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}":
		return refSchema("NamespaceDetail"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}/self-service-policy":
		return refSchema("SelfServicePolicyDetail"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}/management-security-policy":
		return refSchema("NamespaceManagementSecurityPolicyDetail"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}/routing-claim-schema":
		return refSchema("RoutingClaimSchemaDetail"), true
	}
	switch key {
	case "POST " + BasePath + "/namespaces",
		"PATCH " + BasePath + "/namespaces/{namespaceId}",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/self-service-policy",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/management-security-policy",
		"PATCH " + BasePath + "/namespaces/{namespaceId}/routing-claim-schema":
		return refSchema("MutationReceipt"), true
	}
	return JSONSchema{}, false
}

func namespaceParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method == MethodGET && contract.Path == BasePath+"/namespaces" {
		return []OpenAPIParameter{{Name: "status", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"active", "disabled"}}}}
	}
	return nil
}

func namespaceSchemas() map[string]JSONSchema {
	text, uuid := JSONSchema{Type: "string"}, JSONSchema{Type: "string", Format: "uuid"}
	nullableUUID := JSONSchema{OneOf: []JSONSchema{uuid, {Type: "string", Enum: []string{""}}}}
	dateTimeSchema, integer := JSONSchema{Type: "string", Format: "date-time"}, JSONSchema{Type: "integer", Format: "int64"}
	boolean := JSONSchema{Type: "boolean"}
	namespace := objectSchema([]string{"namespaceId", "name", "quotaPartitionId", "billingCurrency", "status", "revision", "runtimeEpoch", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"namespaceId": uuid, "name": text, "quotaPartitionId": text,
		"billingCurrency": {Type: "string", Pattern: `^[A-Z]{3}$`}, "status": {Type: "string", Enum: []string{"active", "disabled"}},
		"revision": integer, "runtimeEpoch": integer, "createdAt": dateTimeSchema, "updatedAt": dateTimeSchema,
	})
	human := objectSchema([]string{"minimumAal", "acceptedAmr", "maxAuthenticationAgeSeconds"}, map[string]JSONSchema{
		"minimumAal": {Type: "string", Enum: []string{"aal1", "aal2", "aal3"}}, "acceptedAmr": arraySchema(text),
		"maxAuthenticationAgeSeconds": integer,
	})
	workload := objectSchema([]string{"minimumWorkloadClass", "maxSourceAgeSeconds"}, map[string]JSONSchema{
		"minimumWorkloadClass": {Type: "string", Enum: []string{"workload_standard", "workload_strong"}}, "maxSourceAgeSeconds": integer,
	})
	authRequirement := JSONSchema{OneOf: []JSONSchema{
		objectSchema([]string{"kind", "human"}, map[string]JSONSchema{"kind": {Type: "string", Enum: []string{"human"}}, "human": human}),
		objectSchema([]string{"kind", "workload"}, map[string]JSONSchema{"kind": {Type: "string", Enum: []string{"workload"}}, "workload": workload}),
	}}
	actionRequirements := JSONSchema{Type: "object", PatternProperties: map[string]JSONSchema{`^[a-z][a-z0-9._]{0,127}$`: arraySchema(authRequirement)}, AdditionalProperties: boolPointer(false)}
	selfService := objectSchema([]string{"namespaceId", "maxKeysPerUser", "maxDelegatedSessions", "delegatedSessionTtlSeconds", "allowTeamKeyDelegation", "automaticFirstKey", "teamAdminCapabilities", "revision", "seedVersion", "updatedAt"}, map[string]JSONSchema{
		"namespaceId": uuid, "maxKeysPerUser": boundedIntegerSchema(0, 1000), "maxDelegatedSessions": boundedIntegerSchema(0, 10000),
		"delegatedSessionTtlSeconds": boundedIntegerSchema(60, 86400), "allowTeamKeyDelegation": boolean, "automaticFirstKey": boolean,
		"teamAdminCapabilities": arraySchema(JSONSchema{Type: "string", Enum: []string{"membership.manage", "key.manage"}}),
		"defaultAccessPolicyId": nullableUUID, "defaultRateLimitPolicyId": nullableUUID, "revision": integer, "seedVersion": integer, "updatedAt": dateTimeSchema,
	})
	claimDefinition := objectSchema([]string{"kind"}, map[string]JSONSchema{
		"kind": {Type: "string", Enum: []string{"string", "boolean", "integer"}}, "minimum": integer, "maximum": integer, "maxLength": boundedIntegerSchema(1, 4096),
	})
	claimMap := JSONSchema{Type: "object", PatternProperties: map[string]JSONSchema{`^[A-Za-z][A-Za-z0-9_.-]{0,63}$`: claimDefinition}, AdditionalProperties: boolPointer(false)}
	security := objectSchema([]string{"namespaceId", "actionRequirements", "seedVersion", "revision", "updatedAt"}, map[string]JSONSchema{
		"namespaceId": uuid, "actionRequirements": actionRequirements, "seedVersion": integer, "revision": integer, "updatedAt": dateTimeSchema,
	})
	claims := objectSchema([]string{"namespaceId", "definitions", "revision", "updatedAt"}, map[string]JSONSchema{
		"namespaceId": uuid, "definitions": claimMap, "revision": integer, "updatedAt": dateTimeSchema,
	})
	return map[string]JSONSchema{
		"Namespace":                                     namespace,
		"NamespacePage":                                 objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(namespace), "page": refSchema("PageInfo")}),
		"NamespaceDetail":                               objectSchema([]string{"data"}, map[string]JSONSchema{"data": namespace}),
		"NamespaceCreateRequest":                        objectSchema([]string{"name", "billingCurrency", "reason"}, map[string]JSONSchema{"name": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)}, "billingCurrency": {Type: "string", Pattern: `^[A-Z]{3}$`}, "reason": text}),
		"NamespacePatchRequest":                         objectSchema([]string{"status", "reason"}, map[string]JSONSchema{"status": {Type: "string", Enum: []string{"active", "disabled"}}, "reason": text}),
		"SelfServicePolicy":                             selfService,
		"SelfServicePolicyDetail":                       objectSchema([]string{"data"}, map[string]JSONSchema{"data": selfService}),
		"SelfServicePolicyPatchRequest":                 objectSchema([]string{"reason"}, map[string]JSONSchema{"maxKeysPerUser": boundedIntegerSchema(0, 1000), "maxDelegatedSessions": boundedIntegerSchema(0, 10000), "delegatedSessionTtlSeconds": boundedIntegerSchema(60, 86400), "allowTeamKeyDelegation": boolean, "automaticFirstKey": boolean, "teamAdminCapabilities": arraySchema(JSONSchema{Type: "string", Enum: []string{"membership.manage", "key.manage"}}), "defaultAccessPolicyId": nullableUUID, "defaultRateLimitPolicyId": nullableUUID, "reason": text}),
		"NamespaceManagementSecurityPolicy":             security,
		"NamespaceManagementSecurityPolicyDetail":       objectSchema([]string{"data"}, map[string]JSONSchema{"data": security}),
		"NamespaceManagementSecurityPolicyPatchRequest": objectSchema([]string{"actionRequirements", "reason"}, map[string]JSONSchema{"actionRequirements": actionRequirements, "reason": text}),
		"RoutingClaimDefinition":                        claimDefinition,
		"RoutingClaimSchema":                            claims,
		"RoutingClaimSchemaDetail":                      objectSchema([]string{"data"}, map[string]JSONSchema{"data": claims}),
		"RoutingClaimSchemaPatchRequest":                objectSchema([]string{"definitions", "reason"}, map[string]JSONSchema{"definitions": claimMap, "reason": text}),
	}
}
