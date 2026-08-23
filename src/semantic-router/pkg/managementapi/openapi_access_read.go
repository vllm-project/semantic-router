package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "access-read", Schemas: accessReadSchemas,
		RequestSchema: accessReadRequestSchema, ResponseSchema: accessReadResponseSchema,
	})
}

func accessReadSchemas() map[string]JSONSchema {
	text := JSONSchema{Type: "string"}
	uuid := JSONSchema{Type: "string", Format: "uuid"}
	timestamp := JSONSchema{Type: "string", Format: "date-time"}
	revision := JSONSchema{Type: "integer", Format: "int64", Minimum: intPointer(1)}
	source := objectSchema([]string{"subjectType", "subjectId"}, map[string]JSONSchema{
		"subjectType": {Type: "string", Enum: []string{"user", "team", "api_key"}}, "subjectId": uuid,
	})
	claimValue := JSONSchema{OneOf: []JSONSchema{
		objectSchema([]string{"kind", "string"}, map[string]JSONSchema{
			"kind": {Type: "string", Enum: []string{"string"}}, "string": {Type: "string", MaxLength: intPointer(4096)},
		}),
		objectSchema([]string{"kind", "boolean"}, map[string]JSONSchema{
			"kind": {Type: "string", Enum: []string{"boolean"}}, "boolean": {Type: "boolean"},
		}),
		objectSchema([]string{"kind", "integer"}, map[string]JSONSchema{
			"kind": {Type: "string", Enum: []string{"integer"}}, "integer": {Type: "integer", Format: "int64"},
		}),
	}}
	stored := objectSchema([]string{"name", "value", "revision", "updatedAt"}, map[string]JSONSchema{
		"name": {Type: "string", Pattern: `^[A-Za-z][A-Za-z0-9_.-]{0,63}$`}, "value": claimValue,
		"revision": revision, "updatedAt": timestamp,
	})
	effective := objectSchema([]string{"name", "value", "source"}, map[string]JSONSchema{
		"name": {Type: "string", Pattern: `^[A-Za-z][A-Za-z0-9_.-]{0,63}$`}, "value": claimValue,
		"source": source, "revision": revision, "updatedAt": timestamp,
	})
	resource := objectSchema([]string{"type", "id"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"entrypoint", "model"}}, "id": text,
	})
	contextMap := JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}
	return map[string]JSONSchema{
		"RoutingContextSource":         source,
		"RoutingContextStoredValue":    stored,
		"RoutingContextEffectiveValue": effective,
		"RoutingContext": objectSchema([]string{"subject", "revision", "schemaRevision", "stored", "effective"}, map[string]JSONSchema{
			"subject": refSchema("PolicySubject"), "revision": revision,
			"schemaRevision": {Type: "integer", Format: "int64", Minimum: intPointer(0)},
			"stored":         arraySchema(stored), "effective": arraySchema(effective),
		}),
		"RoutingContextPutRequest": objectSchema([]string{"values"}, map[string]JSONSchema{"values": contextMap}),
		"AccessCheckResource":      resource,
		"AccessCheckRequest": objectSchema([]string{"subject", "resource", "permission"}, map[string]JSONSchema{
			"subject": refSchema("PolicySubject"), "resource": resource,
			"permission":             {Type: "string", Enum: []string{"discover", "invoke"}},
			"path":                   {Type: "string", Pattern: `^/[^?#]*$`, MaxLength: intPointer(2048)},
			"routingContextOverride": contextMap,
		}),
		"AccessCheckResponse": objectSchema([]string{"subject", "resource", "permission", "decision", "matchedGrants", "routingContext", "simulation", "revision", "appliedRevision"}, map[string]JSONSchema{
			"subject": refSchema("PolicySubject"), "resource": resource,
			"permission":    {Type: "string", Enum: []string{"discover", "invoke"}},
			"decision":      {Type: "string", Enum: []string{"allow", "deny"}},
			"matchedGrants": arraySchema(refSchema("EffectiveGrant")), "routingContext": arraySchema(effective),
			"simulation": {Type: "boolean"}, "revision": revision, "appliedRevision": revision,
		}),
	}
}

func accessReadRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPUT && isRoutingContextPath(contract.Path):
		return "RoutingContextPutRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/access:check":
		return "AccessCheckRequest", true
	default:
		return "", false
	}
}

func accessReadResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && isEffectivePolicyPath(contract.Path):
		return refSchema("EffectivePolicy"), true
	case contract.Method == MethodGET && isQuotaPath(contract.Path):
		return refSchema("EffectiveQuota"), true
	case (contract.Method == MethodGET || contract.Method == MethodPUT) && isRoutingContextPath(contract.Path):
		return refSchema("RoutingContext"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/access:check":
		return refSchema("AccessCheckResponse"), true
	default:
		return JSONSchema{}, false
	}
}

func isEffectivePolicyPath(path string) bool {
	return path == BasePath+"/users/{userId}/effective-policy" ||
		path == BasePath+"/teams/{teamId}/effective-policy" ||
		path == BasePath+"/api-keys/{keyId}/effective-policy"
}

func isRoutingContextPath(path string) bool {
	return path == BasePath+"/users/{userId}/routing-context" ||
		path == BasePath+"/teams/{teamId}/routing-context" ||
		path == BasePath+"/api-keys/{keyId}/routing-context"
}

func isQuotaPath(path string) bool {
	return path == BasePath+"/users/{userId}/quota" || path == BasePath+"/teams/{teamId}/quota" ||
		path == BasePath+"/api-keys/{keyId}/quota"
}
