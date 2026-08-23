package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "operations", Schemas: operationSchemas,
		ResponseSchema: operationResponseSchema, ExtraParameters: operationParametersExtension,
	})
}

func operationSchemas() map[string]JSONSchema {
	return map[string]JSONSchema{
		"OperationPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("Operation")), "page": refSchema("PageInfo"),
		}),
	}
}

func operationResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/operations":
		return refSchema("OperationPage"), true
	case (contract.Method == MethodGET && contract.Path == BasePath+"/operations/{operationId}") ||
		(contract.Method == MethodPOST && contract.Path == BasePath+"/operations/{operationId}:cancel"):
		return refSchema("Operation"), true
	default:
		return JSONSchema{}, false
	}
}

func operationParametersExtension(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET || contract.Path != BasePath+"/operations" {
		return nil
	}
	return []OpenAPIParameter{
		{Name: "kind", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{
			"access_policy_bindings.bulk_apply", "rate_limit_bindings.bulk_apply",
		}}},
		{Name: "state", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{
			"pending", "running", "succeeded", "partially_succeeded", "failed", "cancelled",
		}}},
		{Name: "originPrincipalId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
	}
}
