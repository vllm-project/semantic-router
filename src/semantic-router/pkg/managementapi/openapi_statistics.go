package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "statistics", Schemas: statisticsSchemas, ResponseSchema: statisticsResponseSchema,
	})
}

func statisticsSchemas() map[string]JSONSchema {
	whole := JSONSchema{Type: "string", Pattern: wholeQuantityPattern.String()}
	return map[string]JSONSchema{
		"AccessStatistics": objectSchema([]string{"asOf", "expiringBefore"}, map[string]JSONSchema{
			"asOf": {Type: "string", Format: "date-time"},
			"expiringBefore": {
				Type: "string", Format: "date-time",
				Description: "Inclusive control-plane snapshot window boundary for keys expiring within the next 30 days.",
			},
			"users": whole, "teams": whole,
			"activeApiKeys": whole, "expiringApiKeys": whole,
			"accessPolicies": whole, "activeRatePolicies": whole,
		}),
	}
}

func statisticsResponseSchema(contract OperationContract) (JSONSchema, bool) {
	if contract.Method != MethodGET || contract.Path != BasePath+"/statistics" {
		return JSONSchema{}, false
	}
	return refSchema("AccessStatistics"), true
}
