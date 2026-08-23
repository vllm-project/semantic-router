package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "unknown-usage-fences", Schemas: unknownUsageSchemas,
		ResponseSchema: unknownUsageResponseSchema, RequestSchema: unknownUsageRequestSchema,
		ExtraParameters: unknownUsageParameters,
	})
}

func unknownUsageSchemas() map[string]JSONSchema {
	stringSchema := JSONSchema{Type: "string"}
	timestamp := JSONSchema{Type: "string", Format: "date-time"}
	whole := JSONSchema{Type: "string", Pattern: wholeQuantityPattern.String()}
	cost := objectSchema([]string{"currency", "numerator"}, map[string]JSONSchema{
		"currency":  {Type: "string", Pattern: `^[A-Z]{3}$`},
		"numerator": whole,
	})
	return map[string]JSONSchema{
		"UnknownUsageCost": cost,
		"UnknownUsageCharge": objectSchema([]string{"inputTokens", "outputTokens", "totalTokens", "costs"}, map[string]JSONSchema{
			"inputTokens": whole, "outputTokens": whole, "totalTokens": whole,
			"costs": arraySchema(refSchema("UnknownUsageCost")),
		}),
		"UnknownUsageFenceMeter": objectSchema([]string{"bindingId", "ruleId", "policyId", "subjectKind", "subjectId", "metric", "algorithm", "enforcement"}, map[string]JSONSchema{
			"bindingId": stringSchema, "ruleId": stringSchema, "policyId": stringSchema,
			"subjectKind": {Type: "string", Enum: []string{"api_key", "user", "team"}}, "subjectId": stringSchema,
			"metric": stringSchema, "algorithm": stringSchema, "enforcement": {Type: "string", Enum: []string{"enforce", "shadow"}},
			"admissionLimit": whole, "maximumDebit": whole, "window": stringSchema,
			"calendarPeriod": {Type: "string", Enum: []string{"day", "month"}}, "timezone": stringSchema,
			"currency": {Type: "string", Pattern: `^[A-Z]{3}$`},
		}),
		"UnknownUsageDispatch": objectSchema([]string{"dispatchId"}, map[string]JSONSchema{
			"dispatchId": stringSchema, "modelId": stringSchema, "backendId": stringSchema,
			"providerId": stringSchema, "providerModelId": stringSchema,
			"pricingRevision": {Type: "integer", Format: "int64"},
		}),
		"UnknownUsageEvidence": objectSchema([]string{"dispatchId", "evidenceDigest", "reason"}, map[string]JSONSchema{
			"dispatchId": stringSchema, "evidenceDigest": {Type: "string", Pattern: `^[0-9a-f]{64}$`}, "reason": stringSchema,
		}),
		"UnknownUsageReconciliation": objectSchema([]string{"reconciliationId", "strategy", "createdAt"}, map[string]JSONSchema{
			"reconciliationId": stringSchema, "strategy": {Type: "string", Enum: []string{"actual", "conservative_debit", "waive"}},
			"actorPrincipalId": stringSchema, "reason": stringSchema, "createdAt": timestamp, "appliedAt": timestamp,
		}),
		"UnknownUsageFence": objectSchema([]string{"fenceId", "admissionId", "state", "revision", "reason", "meters", "knownCharge", "createdAt", "updatedAt"}, map[string]JSONSchema{
			"fenceId": stringSchema, "admissionId": stringSchema,
			"state":    {Type: "string", Enum: []string{"open", "reconciling", "resolved"}},
			"revision": {Type: "integer", Format: "int64"}, "reason": stringSchema,
			"meters": arraySchema(refSchema("UnknownUsageFenceMeter")), "knownCharge": refSchema("UnknownUsageCharge"),
			"dispatches": arraySchema(refSchema("UnknownUsageDispatch")), "evidence": arraySchema(refSchema("UnknownUsageEvidence")),
			"reconciliation": refSchema("UnknownUsageReconciliation"), "createdAt": timestamp, "updatedAt": timestamp, "resolvedAt": timestamp,
		}),
		"UnknownUsageFencePage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("UnknownUsageFence")), "page": refSchema("PageInfo"),
		}),
		"UnknownUsageFenceDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("UnknownUsageFence")}),
		"UnknownUsageActualDispatch": objectSchema([]string{"dispatchId", "evidenceDigest", "inputTokens", "cacheReadTokens", "cacheWriteTokens", "outputTokens", "cost"}, map[string]JSONSchema{
			"dispatchId": stringSchema, "evidenceDigest": {Type: "string", Pattern: `^[0-9a-f]{64}$`},
			"inputTokens": whole, "cacheReadTokens": whole, "cacheWriteTokens": whole,
			"outputTokens": whole, "cost": cost,
		}),
		"UnknownUsageActual": objectSchema([]string{"dispatches", "servedInputTokens", "servedOutputTokens"}, map[string]JSONSchema{
			"dispatches":        arraySchema(refSchema("UnknownUsageActualDispatch")),
			"servedInputTokens": whole, "servedOutputTokens": whole,
		}),
		"UnknownUsageReconcileRequest": objectSchema([]string{"strategy", "evidenceReferences", "reason"}, map[string]JSONSchema{
			"strategy": {Type: "string", Enum: []string{"actual", "conservative_debit", "waive"}},
			"actual":   refSchema("UnknownUsageActual"), "evidenceReferences": arraySchema(stringSchema),
			"reason": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(512)},
		}),
	}
}

func unknownUsageResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/unknown-usage-fences":
		return refSchema("UnknownUsageFencePage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/unknown-usage-fences/{fenceId}":
		return refSchema("UnknownUsageFenceDetail"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/unknown-usage-fences/{fenceId}:reconcile":
		return refSchema("Operation"), true
	default:
		return JSONSchema{}, false
	}
}

func unknownUsageRequestSchema(contract OperationContract) (string, bool) {
	if contract.Method == MethodPOST && contract.Path == BasePath+"/unknown-usage-fences/{fenceId}:reconcile" {
		return "UnknownUsageReconcileRequest", true
	}
	return "", false
}

func unknownUsageParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET {
		return nil
	}
	if contract.Path == BasePath+"/unknown-usage-fences" {
		return []OpenAPIParameter{{Name: "state", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"open", "reconciling", "resolved"}}}}
	}
	if contract.Path == BasePath+"/unknown-usage-fences/{fenceId}" {
		return []OpenAPIParameter{
			{Name: "includeInternalDimensions", In: "query", Schema: JSONSchema{Type: "boolean"}},
			{Name: "includeEvidence", In: "query", Schema: JSONSchema{Type: "boolean"}},
			{Name: "includeActor", In: "query", Schema: JSONSchema{Type: "boolean"}},
		}
	}
	return nil
}
