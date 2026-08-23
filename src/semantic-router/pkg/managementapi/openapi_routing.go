package managementapi

import "strings"

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "routing", Schemas: routingSchemas,
		RequestSchema: routingRequestSchema, ResponseSchema: routingResponseSchema,
		ExtraParameters: routingParameters,
	})
}

func routingRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/models":
		return "RoutingModelWrite", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/routing/models/{modelId}":
		return "RoutingModelPatch", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/models:bulk-import":
		return "RoutingBulkImportRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/recipes",
		contract.Method == MethodPATCH && contract.Path == BasePath+"/routing/recipes/{recipeId}":
		return "RoutingRecipeWrite", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/entrypoints",
		contract.Method == MethodPATCH && contract.Path == BasePath+"/routing/entrypoints/{entrypointId}":
		return "RoutingEntrypointWrite", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/entrypoints/{entrypointId}:resolve":
		return "RoutingResolveRequest", true
	default:
		return "", false
	}
}

func routingResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/models":
		return refSchema("RoutingModelPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/model-cards":
		return refSchema("RoutingModelCardPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/models/{modelId}":
		return refSchema("RoutingModelDetail"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/models/{modelId}:probe":
		return refSchema("RoutingProbeResponse"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/recipes":
		return refSchema("RoutingRecipePage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/recipes/{recipeId}":
		return refSchema("RoutingRecipeDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/entrypoints":
		return refSchema("RoutingEntrypointPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/entrypoints/{entrypointId}":
		return refSchema("RoutingEntrypointDetail"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/entrypoints/{entrypointId}:resolve":
		return refSchema("RoutingResolveResponse"), true
	case strings.HasPrefix(contract.Path, BasePath+"/routing/") &&
		(contract.Method == MethodPOST || contract.Method == MethodPATCH) &&
		contract.Path != BasePath+"/routing/models/{modelId}:probe":
		return refSchema("MutationReceipt"), true
	default:
		return JSONSchema{}, false
	}
}

func routingParameters(contract OperationContract) []OpenAPIParameter {
	var parameters []OpenAPIParameter
	if contract.Method == MethodGET && (contract.Path == BasePath+"/routing/models" ||
		contract.Path == BasePath+"/routing/model-cards" ||
		contract.Path == BasePath+"/routing/recipes" ||
		contract.Path == BasePath+"/routing/entrypoints") {
		parameters = append(parameters,
			OpenAPIParameter{Name: "search", In: "query", Schema: JSONSchema{Type: "string"}},
			OpenAPIParameter{Name: "status", In: "query", Schema: JSONSchema{
				Type: "string", Enum: []string{"draft", "active", "disabled"},
			}},
		)
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/routing/entrypoints/{entrypointId}" {
		parameters = append(parameters, OpenAPIParameter{
			Name: "includeTopology", In: "query", Schema: JSONSchema{Type: "boolean"},
			Description: "Include rules only when every exact Recipe and Model dependency is authorized.",
		})
	}
	return parameters
}

func routingSchemas() map[string]JSONSchema {
	stringSchema := JSONSchema{Type: "string"}
	timestampSchema := JSONSchema{Type: "string", Format: "date-time"}
	resourceID := JSONSchema{Type: "string", Pattern: `^[a-z][a-z0-9_-]{2,127}$`}
	revision := JSONSchema{Type: "integer", Format: "int64"}
	digest := JSONSchema{Type: "string", Pattern: `^sha256:[a-f0-9]{64}$`}
	decimal := JSONSchema{Type: "string", Pattern: `^(0|[1-9][0-9]*)(\.[0-9]{1,9})?$`}
	nullableDecimal := JSONSchema{OneOf: []JSONSchema{decimal, {Type: "null"}}}
	openObject := JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}

	return map[string]JSONSchema{
		"RoutingExecution": objectSchema(nil, map[string]JSONSchema{
			"maxRetries": boundedIntegerSchema(0, 5), "requestTimeout": stringSchema, "streamTimeout": stringSchema,
		}),
		"RoutingPricing": objectSchema(nil, map[string]JSONSchema{
			"inputCostPerMillionTokens": nullableDecimal, "outputCostPerMillionTokens": nullableDecimal,
			"cacheReadCostPerMillionTokens": nullableDecimal, "cacheWriteCostPerMillionTokens": nullableDecimal,
		}),
		"RoutingReasoningFamily": objectSchema(nil, map[string]JSONSchema{
			"type": stringSchema, "efforts": arraySchema(stringSchema),
		}),
		"RoutingModelBackendInput": objectSchema([]string{"providerId", "providerModelId"}, map[string]JSONSchema{
			"providerId": stringSchema, "interfaceId": stringSchema,
			"providerModelId": stringSchema, "credentialId": stringSchema,
			"baseUrl": stringSchema, "connectionFields": openObject, "weight": decimal,
		}),
		"RoutingModelWrite": objectSchema([]string{"name", "backends"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "aliases": arraySchema(stringSchema),
			"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
			"description":  stringSchema,
			"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(stringSchema), "execution": refSchema("RoutingExecution"),
			"qualityScore": {Type: "number"},
			"modality":     stringSchema, "tags": arraySchema(stringSchema),
			"pricing": refSchema("RoutingPricing"), "backends": arraySchema(refSchema("RoutingModelBackendInput")),
		}),
		"RoutingModelPatch": objectSchema(nil, map[string]JSONSchema{
			"name": stringSchema, "aliases": arraySchema(stringSchema),
			"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
			"description":  stringSchema,
			"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(stringSchema), "execution": refSchema("RoutingExecution"),
			"qualityScore": {Type: "number"},
			"modality":     stringSchema, "tags": arraySchema(stringSchema),
			"pricing": refSchema("RoutingPricing"), "backends": arraySchema(refSchema("RoutingModelBackendInput")),
		}),
		"RoutingModelBackendView": objectSchema([]string{"providerId", "providerModelId", "credentialConfigured", "weight"}, map[string]JSONSchema{
			"providerId": stringSchema, "providerModelId": stringSchema,
			"credentialConfigured": {Type: "boolean"}, "weight": decimal,
		}),
		"RoutingModelView": objectSchema([]string{
			"id", "name", "status", "revision", "modelRevision", "catalogRevision", "aliases",
			"capabilities", "loras", "tags", "execution", "pricing", "backends", "createdAt", "updatedAt",
		}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema,
			"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
			"revision": revision, "modelRevision": revision, "catalogRevision": digest,
			"aliases": arraySchema(stringSchema), "capabilities": arraySchema(stringSchema),
			"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
			"description": stringSchema,
			"reasoning":   refSchema("RoutingReasoningFamily"), "loras": arraySchema(stringSchema),
			"qualityScore": {Type: "number"},
			"modality":     stringSchema, "tags": arraySchema(stringSchema),
			"execution": refSchema("RoutingExecution"), "pricing": refSchema("RoutingPricing"),
			"backends":  arraySchema(refSchema("RoutingModelBackendView")),
			"createdAt": timestampSchema, "updatedAt": timestampSchema,
		}),
		"RoutingModelPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RoutingModelView")), "page": refSchema("PageInfo"),
		}),
		"RoutingModelCard": objectSchema([]string{"aliases", "capabilities", "loras", "tags"}, map[string]JSONSchema{
			"aliases": arraySchema(stringSchema), "paramSize": stringSchema,
			"contextWindowSize": boundedIntegerSchema(0, 100_000_000), "description": stringSchema,
			"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(stringSchema), "qualityScore": {Type: "number"},
			"modality": stringSchema, "tags": arraySchema(stringSchema),
		}),
		"RoutingModelCardView": objectSchema([]string{"id", "name", "card"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "card": refSchema("RoutingModelCard"),
		}),
		"RoutingModelCardPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RoutingModelCardView")), "page": refSchema("PageInfo"),
		}),
		"RoutingModelDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("RoutingModelView")}),
		"RoutingBulkModelSelection": objectSchema([]string{"catalogItemId", "name"}, map[string]JSONSchema{
			"catalogItemId": stringSchema, "id": resourceID, "name": stringSchema,
			"aliases": arraySchema(stringSchema), "capabilities": arraySchema(stringSchema),
			"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
			"description": stringSchema,
			"reasoning":   refSchema("RoutingReasoningFamily"), "loras": arraySchema(stringSchema),
			"qualityScore": {Type: "number"},
			"modality":     stringSchema, "tags": arraySchema(stringSchema),
			"execution": refSchema("RoutingExecution"), "pricing": refSchema("RoutingPricing"),
		}),
		"RoutingBulkImportRequest": objectSchema([]string{
			"providerId", "catalogRevision", "discoveryClaim", "selections",
		}, map[string]JSONSchema{
			"providerId": stringSchema, "interfaceId": stringSchema,
			"catalogRevision": digest, "discoveryClaim": stringSchema,
			"credentialId": stringSchema, "baseUrl": stringSchema, "connectionFields": openObject,
			"weight": decimal, "selections": arraySchema(refSchema("RoutingBulkModelSelection")),
		}),
		"RoutingDecision": objectSchema([]string{"id", "name", "dispatchCardinality"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema,
			"dispatchCardinality": {Type: "string", Enum: []string{"single", "multi"}},
		}),
		"RoutingRecipeWrite": objectSchema([]string{"name", "document"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "description": stringSchema, "document": openObject,
		}),
		"RoutingRecipeView": objectSchema([]string{
			"id", "name", "status", "revision", "recipeRevision", "origin", "immutable", "decisions", "document", "createdAt", "updatedAt",
		}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "description": stringSchema,
			"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
			"revision": revision, "recipeRevision": revision,
			"origin":    {Type: "string", Enum: []string{"custom", "distribution"}},
			"immutable": {Type: "boolean"}, "provenance": refSchema("RoutingRecipeProvenanceView"),
			"decisions": arraySchema(refSchema("RoutingDecision")), "document": openObject,
			"createdAt": timestampSchema, "updatedAt": timestampSchema,
		}),
		"RoutingRecipeProvenanceView": objectSchema([]string{
			"distributionId", "distributionVersion", "assetDigest", "sourceRecipeId",
			"sourceRevision", "recipeDigest", "installedAt",
		}, map[string]JSONSchema{
			"distributionId": resourceID, "distributionVersion": stringSchema,
			"assetDigest": digest, "sourceRecipeId": resourceID, "sourceRevision": revision,
			"recipeDigest": digest, "installedAt": timestampSchema,
		}),
		"RoutingRecipePage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RoutingRecipeView")), "page": refSchema("PageInfo"),
		}),
		"RoutingRecipeDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("RoutingRecipeView")}),
		"RoutingClaimValue": {OneOf: []JSONSchema{
			objectSchema([]string{"kind", "string"}, map[string]JSONSchema{
				"kind": {Type: "string", Enum: []string{"string"}}, "string": {Type: "string", MaxLength: intPointer(4096)},
			}),
			objectSchema([]string{"kind", "boolean"}, map[string]JSONSchema{
				"kind": {Type: "string", Enum: []string{"boolean"}}, "boolean": {Type: "boolean"},
			}),
			objectSchema([]string{"kind", "integer"}, map[string]JSONSchema{
				"kind": {Type: "string", Enum: []string{"integer"}}, "integer": {Type: "integer", Format: "int64"},
			}),
		}},
		"RoutingClaimMatcher": objectSchema([]string{"name", "value"}, map[string]JSONSchema{
			"name": stringSchema, "value": refSchema("RoutingClaimValue"),
		}),
		"RoutingMatcher": {OneOf: []JSONSchema{
			objectSchema([]string{"claim"}, map[string]JSONSchema{"claim": refSchema("RoutingClaimMatcher")}),
			objectSchema([]string{"exactPath"}, map[string]JSONSchema{"exactPath": stringSchema}),
			objectSchema([]string{"pathPrefix"}, map[string]JSONSchema{"pathPrefix": stringSchema}),
		}},
		"RoutingAssignmentReasoning": objectSchema([]string{"enabled"}, map[string]JSONSchema{
			"enabled": {Type: "boolean"}, "effort": stringSchema, "description": stringSchema,
		}),
		"RoutingAssignmentWrite": objectSchema([]string{"modelId"}, map[string]JSONSchema{
			"modelId": resourceID, "priority": boundedIntegerSchema(0, 31), "weight": decimal, "loraName": stringSchema,
			"reasoning": refSchema("RoutingAssignmentReasoning"),
		}),
		"RoutingFallbackPolicy": objectSchema([]string{"strategy", "on"}, map[string]JSONSchema{
			"strategy": {Type: "string", Enum: []string{"priority"}},
			"on":       {Type: "array", Items: schemaPointer(JSONSchema{Type: "string", Enum: []string{"unavailable", "overloaded", "timeout"}}), MinItems: intPointer(1), MaxItems: intPointer(3)},
		}),
		"RoutingAssignmentSetWrite": objectSchema([]string{"models"}, map[string]JSONSchema{
			"models":   {Type: "array", Items: schemaPointer(refSchema("RoutingAssignmentWrite")), MinItems: intPointer(1), MaxItems: intPointer(32)},
			"fallback": refSchema("RoutingFallbackPolicy"),
		}),
		"RoutingEntrypointRuleWrite": objectSchema([]string{"name", "recipeId", "assignments"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "matchers": arraySchema(refSchema("RoutingMatcher")),
			"recipeId": resourceID, "assignments": {
				Type: "object", PatternProperties: map[string]JSONSchema{`^[a-z][a-z0-9_-]{2,127}$`: refSchema("RoutingAssignmentSetWrite")},
				AdditionalProperties: boolPointer(false),
			},
		}),
		"RoutingEntrypointWrite": objectSchema([]string{"name", "aliases", "rules"}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "aliases": arraySchema(stringSchema),
			"rules": arraySchema(refSchema("RoutingEntrypointRuleWrite")),
		}),
		"RoutingAssignmentView": objectSchema([]string{"modelId", "modelRevision", "priority", "weight"}, map[string]JSONSchema{
			"modelId": resourceID, "modelRevision": revision, "priority": boundedIntegerSchema(0, 31), "weight": decimal, "loraName": stringSchema,
			"reasoning": refSchema("RoutingAssignmentReasoning"),
		}),
		"RoutingAssignmentSetView": objectSchema([]string{"models"}, map[string]JSONSchema{
			"models":   {Type: "array", Items: schemaPointer(refSchema("RoutingAssignmentView")), MinItems: intPointer(1), MaxItems: intPointer(32)},
			"fallback": refSchema("RoutingFallbackPolicy"),
		}),
		"RoutingEntrypointRuleView": objectSchema([]string{
			"id", "name", "recipeId", "recipeRevision", "assignments",
		}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema, "matchers": arraySchema(refSchema("RoutingMatcher")),
			"recipeId": resourceID, "recipeRevision": revision, "assignments": {
				Type: "object", PatternProperties: map[string]JSONSchema{`^[a-z][a-z0-9_-]{2,127}$`: refSchema("RoutingAssignmentSetView")},
				AdditionalProperties: boolPointer(false),
			},
		}),
		"RoutingEntrypointView": objectSchema([]string{
			"id", "name", "status", "revision", "entrypointRevision", "aliases", "ruleCount", "assignedModelCount", "createdAt", "updatedAt",
		}, map[string]JSONSchema{
			"id": resourceID, "name": stringSchema,
			"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
			"revision": revision, "entrypointRevision": revision, "aliases": arraySchema(stringSchema),
			"ruleCount": boundedIntegerSchema(0, 64), "assignedModelCount": boundedIntegerSchema(0, 131072),
			"rules":     arraySchema(refSchema("RoutingEntrypointRuleView")),
			"createdAt": timestampSchema, "updatedAt": timestampSchema,
		}),
		"RoutingEntrypointPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RoutingEntrypointView")), "page": refSchema("PageInfo"),
		}),
		"RoutingEntrypointDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": refSchema("RoutingEntrypointView"),
		}),
		"RoutingResolveRequest": objectSchema(nil, map[string]JSONSchema{
			"path": stringSchema, "claims": openObject,
		}),
		"RoutingResolvedEntrypoint": objectSchema([]string{"id", "revision", "name", "aliases"}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": stringSchema, "aliases": arraySchema(stringSchema),
		}),
		"RoutingResolvedRecipe": objectSchema([]string{"id", "revision", "name", "decisions", "document"}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": stringSchema,
			"decisions": arraySchema(refSchema("RoutingDecision")), "document": openObject,
		}),
		"RoutingResolveResponse": objectSchema([]string{"outcome"}, map[string]JSONSchema{
			"outcome":    {Type: "string", Enum: []string{"matched", "claimed_no_match", "unclaimed"}},
			"entrypoint": refSchema("RoutingResolvedEntrypoint"), "rule": refSchema("RoutingEntrypointRuleView"),
			"recipe": refSchema("RoutingResolvedRecipe"),
		}),
		"RoutingProbeResponse": objectSchema([]string{"reachable", "latencyMilliseconds", "checkedAt"}, map[string]JSONSchema{
			"reachable": {Type: "boolean"}, "latencyMilliseconds": {Type: "integer", Format: "int64"},
			"checkedAt": timestampSchema,
		}),
	}
}
