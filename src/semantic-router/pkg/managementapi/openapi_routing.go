package managementapi

import "strings"

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "routing", Schemas: routingSchemas,
		RequestSchema: routingRequestSchema, ResponseSchema: routingResponseSchema,
		ExtraParameters: routingParameters, AmendResponses: amendRoutingResponses,
	})
}

func amendRoutingResponses(contract OperationContract, responses map[string]OpenAPIResponse) {
	if contract.Method == MethodPOST && contract.Path == BasePath+"/routing/imports" {
		responses["200"] = OpenAPIResponse{
			Description: "Validated dry-run diff; no durable state was changed.",
			Content: map[string]OpenAPIMedia{
				JSONMediaType: {Schema: refSchema("RoutingManifestImportResult")},
			},
		}
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/routing/exports/current" {
		response := responses["200"]
		response.Content = map[string]OpenAPIMedia{
			YAMLMediaType: {Schema: JSONSchema{Type: "string", Description: "Portable strict v0.3 routing manifest."}},
		}
		responses["200"] = response
	}
}

func routingRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/imports":
		return "RoutingManifestImportRequest", true
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
	case contract.Method == MethodPOST && contract.Path == BasePath+"/routing/imports":
		return refSchema("RoutingManifestImportResult"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/routing/exports/current":
		return JSONSchema{Type: "string", Description: "Portable strict v0.3 routing manifest."}, true
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

var (
	routingResourceID         = JSONSchema{Type: "string", Pattern: `^[a-z][a-z0-9_-]{2,127}$`}
	routingRevision           = JSONSchema{Type: "integer", Format: "int64"}
	routingDigest             = JSONSchema{Type: "string", Pattern: `^sha256:[a-f0-9]{64}$`}
	routingModelRetryEvidence = []string{"unavailable", "timeout"}
	routingModelDuration      = JSONSchema{
		Type: "string", Pattern: routingModelDurationPattern,
		Description: "Go-style duration between 1s and 24h, such as 30s, 2m, or 1h30m.",
	}
	routingModelPrice = JSONSchema{
		Type: "string", Pattern: routingModelPricePattern,
		Description: "Exact non-negative price per million tokens, with at most 9 fractional digits and a maximum of 1000000.",
	}
	routingNullableModelPrice = JSONSchema{OneOf: []JSONSchema{routingModelPrice, {Type: "null"}}}
	routingModelQuality       = JSONSchema{Type: "number", Minimum: intPointer(0), Maximum: intPointer(1)}
)

const routingModelDurationPattern = `^\+?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ns|us|µs|μs|ms|s|m|h))+$`
const routingModelPricePattern = `^(?:(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,9})?|1000000(?:\.0{1,9})?)$`

var routingSchemaCatalog = map[string]JSONSchema{
	"RoutingManifestImportRequest": objectSchema([]string{"manifest"}, map[string]JSONSchema{
		"manifest": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(3 << 20)},
		"dryRun":   {Type: "boolean"},
	}),
	"RoutingManifestResourceDiff": objectSchema([]string{"create", "update", "disable"}, map[string]JSONSchema{
		"create": arraySchema(stringSchema), "update": arraySchema(stringSchema), "disable": arraySchema(stringSchema),
	}),
	"RoutingManifestDiff": objectSchema([]string{"models", "recipes", "entrypoints"}, map[string]JSONSchema{
		"models": refSchema("RoutingManifestResourceDiff"), "recipes": refSchema("RoutingManifestResourceDiff"),
		"entrypoints": refSchema("RoutingManifestResourceDiff"),
	}),
	"RoutingManifestImportResult": objectSchema([]string{"diff", "replayed"}, map[string]JSONSchema{
		"diff": refSchema("RoutingManifestDiff"), "operationId": {Type: "string", Format: "uuid"},
		"desiredRevision": routingRevision, "replayed": {Type: "boolean"},
	}),
	"RoutingModelRetryControl": objectSchema(nil, map[string]JSONSchema{
		"count": boundedIntegerSchema(0, 5),
		"on": {
			Type: "array", Items: schemaPointer(JSONSchema{
				Type: "string", Enum: append([]string(nil), routingModelRetryEvidence...),
			}),
			MaxItems: intPointer(int64(len(routingModelRetryEvidence))), UniqueItems: true,
		},
	}),
	"RoutingModelTimeoutControl": objectSchema(nil, map[string]JSONSchema{
		"request": routingModelDuration,
		"stream":  routingModelDuration,
	}),
	"RoutingModelControl": objectSchema(nil, map[string]JSONSchema{
		"retry":   refSchema("RoutingModelRetryControl"),
		"timeout": refSchema("RoutingModelTimeoutControl"),
	}),
	"RoutingPricing": objectSchema(nil, map[string]JSONSchema{
		"inputCostPerMillionTokens": routingNullableModelPrice, "outputCostPerMillionTokens": routingNullableModelPrice,
		"cacheReadCostPerMillionTokens": routingNullableModelPrice, "cacheWriteCostPerMillionTokens": routingNullableModelPrice,
	}),
	"RoutingReasoningFamily": objectSchema(nil, map[string]JSONSchema{
		"type": stringSchema, "efforts": arraySchema(stringSchema),
	}),
	"RoutingModelBackendInput": objectSchema([]string{"providerId", "providerModelId"}, map[string]JSONSchema{
		"providerId": stringSchema, "interfaceId": stringSchema,
		"providerModelId": stringSchema, "credentialId": stringSchema,
		"baseUrl": stringSchema, "connectionFields": openObjectSchema, "weight": decimal,
	}),
	"RoutingModelWrite": objectSchema([]string{"name", "backends"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "aliases": arraySchema(stringSchema),
		"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
		"description":  stringSchema,
		"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
		"loras": arraySchema(stringSchema), "control": refSchema("RoutingModelControl"),
		"qualityScore": routingModelQuality,
		"modality":     stringSchema, "tags": arraySchema(stringSchema),
		"pricing": refSchema("RoutingPricing"), "backends": arraySchema(refSchema("RoutingModelBackendInput")),
	}),
	"RoutingModelPatch": objectSchema(nil, map[string]JSONSchema{
		"name": stringSchema, "aliases": arraySchema(stringSchema),
		"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
		"description":  stringSchema,
		"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
		"loras": arraySchema(stringSchema), "control": refSchema("RoutingModelControl"),
		"qualityScore": routingModelQuality,
		"modality":     stringSchema, "tags": arraySchema(stringSchema),
		"pricing": refSchema("RoutingPricing"), "backends": arraySchema(refSchema("RoutingModelBackendInput")),
	}),
	"RoutingModelBackendView": objectSchema([]string{"providerId", "providerModelId", "credentialConfigured", "weight"}, map[string]JSONSchema{
		"providerId": stringSchema, "providerModelId": stringSchema,
		"credentialConfigured": {Type: "boolean"}, "weight": decimal,
	}),
	"RoutingModelView": objectSchema([]string{
		"id", "name", "status", "revision", "modelRevision", "catalogRevision", "aliases",
		"capabilities", "loras", "tags", "control", "pricing", "backends", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema,
		"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
		"revision": routingRevision, "modelRevision": routingRevision, "catalogRevision": routingDigest,
		"aliases": arraySchema(stringSchema), "capabilities": arraySchema(stringSchema),
		"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
		"description": stringSchema,
		"reasoning":   refSchema("RoutingReasoningFamily"), "loras": arraySchema(stringSchema),
		"qualityScore": routingModelQuality,
		"modality":     stringSchema, "tags": arraySchema(stringSchema),
		"control": refSchema("RoutingModelControl"), "pricing": refSchema("RoutingPricing"),
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
		"loras": arraySchema(stringSchema), "qualityScore": routingModelQuality,
		"modality": stringSchema, "tags": arraySchema(stringSchema),
	}),
	"RoutingModelCardView": objectSchema([]string{"id", "name", "card"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "card": refSchema("RoutingModelCard"),
	}),
	"RoutingModelCardPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
		"data": arraySchema(refSchema("RoutingModelCardView")), "page": refSchema("PageInfo"),
	}),
	"RoutingModelDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("RoutingModelView")}),
	"RoutingBulkModelSelection": objectSchema([]string{"catalogItemId", "name"}, map[string]JSONSchema{
		"catalogItemId": stringSchema, "id": routingResourceID, "name": stringSchema,
		"aliases": arraySchema(stringSchema), "capabilities": arraySchema(stringSchema),
		"paramSize": stringSchema, "contextWindowSize": boundedIntegerSchema(0, 100_000_000),
		"description": stringSchema,
		"reasoning":   refSchema("RoutingReasoningFamily"), "loras": arraySchema(stringSchema),
		"qualityScore": routingModelQuality,
		"modality":     stringSchema, "tags": arraySchema(stringSchema),
		"control": refSchema("RoutingModelControl"), "pricing": refSchema("RoutingPricing"),
	}),
	"RoutingBulkImportRequest": objectSchema([]string{
		"providerId", "catalogRevision", "discoveryClaim", "selections",
	}, map[string]JSONSchema{
		"providerId": stringSchema, "interfaceId": stringSchema,
		"catalogRevision": routingDigest, "discoveryClaim": stringSchema,
		"credentialId": stringSchema, "baseUrl": stringSchema, "connectionFields": openObjectSchema,
		"weight": decimal, "selections": arraySchema(refSchema("RoutingBulkModelSelection")),
	}),
	"RoutingDecision": objectSchema([]string{"id", "name", "dispatchCardinality"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema,
		"dispatchCardinality": {Type: "string", Enum: []string{"single", "multi"}},
	}),
	"RoutingRecipeWrite": objectSchema([]string{"name", "document"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "description": stringSchema, "document": openObjectSchema,
	}),
	"RoutingRecipeView": objectSchema([]string{
		"id", "name", "status", "revision", "recipeRevision", "origin", "immutable", "decisions", "document", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "description": stringSchema,
		"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
		"revision": routingRevision, "recipeRevision": routingRevision,
		"origin":    {Type: "string", Enum: []string{"custom", "distribution"}},
		"immutable": {Type: "boolean"}, "provenance": refSchema("RoutingRecipeProvenanceView"),
		"decisions": arraySchema(refSchema("RoutingDecision")), "document": openObjectSchema,
		"createdAt": timestampSchema, "updatedAt": timestampSchema,
	}),
	"RoutingRecipeProvenanceView": objectSchema([]string{
		"distributionId", "distributionVersion", "assetDigest", "sourceRecipeId",
		"sourceRevision", "recipeDigest", "installedAt",
	}, map[string]JSONSchema{
		"distributionId": routingResourceID, "distributionVersion": stringSchema,
		"assetDigest": routingDigest, "sourceRecipeId": routingResourceID, "sourceRevision": routingRevision,
		"recipeDigest": routingDigest, "installedAt": timestampSchema,
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
		"modelId": routingResourceID, "priority": boundedIntegerSchema(0, 31), "weight": decimal, "loraName": stringSchema,
		"reasoning": refSchema("RoutingAssignmentReasoning"),
	}),
	"RoutingFallbackPolicy": objectSchema([]string{"strategy", "on"}, map[string]JSONSchema{
		"strategy": {Type: "string", Enum: []string{"priority"}},
		"on":       {Type: "array", Items: schemaPointer(JSONSchema{Type: "string", Enum: []string{"unavailable", "timeout"}}), MinItems: intPointer(1), MaxItems: intPointer(2)},
	}),
	"RoutingAssignmentSetWrite": objectSchema([]string{"models"}, map[string]JSONSchema{
		"models":   {Type: "array", Items: schemaPointer(refSchema("RoutingAssignmentWrite")), MinItems: intPointer(1), MaxItems: intPointer(32)},
		"fallback": refSchema("RoutingFallbackPolicy"),
	}),
	"RoutingEntrypointRuleWrite": objectSchema([]string{"name", "recipeId", "assignments"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "matchers": arraySchema(refSchema("RoutingMatcher")),
		"recipeId": routingResourceID, "assignments": {
			Type: "object", PatternProperties: map[string]JSONSchema{`^[a-z][a-z0-9_-]{2,127}$`: refSchema("RoutingAssignmentSetWrite")},
			AdditionalProperties: boolPointer(false),
		},
	}),
	"RoutingEntrypointWrite": objectSchema([]string{"name", "aliases", "rules"}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "aliases": arraySchema(stringSchema),
		"rules": arraySchema(refSchema("RoutingEntrypointRuleWrite")),
	}),
	"RoutingAssignmentView": objectSchema([]string{"modelId", "modelRevision", "priority", "weight"}, map[string]JSONSchema{
		"modelId": routingResourceID, "modelRevision": routingRevision, "priority": boundedIntegerSchema(0, 31), "weight": decimal, "loraName": stringSchema,
		"reasoning": refSchema("RoutingAssignmentReasoning"),
	}),
	"RoutingAssignmentSetView": objectSchema([]string{"models"}, map[string]JSONSchema{
		"models":   {Type: "array", Items: schemaPointer(refSchema("RoutingAssignmentView")), MinItems: intPointer(1), MaxItems: intPointer(32)},
		"fallback": refSchema("RoutingFallbackPolicy"),
	}),
	"RoutingEntrypointRuleView": objectSchema([]string{
		"id", "name", "recipeId", "recipeRevision", "assignments",
	}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema, "matchers": arraySchema(refSchema("RoutingMatcher")),
		"recipeId": routingResourceID, "recipeRevision": routingRevision, "assignments": {
			Type: "object", PatternProperties: map[string]JSONSchema{`^[a-z][a-z0-9_-]{2,127}$`: refSchema("RoutingAssignmentSetView")},
			AdditionalProperties: boolPointer(false),
		},
	}),
	"RoutingEntrypointView": objectSchema([]string{
		"id", "name", "status", "revision", "entrypointRevision", "aliases", "ruleCount", "assignedModelCount", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"id": routingResourceID, "name": stringSchema,
		"status":   {Type: "string", Enum: []string{"draft", "active", "disabled"}},
		"revision": routingRevision, "entrypointRevision": routingRevision, "aliases": arraySchema(stringSchema),
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
		"path": stringSchema, "claims": openObjectSchema,
	}),
	"RoutingResolvedEntrypoint": objectSchema([]string{"id", "revision", "name", "aliases"}, map[string]JSONSchema{
		"id": routingResourceID, "revision": routingRevision, "name": stringSchema, "aliases": arraySchema(stringSchema),
	}),
	"RoutingResolvedRecipe": objectSchema([]string{"id", "revision", "name", "decisions", "document"}, map[string]JSONSchema{
		"id": routingResourceID, "revision": routingRevision, "name": stringSchema,
		"decisions": arraySchema(refSchema("RoutingDecision")), "document": openObjectSchema,
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

func routingSchemas() map[string]JSONSchema {
	return cloneSchemas(routingSchemaCatalog)
}
