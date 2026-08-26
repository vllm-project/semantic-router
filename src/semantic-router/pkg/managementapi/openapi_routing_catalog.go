package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "routing-catalog", Schemas: routingCatalogSchemas,
		ResponseSchema: routingCatalogResponseSchema,
	})
}

func routingCatalogResponseSchema(contract OperationContract) (JSONSchema, bool) {
	if contract.Method == MethodGET && contract.Path == BasePath+"/api-keys/{keyId}/routing-catalog" {
		return refSchema("RoutingCatalog"), true
	}
	return JSONSchema{}, false
}

func routingCatalogSchemas() map[string]JSONSchema {
	text := JSONSchema{Type: "string"}
	resourceID := JSONSchema{Type: "string", Pattern: `^[a-z][a-z0-9_-]{2,127}$`}
	revision := JSONSchema{Type: "integer", Format: "int64", Minimum: intPointer(1)}
	digest := JSONSchema{Type: "string", Pattern: `^[a-f0-9]{64}$`}
	assignments := JSONSchema{
		Type: "object",
		PatternProperties: map[string]JSONSchema{
			`^[a-z][a-z0-9_-]{2,127}$`: refSchema("RoutingCatalogAssignmentSet"),
		},
		AdditionalProperties: boolPointer(false),
	}
	return map[string]JSONSchema{
		"RoutingCatalogSignal": objectSchema([]string{"type", "name"}, map[string]JSONSchema{
			"type": text, "name": text,
		}),
		"RoutingCatalogProjectionReference": objectSchema([]string{"type"}, map[string]JSONSchema{
			"type": text, "name": text, "kb": text, "metric": text,
		}),
		"RoutingCatalogProjection": objectSchema([]string{"type", "name", "members", "inputs", "outputs"}, map[string]JSONSchema{
			"type": {Type: "string", Enum: []string{"partition", "score", "mapping"}},
			"name": text, "members": arraySchema(text),
			"inputs": arraySchema(refSchema("RoutingCatalogProjectionReference")),
			"source": text, "outputs": arraySchema(text),
		}),
		"RoutingCatalogModel": objectSchema([]string{
			"id", "revision", "name", "aliases", "capabilities", "loras", "tags", "pricing",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": text,
			"aliases": arraySchema(text), "paramSize": text,
			"contextWindowSize": boundedIntegerSchema(0, 100_000_000), "description": text,
			"capabilities": arraySchema(text), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(text), "qualityScore": {Type: "number"},
			"modality": text, "tags": arraySchema(text), "pricing": refSchema("RoutingPricing"),
		}),
		"RoutingCatalogRecipe": objectSchema([]string{"id", "revision", "name", "decisions", "signals", "projections"}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": text, "description": text,
			"decisions":   arraySchema(refSchema("RoutingDecision")),
			"signals":     arraySchema(refSchema("RoutingCatalogSignal")),
			"projections": arraySchema(refSchema("RoutingCatalogProjection")),
		}),
		"RoutingCatalogAssignmentSet": objectSchema([]string{"models"}, map[string]JSONSchema{
			"models":   {Type: "array", Items: schemaPointer(refSchema("RoutingAssignmentView")), MaxItems: intPointer(32)},
			"fallback": refSchema("RoutingFallbackPolicy"),
		}),
		"RoutingCatalogRule": objectSchema([]string{
			"id", "name", "recipeId", "recipeRevision", "assignments",
		}, map[string]JSONSchema{
			"id": resourceID, "name": text, "matchers": arraySchema(refSchema("RoutingMatcher")),
			"recipeId": resourceID, "recipeRevision": revision, "assignments": assignments,
		}),
		"RoutingCatalogEntrypoint": objectSchema([]string{"id", "revision", "name", "aliases", "rules"}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": text,
			"aliases": arraySchema(text), "rules": arraySchema(refSchema("RoutingCatalogRule")),
		}),
		"RoutingCatalog": objectSchema([]string{
			"keyId", "policyRevision", "policyDigest", "routingRevision", "routingDigest",
			"models", "recipes", "entrypoints",
		}, map[string]JSONSchema{
			"keyId": {Type: "string", Format: "uuid"}, "policyRevision": revision,
			"policyDigest": digest, "routingRevision": revision, "routingDigest": digest,
			"models":      arraySchema(refSchema("RoutingCatalogModel")),
			"recipes":     arraySchema(refSchema("RoutingCatalogRecipe")),
			"entrypoints": arraySchema(refSchema("RoutingCatalogEntrypoint")),
		}),
	}
}
