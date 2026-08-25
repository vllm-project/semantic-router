package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "routing-snapshots", Schemas: routingSnapshotSchemas,
		ResponseSchema: routingSnapshotResponseSchema,
	})
}

func routingSnapshotResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET &&
		contract.Path == BasePath+"/namespaces/{namespaceId}/routing/snapshots":
		return refSchema("RoutingSnapshotPage"), true
	case contract.Method == MethodGET &&
		contract.Path == BasePath+"/namespaces/{namespaceId}/routing/snapshots/{routingRevision}":
		return refSchema("RoutingSnapshotDetail"), true
	default:
		return JSONSchema{}, false
	}
}

func routingSnapshotSchemas() map[string]JSONSchema {
	textSchema := JSONSchema{Type: "string"}
	dateTimeSchema := JSONSchema{Type: "string", Format: "date-time"}
	resourceID := JSONSchema{Type: "string", Pattern: `^[a-z][a-z0-9_-]{2,127}$`}
	revision := JSONSchema{Type: "integer", Format: "int64", Minimum: int64Pointer(1)}
	contentDigest := JSONSchema{Type: "string", Pattern: `^sha256:[a-f0-9]{64}$`}
	rawDigest := JSONSchema{Type: "string", Pattern: `^[a-f0-9]{64}$`}
	stringMap := JSONSchema{
		Type: "object", PatternProperties: map[string]JSONSchema{`^.+$`: textSchema},
		AdditionalProperties: boolPointer(false),
	}
	return map[string]JSONSchema{
		"RoutingSnapshotMetadata": objectSchema([]string{
			"namespaceId", "routingRevision", "contentDigest", "status", "memberCount", "createdAt",
		}, map[string]JSONSchema{
			"namespaceId": textSchema, "routingRevision": revision, "contentDigest": contentDigest,
			"status":        {Type: "string", Enum: []string{"staged", "active", "failed", "retired"}},
			"failureReason": textSchema,
			"memberCount":   {Type: "integer", Format: "int64", Minimum: int64Pointer(0)},
			"createdAt":     dateTimeSchema, "activatedAt": dateTimeSchema,
		}),
		"RoutingSnapshotMember": objectSchema([]string{
			"resourceType", "resourceId", "resourceRevision",
		}, map[string]JSONSchema{
			"resourceType": {Type: "string", Enum: []string{"model", "recipe", "entrypoint"}},
			"resourceId":   resourceID, "resourceRevision": revision,
		}),
		"RoutingSnapshotBackendConnection": objectSchema([]string{"path"}, map[string]JSONSchema{
			"path": textSchema, "headers": stringMap,
		}),
		"RoutingSnapshotBackend": objectSchema([]string{
			"id", "providerId", "wireFormat", "origin", "providerModelId", "connection", "weight",
		}, map[string]JSONSchema{
			"id": textSchema, "providerId": textSchema, "wireFormat": textSchema,
			"origin": {Type: "string", Format: "uri"}, "providerModelId": textSchema,
			"providerCredentialId": textSchema,
			"connection":           refSchema("RoutingSnapshotBackendConnection"), "weight": textSchema,
		}),
		"RoutingSnapshotModel": objectSchema([]string{
			"id", "revision", "catalogRevision", "name", "control", "pricing", "backends",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "catalogRevision": contentDigest, "name": textSchema,
			"aliases": arraySchema(textSchema), "paramSize": textSchema,
			"contextWindowSize": boundedIntegerSchema(0, 100_000_000), "description": textSchema,
			"capabilities": arraySchema(textSchema), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(textSchema), "qualityScore": {Type: "number"},
			"modality": textSchema, "tags": arraySchema(textSchema),
			"control": refSchema("RoutingModelControl"), "pricing": refSchema("RoutingPricing"),
			"backends": arraySchema(refSchema("RoutingSnapshotBackend")),
		}),
		"RoutingSnapshotRecipe": objectSchema([]string{
			"id", "revision", "name", "decisions", "document",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": textSchema, "description": textSchema,
			"decisions": arraySchema(refSchema("RoutingDecision")),
			"document":  {Type: "object", AdditionalProperties: boolPointer(true)},
		}),
		"RoutingSnapshotEntrypoint": objectSchema([]string{
			"id", "revision", "name", "aliases", "rules",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": textSchema,
			"aliases": arraySchema(textSchema), "rules": arraySchema(refSchema("RoutingEntrypointRuleView")),
		}),
		"RoutingSnapshotExport": objectSchema([]string{
			"namespaceId", "revision", "models", "recipes", "entrypoints", "digest",
		}, map[string]JSONSchema{
			"namespaceId": textSchema, "revision": revision, "currency": textSchema,
			"models":      arraySchema(refSchema("RoutingSnapshotModel")),
			"recipes":     arraySchema(refSchema("RoutingSnapshotRecipe")),
			"entrypoints": arraySchema(refSchema("RoutingSnapshotEntrypoint")), "digest": rawDigest,
		}),
		"RoutingSnapshotRecord": objectSchema([]string{"metadata", "members", "export"}, map[string]JSONSchema{
			"metadata": refSchema("RoutingSnapshotMetadata"),
			"members":  arraySchema(refSchema("RoutingSnapshotMember")),
			"export":   refSchema("RoutingSnapshotExport"),
		}),
		"RoutingSnapshotPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RoutingSnapshotMetadata")), "page": refSchema("PageInfo"),
		}),
		"RoutingSnapshotDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": refSchema("RoutingSnapshotRecord"),
		}),
	}
}
