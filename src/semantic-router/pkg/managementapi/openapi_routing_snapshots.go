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
	stringSchema := JSONSchema{Type: "string"}
	timestampSchema := JSONSchema{Type: "string", Format: "date-time"}
	resourceID := JSONSchema{Type: "string", Pattern: `^[a-z][a-z0-9_-]{2,127}$`}
	revision := JSONSchema{Type: "integer", Format: "int64", Minimum: int64Pointer(1)}
	contentDigest := JSONSchema{Type: "string", Pattern: `^sha256:[a-f0-9]{64}$`}
	rawDigest := JSONSchema{Type: "string", Pattern: `^[a-f0-9]{64}$`}
	stringMap := JSONSchema{
		Type: "object", PatternProperties: map[string]JSONSchema{`^.+$`: stringSchema},
		AdditionalProperties: boolPointer(false),
	}
	return map[string]JSONSchema{
		"RoutingSnapshotMetadata": objectSchema([]string{
			"namespaceId", "routingRevision", "contentDigest", "status", "memberCount", "createdAt",
		}, map[string]JSONSchema{
			"namespaceId": stringSchema, "routingRevision": revision, "contentDigest": contentDigest,
			"status":        {Type: "string", Enum: []string{"staged", "active", "failed", "retired"}},
			"failureReason": stringSchema,
			"memberCount":   {Type: "integer", Format: "int64", Minimum: int64Pointer(0)},
			"createdAt":     timestampSchema, "activatedAt": timestampSchema,
		}),
		"RoutingSnapshotMember": objectSchema([]string{
			"resourceType", "resourceId", "resourceRevision",
		}, map[string]JSONSchema{
			"resourceType": {Type: "string", Enum: []string{"model", "recipe", "entrypoint"}},
			"resourceId":   resourceID, "resourceRevision": revision,
		}),
		"RoutingSnapshotBackendConnection": objectSchema([]string{"path"}, map[string]JSONSchema{
			"path": stringSchema, "headers": stringMap,
		}),
		"RoutingSnapshotBackend": objectSchema([]string{
			"id", "providerId", "wireFormat", "origin", "providerModelId", "connection", "weight",
		}, map[string]JSONSchema{
			"id": stringSchema, "providerId": stringSchema, "wireFormat": stringSchema,
			"origin": {Type: "string", Format: "uri"}, "providerModelId": stringSchema,
			"providerCredentialId": stringSchema,
			"connection":           refSchema("RoutingSnapshotBackendConnection"), "weight": stringSchema,
		}),
		"RoutingSnapshotModel": objectSchema([]string{
			"id", "revision", "catalogRevision", "name", "execution", "pricing", "backends",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "catalogRevision": contentDigest, "name": stringSchema,
			"aliases": arraySchema(stringSchema), "paramSize": stringSchema,
			"contextWindowSize": boundedIntegerSchema(0, 100_000_000), "description": stringSchema,
			"capabilities": arraySchema(stringSchema), "reasoning": refSchema("RoutingReasoningFamily"),
			"loras": arraySchema(stringSchema), "qualityScore": {Type: "number"},
			"modality": stringSchema, "tags": arraySchema(stringSchema),
			"execution": refSchema("RoutingExecution"), "pricing": refSchema("RoutingPricing"),
			"backends": arraySchema(refSchema("RoutingSnapshotBackend")),
		}),
		"RoutingSnapshotRecipe": objectSchema([]string{
			"id", "revision", "name", "decisions", "document",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": stringSchema, "description": stringSchema,
			"decisions": arraySchema(refSchema("RoutingDecision")),
			"document":  {Type: "object", AdditionalProperties: boolPointer(true)},
		}),
		"RoutingSnapshotEntrypoint": objectSchema([]string{
			"id", "revision", "name", "aliases", "rules",
		}, map[string]JSONSchema{
			"id": resourceID, "revision": revision, "name": stringSchema,
			"aliases": arraySchema(stringSchema), "rules": arraySchema(refSchema("RoutingEntrypointRuleView")),
		}),
		"RoutingSnapshotExport": objectSchema([]string{
			"namespaceId", "revision", "models", "recipes", "entrypoints", "digest",
		}, map[string]JSONSchema{
			"namespaceId": stringSchema, "revision": revision, "currency": stringSchema,
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
