package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "api-keys", Schemas: apiKeySchemas,
		RequestSchema: apiKeyRequestSchema, ResponseSchema: apiKeyResponseSchema,
		ExtraParameters: apiKeyParameters,
	})
}

func apiKeySchemas() map[string]JSONSchema {
	textSchema := JSONSchema{Type: "string"}
	uuidSchema := JSONSchema{Type: "string", Format: "uuid"}
	dateTimeSchema := JSONSchema{Type: "string", Format: "date-time"}
	owner := objectSchema([]string{"type", "id"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"user", "team"}},
		"id":   uuidSchema,
	})
	key := objectSchema([]string{"keyId", "name", "owner", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"keyId": uuidSchema, "name": textSchema, "owner": owner,
		"contextTeamId": uuidSchema,
		"status":        {Type: "string", Enum: []string{"active", "disabled", "deleted"}},
		"expiresAt":     dateTimeSchema, "lastUsedAt": dateTimeSchema,
		"revision": {Type: "integer", Format: "int64"}, "createdAt": dateTimeSchema,
		"updatedAt": dateTimeSchema, "deletedAt": dateTimeSchema,
	})
	credential := objectSchema([]string{
		"credentialId", "keyId", "kid", "status", "revealable", "notBefore", "createdAt",
	}, map[string]JSONSchema{
		"credentialId": uuidSchema, "keyId": uuidSchema, "kid": textSchema,
		"status":     {Type: "string", Enum: []string{"active", "retiring", "expired", "revoked"}},
		"revealable": {Type: "boolean"}, "notBefore": dateTimeSchema,
		"expiresAt": dateTimeSchema, "revokedAt": dateTimeSchema, "createdAt": dateTimeSchema,
	})
	rateRules := arraySchema(policyRateRuleSchema(false))
	rateRules.MinItems, rateRules.MaxItems = intPointer(1), intPointer(128)
	inlineRatePolicy := objectSchema([]string{"name", "rules"}, map[string]JSONSchema{
		"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
		"description": {Type: "string", MaxLength: intPointer(1000)}, "rules": rateRules,
	})
	rateOverride := JSONSchema{OneOf: []JSONSchema{
		objectSchema([]string{"policyId"}, map[string]JSONSchema{"policyId": uuidSchema}),
		objectSchema([]string{"inlinePolicy"}, map[string]JSONSchema{"inlinePolicy": inlineRatePolicy}),
	}}
	accessPolicyIDs := arraySchema(uuidSchema)
	accessPolicyIDs.MaxItems = intPointer(12)
	policyReceipt := objectSchema([]string{"policyId", "bindingId"}, map[string]JSONSchema{
		"policyId": uuidSchema, "bindingId": uuidSchema,
	})
	rateReceipt := objectSchema([]string{"policyId", "bindingId", "created"}, map[string]JSONSchema{
		"policyId": uuidSchema, "bindingId": uuidSchema, "created": {Type: "boolean"},
	})
	return map[string]JSONSchema{
		"APIKeyOwner":                 owner,
		"APIKeyInlineRateLimitPolicy": inlineRatePolicy,
		"APIKeyRateLimitOverride":     rateOverride,
		"PolicyBindingReceipt":        policyReceipt,
		"RateLimitOverrideReceipt":    rateReceipt,
		"APIKeyCreateRequest": objectSchema([]string{"name", "owner"}, map[string]JSONSchema{
			"name":  {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"owner": owner, "contextTeamId": uuidSchema, "expiresAt": dateTimeSchema,
			"revealable": {Type: "boolean"}, "accessPolicyIds": accessPolicyIDs,
			"rateLimitOverride": rateOverride,
		}),
		"APIKeyPatchRequest": objectSchema([]string{"name"}, map[string]JSONSchema{
			"name": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
		}),
		"APIKeyLifecycleRequest": objectSchema(nil, map[string]JSONSchema{}),
		"APIKeyRenewRequest": objectSchema([]string{"expiresAt"}, map[string]JSONSchema{
			"expiresAt": {OneOf: []JSONSchema{dateTimeSchema, {Type: "null"}}},
		}),
		"APIKeyReassignRequest": objectSchema([]string{"owner"}, map[string]JSONSchema{
			"owner": owner, "contextTeamId": uuidSchema,
		}),
		"APIKeyRotateRequest": objectSchema([]string{"overlapSeconds"}, map[string]JSONSchema{
			"overlapSeconds": boundedIntegerSchema(0, 86400), "revealable": {Type: "boolean"},
		}),
		"APIKey": key,
		"APIKeyPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(key), "page": refSchema("PageInfo"),
		}),
		"APIKeyDetail":     objectSchema([]string{"data"}, map[string]JSONSchema{"data": key}),
		"APIKeyCredential": credential,
		"APIKeyCredentialPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(credential), "page": refSchema("PageInfo"),
		}),
		"APIKeyIssuedSecret": objectSchema([]string{"data", "credential", "secret", "deliveryExpiresAt"}, map[string]JSONSchema{
			"data": key, "credential": credential,
			"secret":               {Type: "string", Format: "password", Description: "One-time inference API key."},
			"accessPolicyBindings": arraySchema(policyReceipt), "rateLimitOverride": rateReceipt,
			"deliveryExpiresAt": dateTimeSchema,
		}),
		"APIKeyRevealResponse": objectSchema([]string{"keyId", "credentialId", "secret"}, map[string]JSONSchema{
			"keyId": uuidSchema, "credentialId": uuidSchema,
			"secret": {Type: "string", Format: "password", Description: "Revealed inference API key."},
		}),
	}
}

func apiKeyRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/api-keys":
		return "APIKeyCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/api-keys/{keyId}":
		return "APIKeyPatchRequest", true
	case contract.Method == MethodPOST && (contract.Path == BasePath+"/api-keys/{keyId}:enable" || contract.Path == BasePath+"/api-keys/{keyId}:disable"):
		return "APIKeyLifecycleRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/api-keys/{keyId}:renew":
		return "APIKeyRenewRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/api-keys/{keyId}:reassign":
		return "APIKeyReassignRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/api-keys/{keyId}/credentials:rotate":
		return "APIKeyRotateRequest", true
	default:
		return "", false
	}
}

func apiKeyResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/api-keys":
		return refSchema("APIKeyPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/api-keys/{keyId}":
		return refSchema("APIKeyDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/api-keys/{keyId}/credentials":
		return refSchema("APIKeyCredentialPage"), true
	case contract.Method == MethodPOST && (contract.Path == BasePath+"/api-keys" || contract.Path == BasePath+"/api-keys/{keyId}/credentials:rotate"):
		return refSchema("APIKeyIssuedSecret"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/api-keys/{keyId}/credentials/{credentialId}:reveal":
		return refSchema("APIKeyRevealResponse"), true
	case stringsHasAPIKeyMutationResponse(contract):
		return refSchema("APIKeyDetail"), true
	default:
		return JSONSchema{}, false
	}
}

func stringsHasAPIKeyMutationResponse(contract OperationContract) bool {
	if contract.Method == MethodPATCH && contract.Path == BasePath+"/api-keys/{keyId}" {
		return true
	}
	if contract.Method != MethodPOST {
		return false
	}
	return contract.Path == BasePath+"/api-keys/{keyId}:enable" ||
		contract.Path == BasePath+"/api-keys/{keyId}:disable" ||
		contract.Path == BasePath+"/api-keys/{keyId}:renew" ||
		contract.Path == BasePath+"/api-keys/{keyId}:reassign"
}

func apiKeyParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET || contract.Path != BasePath+"/api-keys" {
		return nil
	}
	return []OpenAPIParameter{
		{Name: "status", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"active", "disabled", "deleted"}}},
		{Name: "ownerType", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"user", "team"}}},
		{Name: "ownerId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
	}
}
