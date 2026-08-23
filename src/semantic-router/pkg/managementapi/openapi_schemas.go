package managementapi

func canonicalSchemas() map[string]JSONSchema {
	stringSchema := JSONSchema{Type: "string"}
	timestampSchema := JSONSchema{Type: "string", Format: "date-time"}
	whole := JSONSchema{Type: "string", Pattern: wholeQuantityPattern.String()}
	decimal := JSONSchema{Type: "string", Pattern: decimalQuantityPattern.String()}
	nullableDecimal := JSONSchema{OneOf: []JSONSchema{decimal, {Type: "null"}}}
	currencyDecimal := JSONSchema{Type: "string", Pattern: currencyDecimalPattern.String()}
	object := JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}

	schemas := map[string]JSONSchema{
		"ErrorResponse": objectSchema([]string{"error"}, map[string]JSONSchema{"error": refSchema("APIError")}),
		"APIError": objectSchema([]string{"code", "message"}, map[string]JSONSchema{
			"code": stringSchema, "message": stringSchema, "requestId": stringSchema,
			"details": arraySchema(refSchema("ErrorDetail")), "stepUp": refSchema("StepUpChallenge"),
		}),
		"ErrorDetail": objectSchema([]string{"reason"}, map[string]JSONSchema{"field": stringSchema, "reason": stringSchema}),
		"StepUpChallenge": objectSchema([]string{"challengeId", "expiresAt", "methods"}, map[string]JSONSchema{
			"challengeId": stringSchema, "expiresAt": timestampSchema, "methods": arraySchema(stringSchema),
		}),
		"Page": objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(object), "page": refSchema("PageInfo")}),
		"PageInfo": objectSchema([]string{"hasMore", "pageSize"}, map[string]JSONSchema{
			"nextCursor": stringSchema, "hasMore": {Type: "boolean"}, "pageSize": boundedIntegerSchema(1, 200),
		}),
		"RevisionState": objectSchema([]string{"desiredRevision", "appliedRevision"}, map[string]JSONSchema{
			"desiredRevision": {Type: "integer", Format: "int64"}, "stagedRevision": {Type: "integer", Format: "int64"},
			"publicationRevision": {Type: "integer", Format: "int64"}, "appliedRevision": {Type: "integer", Format: "int64"},
		}),
		"Operation": objectSchema([]string{"operationId", "kind", "state", "progress", "revisions", "createdAt", "updatedAt"}, map[string]JSONSchema{
			"operationId": stringSchema, "kind": stringSchema,
			"state":    {Type: "string", Enum: []string{string(OperationPending), string(OperationRunning), string(OperationSucceeded), string(OperationPartiallySucceeded), string(OperationFailed), string(OperationCancelled)}},
			"progress": refSchema("OperationProgress"), "targetIds": arraySchema(stringSchema), "revisions": refSchema("RevisionState"),
			"itemErrors": arraySchema(refSchema("OperationItemFailure")), "createdAt": timestampSchema, "updatedAt": timestampSchema, "completedAt": timestampSchema,
		}),
		"OperationProgress":    objectSchema([]string{"total", "completed", "failed"}, map[string]JSONSchema{"total": whole, "completed": whole, "failed": whole}),
		"OperationItemFailure": objectSchema([]string{"code", "reason"}, map[string]JSONSchema{"itemId": stringSchema, "code": stringSchema, "reason": stringSchema}),
		"SecretEnvelope": objectSchema([]string{"resourceId", "kind", "secret"}, map[string]JSONSchema{
			"resourceId": stringSchema, "kind": stringSchema, "secret": {Type: "string", Format: "password", Description: "One-time secret; never logged or cached."}, "expiresAt": timestampSchema,
		}),
		"ManagementTokenEnvelope": objectSchema([]string{"accessToken", "tokenType", "expiresIn", "managementSessionId"}, map[string]JSONSchema{
			"accessToken": {Type: "string", Format: "password"}, "tokenType": {Type: "string", Enum: []string{"Bearer"}},
			"expiresIn": {Type: "integer", Format: "int64"}, "managementSessionId": stringSchema,
		}),
		"CostSummary": objectSchema([]string{"currency", "knownAmount", "completeness", "knownDispatches", "incompleteDispatches"}, map[string]JSONSchema{
			"currency": {Type: "string", Pattern: `^[A-Z]{3}$`}, "knownAmount": currencyDecimal,
			"completeness":    {Type: "string", Enum: []string{string(CostComplete), string(CostPartial), string(CostUnknown)}},
			"knownDispatches": whole, "incompleteDispatches": whole,
		}),
		"QuotaMeter": objectSchema([]string{"policyId", "ruleId", "bindingId", "source", "counterOwner", "metric", "algorithm", "accounting", "enforcement", "limit", "used", "remaining", "completeness", "knownDispatches", "incompleteDispatches", "capacityState", "activeFenceIds", "freshness"}, map[string]JSONSchema{
			"policyId": stringSchema, "ruleId": stringSchema, "bindingId": stringSchema, "source": refSchema("GrantSource"), "counterOwner": stringSchema,
			"metric": stringSchema, "algorithm": stringSchema, "accounting": stringSchema, "enforcement": stringSchema,
			"window": stringSchema, "currency": {Type: "string", Pattern: `^[A-Z]{3}$`}, "limit": decimal, "used": decimal, "remaining": nullableDecimal, "overage": decimal,
			"resetAt": timestampSchema, "completeness": {Type: "string", Enum: []string{"complete", "partial", "unknown"}},
			"knownDispatches": whole, "incompleteDispatches": whole,
			"capacityState":  {Type: "string", Enum: []string{"available", "exhausted", "over_limit", "fenced", "unknown"}},
			"activeFenceIds": arraySchema(stringSchema), "freshness": refSchema("MeterFreshness"),
		}),
		"GrantSource": objectSchema([]string{"subjectType", "subjectId", "bindingId"}, map[string]JSONSchema{"subjectType": stringSchema, "subjectId": stringSchema, "bindingId": stringSchema}),
		"EffectiveGrant": objectSchema([]string{"resourceType", "resourceId", "permissions", "effect", "source"}, map[string]JSONSchema{
			"resourceType": {Type: "string", Enum: []string{"entrypoint", "model"}}, "resourceId": stringSchema,
			"permissions": arraySchema(JSONSchema{Type: "string", Enum: []string{"discover", "invoke"}}),
			"effect":      {Type: "string", Enum: []string{"allow", "deny"}}, "source": refSchema("GrantSource"),
		}),
		"EffectiveAccess": objectSchema([]string{"grants"}, map[string]JSONSchema{
			"grants": arraySchema(refSchema("EffectiveGrant")),
		}),
		"MeterFreshness": objectSchema([]string{"source", "asOf"}, map[string]JSONSchema{"source": stringSchema, "asOf": timestampSchema}),
		"EffectiveQuota": objectSchema([]string{"meters", "unknownUsageFences", "asOf"}, map[string]JSONSchema{
			"meters": arraySchema(refSchema("QuotaMeter")), "limitingRuleId": stringSchema, "unknownUsageFences": arraySchema(stringSchema), "asOf": timestampSchema,
		}),
		"EffectivePolicy": objectSchema([]string{"subject", "revision", "appliedRevision", "access", "quota"}, map[string]JSONSchema{
			"subject": refSchema("PolicySubject"), "revision": {Type: "integer", Format: "int64"}, "appliedRevision": {Type: "integer", Format: "int64"},
			"access": refSchema("EffectiveAccess"), "quota": refSchema("EffectiveQuota"),
		}),
		"ProviderCatalogIcon": objectSchema([]string{"source", "value", "color"}, map[string]JSONSchema{
			"source": {Type: "string", Enum: []string{"lobe", "asset", "url"}},
			"value":  stringSchema, "color": {Type: "boolean"},
		}),
		"ProviderCatalogDisplay": objectSchema([]string{"name", "description", "category", "icon"}, map[string]JSONSchema{
			"name": stringSchema, "description": stringSchema, "category": stringSchema,
			"icon": refSchema("ProviderCatalogIcon"), "monogram": stringSchema, "accent": stringSchema,
		}),
		"ProviderCredentialPrompt": objectSchema([]string{"mode"}, map[string]JSONSchema{
			"mode":  {Type: "string", Enum: []string{"none", "optional", "required"}},
			"label": stringSchema, "hint": stringSchema,
		}),
		"ProviderOriginPrompt": objectSchema([]string{"mode", "baseUrlRequired"}, map[string]JSONSchema{
			"mode":       {Type: "string", Enum: []string{"fixed", "user_supplied"}},
			"defaultUrl": stringSchema, "baseUrlRequired": {Type: "boolean"},
			"label": stringSchema, "hint": stringSchema,
		}),
		"ProviderFieldOption": objectSchema([]string{"value", "label"}, map[string]JSONSchema{
			"value": stringSchema, "label": stringSchema,
		}),
		"ProviderConnectionField": objectSchema([]string{"name", "label", "kind", "required", "advanced"}, map[string]JSONSchema{
			"name": stringSchema, "label": stringSchema,
			"kind":     {Type: "string", Enum: []string{"text", "boolean", "integer", "select"}},
			"required": {Type: "boolean"}, "advanced": {Type: "boolean"},
			"default": stringSchema, "hint": stringSchema, "placeholder": stringSchema,
			"options": arraySchema(refSchema("ProviderFieldOption")),
		}),
		"ProviderInterface": objectSchema([]string{"id", "label", "default", "capabilities"}, map[string]JSONSchema{
			"id": stringSchema, "label": stringSchema, "default": {Type: "boolean"},
			"capabilities": arraySchema(stringSchema),
		}),
		"ProviderCatalogItem": objectSchema([]string{
			"providerId", "revision", "display", "credential",
			"origin", "discoverySupported", "capabilities", "connectionFields", "interfaces",
		}, map[string]JSONSchema{
			"providerId": stringSchema, "revision": stringSchema,
			"display":    refSchema("ProviderCatalogDisplay"),
			"credential": refSchema("ProviderCredentialPrompt"), "origin": refSchema("ProviderOriginPrompt"),
			"discoverySupported": {Type: "boolean"}, "capabilities": arraySchema(stringSchema),
			"connectionFields": arraySchema(refSchema("ProviderConnectionField")),
			"interfaces":       arraySchema(refSchema("ProviderInterface")),
		}),
		"ProviderCatalogPage": objectSchema([]string{"data", "page", "catalogRevision", "categories"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("ProviderCatalogItem")), "page": refSchema("PageInfo"),
			"catalogRevision": stringSchema, "categories": arraySchema(stringSchema),
		}),
		"ProviderCatalogDetail": objectSchema([]string{"data", "catalogRevision"}, map[string]JSONSchema{
			"data": refSchema("ProviderCatalogItem"), "catalogRevision": stringSchema,
		}),
		"DiscoverModelsRequest": objectSchema(nil, map[string]JSONSchema{
			"credentialId": stringSchema, "baseUrl": stringSchema,
			"connectionFields": object, "search": stringSchema,
			"pageSize": boundedIntegerSchema(1, 200), "cursor": stringSchema,
		}),
		"DiscoveredModel": objectSchema([]string{"catalogItemId", "providerModelId", "displayName", "capabilities"}, map[string]JSONSchema{
			"catalogItemId": stringSchema, "providerModelId": stringSchema,
			"displayName": stringSchema, "capabilities": arraySchema(stringSchema),
		}),
		"DiscoverModelsPage": objectSchema([]string{"data", "page", "catalogRevision", "discoveryRevision", "expiresAt"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("DiscoveredModel")), "page": refSchema("PageInfo"),
			"catalogRevision": stringSchema, "discoveryRevision": stringSchema, "expiresAt": timestampSchema,
		}),
		"IdempotencyMetadata": objectSchema([]string{"replayed"}, map[string]JSONSchema{
			"replayed": {Type: "boolean"}, "originalRequestId": stringSchema,
		}),
		"ResourceReference": objectSchema([]string{"kind", "id", "revision"}, map[string]JSONSchema{
			"kind": stringSchema, "id": stringSchema, "revision": {Type: "integer", Format: "int64"},
		}),
		"OperationReference": objectSchema([]string{"operationId"}, map[string]JSONSchema{
			"operationId": stringSchema, "desiredRevision": {Type: "integer", Format: "int64"},
		}),
		"MutationReceipt": {
			OneOf: []JSONSchema{
				objectSchema([]string{"resource"}, map[string]JSONSchema{
					"resource": refSchema("ResourceReference"), "idempotency": refSchema("IdempotencyMetadata"),
				}),
				objectSchema([]string{"operation"}, map[string]JSONSchema{
					"operation": refSchema("OperationReference"), "idempotency": refSchema("IdempotencyMetadata"),
				}),
			},
		},
		"UserCreateRequest": objectSchema([]string{"email", "displayName"}, map[string]JSONSchema{
			"email":       {Type: "string", Format: "email", MaxLength: intPointer(320)},
			"displayName": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
		}),
		"UserPatchRequest": objectSchema(nil, map[string]JSONSchema{
			"email":       {Type: "string", Format: "email", MaxLength: intPointer(320)},
			"displayName": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"status":      {Type: "string", Enum: []string{"active", "disabled"}},
		}),
		"User": objectSchema([]string{"userId", "email", "displayName", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
			"userId": stringSchema, "email": {Type: "string", Format: "email"}, "displayName": stringSchema,
			"status":   {Type: "string", Enum: []string{"active", "disabled", "deleted"}},
			"revision": {Type: "integer", Format: "int64"}, "createdAt": timestampSchema,
			"updatedAt": timestampSchema, "deletedAt": timestampSchema,
		}),
		"UserPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("User")), "page": refSchema("PageInfo"),
		}),
		"UserDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("User")}),
		"TeamCreateRequest": objectSchema([]string{"name"}, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)},
			"accessPolicyIds": {
				Type: "array", Items: schemaPointer(JSONSchema{Type: "string", Format: "uuid"}),
				MinItems: intPointer(1), UniqueItems: true,
				Description: "Active same-namespace AccessPolicies. Omit to use the namespace default.",
			},
			"rateLimitPolicyId": {
				Type: "string", Format: "uuid",
				Description: "Active same-namespace RateLimitPolicy. Omit to use the namespace default.",
			},
		}),
		"TeamPatchRequest": objectSchema(nil, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)},
			"status":      {Type: "string", Enum: []string{"active", "disabled"}},
		}),
		"Team": objectSchema([]string{"teamId", "name", "description", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
			"teamId": stringSchema, "name": stringSchema, "description": stringSchema,
			"status":   {Type: "string", Enum: []string{"active", "disabled"}},
			"revision": {Type: "integer", Format: "int64"}, "createdAt": timestampSchema,
			"updatedAt": timestampSchema, "deletedAt": timestampSchema,
		}),
		"TeamPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("Team")), "page": refSchema("PageInfo"),
		}),
		"TeamDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("Team")}),
		"MembershipPutRequest": objectSchema([]string{"role"}, map[string]JSONSchema{
			"role": {Type: "string", Enum: []string{"member", "admin"}},
		}),
		"MembershipPatchRequest": objectSchema(nil, map[string]JSONSchema{
			"role":   {Type: "string", Enum: []string{"member", "admin"}},
			"status": {Type: "string", Enum: []string{"active", "disabled"}},
		}),
		"Membership": objectSchema([]string{"teamId", "userId", "role", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
			"teamId": stringSchema, "userId": stringSchema,
			"role":     {Type: "string", Enum: []string{"member", "admin"}},
			"status":   {Type: "string", Enum: []string{"active", "disabled"}},
			"revision": {Type: "integer", Format: "int64"}, "createdAt": timestampSchema, "updatedAt": timestampSchema,
		}),
		"UserMembership": objectSchema([]string{"teamId", "userId", "role", "status", "revision", "createdAt", "updatedAt", "teamName", "teamStatus"}, map[string]JSONSchema{
			"teamId": stringSchema, "userId": stringSchema, "role": stringSchema, "status": stringSchema,
			"revision": {Type: "integer", Format: "int64"}, "createdAt": timestampSchema, "updatedAt": timestampSchema,
			"teamName": stringSchema, "teamStatus": {Type: "string", Enum: []string{"active", "disabled"}},
		}),
		"TeamMember": objectSchema([]string{"teamId", "userId", "role", "status", "revision", "createdAt", "updatedAt", "displayName", "userStatus"}, map[string]JSONSchema{
			"teamId": stringSchema, "userId": stringSchema, "role": stringSchema, "status": stringSchema,
			"revision": {Type: "integer", Format: "int64"}, "createdAt": timestampSchema, "updatedAt": timestampSchema,
			"displayName": stringSchema, "userStatus": {Type: "string", Enum: []string{"active", "disabled"}},
		}),
		"UserMembershipPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("UserMembership")), "page": refSchema("PageInfo"),
		}),
		"TeamMemberPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("TeamMember")), "page": refSchema("PageInfo"),
		}),
		"ProviderCredentialCreateRequest": objectSchema([]string{"name", "providerId", "secret"}, map[string]JSONSchema{
			"name": stringSchema, "providerId": stringSchema, "baseUrl": stringSchema,
			"secret": {Type: "string", Format: "password", Description: "Write-only backend credential secret."},
		}),
		"ProviderCredentialPatchRequest": {
			OneOf: []JSONSchema{
				objectSchema([]string{"name"}, map[string]JSONSchema{"name": stringSchema}),
				objectSchema([]string{"status"}, map[string]JSONSchema{"status": {Type: "string", Enum: []string{"disabled"}}}),
				objectSchema([]string{"status", "secret"}, map[string]JSONSchema{
					"status": {Type: "string", Enum: []string{"active"}},
					"secret": {Type: "string", Format: "password", Description: "Write-only replacement secret."},
				}),
			},
		},
		"ProviderCredentialRotateRequest": objectSchema([]string{"secret"}, map[string]JSONSchema{
			"secret": {Type: "string", Format: "password", Description: "Write-only replacement secret."},
		}),
		"ProviderCredential": objectSchema([]string{
			"credentialId", "name", "providerId", "catalogRevision", "normalizedOrigin",
			"status", "revision", "createdAt", "updatedAt",
		}, map[string]JSONSchema{
			"credentialId": stringSchema, "name": stringSchema, "providerId": stringSchema,
			"catalogRevision": stringSchema, "normalizedOrigin": stringSchema,
			"status":    {Type: "string", Enum: []string{"active", "disabled", "deleted"}},
			"revision":  {Type: "integer", Format: "int64"},
			"createdAt": timestampSchema, "updatedAt": timestampSchema, "deletedAt": timestampSchema,
		}),
		"ProviderCredentialPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("ProviderCredential")), "page": refSchema("PageInfo"),
		}),
		"ProviderCredentialDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": refSchema("ProviderCredential"),
		}),
	}
	return mergeOpenAPIExtensionSchemas(schemas)
}

func securitySchemes() map[string]OpenAPISecurityScheme {
	return map[string]OpenAPISecurityScheme{
		"managementBearer":    {Type: "http", Scheme: "bearer", BearerFormat: "Router Management JWT"},
		"serviceCredential":   {Type: "apiKey", Name: "Authorization", In: "header", Description: "VSR-Service credential."},
		"mutualTLS":           {Type: "mutualTLS", Description: "Listener-verified client certificate matched to an active Management identity."},
		"bootstrapCredential": {Type: "apiKey", Name: "Authorization", In: "header", Description: "VSR-Bootstrap credential on the private listener."},
		"recoveryCredential":  {Type: "apiKey", Name: "Authorization", In: "header", Description: "VSR-Recovery credential on the loopback-only route."},
		"issuerLogoutToken":   {Type: "http", Scheme: "bearer", BearerFormat: "OIDC back-channel logout token"},
	}
}

func objectSchema(required []string, properties map[string]JSONSchema) JSONSchema {
	return JSONSchema{Type: "object", Required: required, Properties: properties, AdditionalProperties: boolPointer(false)}
}

func arraySchema(item JSONSchema) JSONSchema {
	return JSONSchema{Type: "array", Items: &item}
}

func refSchema(name string) JSONSchema {
	return JSONSchema{Ref: "#/components/schemas/" + name}
}

func boundedIntegerSchema(minimum, maximum int64) JSONSchema {
	return JSONSchema{Type: "integer", Minimum: &minimum, Maximum: &maximum}
}

func boolPointer(value bool) *bool { return &value }

func intPointer(value int64) *int64 { return &value }

func schemaPointer(value JSONSchema) *JSONSchema { return &value }
