package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "invitations", Schemas: invitationSchemas,
		RequestSchema: invitationRequestSchema, ResponseSchema: invitationResponseSchema,
		ExtraParameters: invitationParameters,
	})
}

func invitationSchemas() map[string]JSONSchema {
	stringSchema := JSONSchema{Type: "string"}
	uuidSchema := JSONSchema{Type: "string", Format: "uuid"}
	timestampSchema := JSONSchema{Type: "string", Format: "date-time"}
	digestSchema := JSONSchema{Type: "string", Pattern: `^[a-f0-9]{64}$`}
	expectedIdentity := objectSchema([]string{"issuer"}, map[string]JSONSchema{
		"issuer":  {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(512)},
		"subject": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(512)},
		"email":   {Type: "string", Format: "email", MaxLength: intPointer(320)},
	})
	team := objectSchema([]string{"teamId", "role"}, map[string]JSONSchema{
		"teamId": uuidSchema, "role": {Type: "string", Enum: []string{"member", "admin"}},
	})
	requestedGrant := objectSchema([]string{"roleId", "scopeKind"}, map[string]JSONSchema{
		"roleId":            uuidSchema,
		"scopeKind":         {Type: "string", Enum: []string{"namespace", "user"}},
		"delegationCeiling": arraySchema(stringSchema),
	})
	pinnedGrant := objectSchema([]string{
		"roleId", "roleRevision", "rolePermissionsDigest", "scopeKind", "delegationCeiling",
		"sourceBindingId", "sourceBindingRevision", "sourceRoleId", "sourcePermissionsDigest",
	}, map[string]JSONSchema{
		"roleId": uuidSchema, "roleRevision": {Type: "integer", Format: "int64"},
		"rolePermissionsDigest": digestSchema,
		"scopeKind":             {Type: "string", Enum: []string{"namespace", "user"}},
		"delegationCeiling":     arraySchema(stringSchema),
		"sourceBindingId":       uuidSchema, "sourceBindingRevision": {Type: "integer", Format: "int64"},
		"sourceRoleId": uuidSchema, "sourcePermissionsDigest": digestSchema,
	})
	snapshot := objectSchema([]string{
		"roleGrants", "selfServicePolicyRevision", "accessPolicyId", "accessPolicyRevision",
		"rateLimitPolicyId", "rateLimitPolicyRevision", "automaticFirstKey",
	}, map[string]JSONSchema{
		"roleGrants": arraySchema(pinnedGrant), "team": team,
		"selfServicePolicyRevision": {Type: "integer", Format: "int64"},
		"accessPolicyId":            uuidSchema, "accessPolicyRevision": {Type: "integer", Format: "int64"},
		"rateLimitPolicyId": uuidSchema, "rateLimitPolicyRevision": {Type: "integer", Format: "int64"},
		"automaticFirstKey": {Type: "boolean"},
	})
	invitation := objectSchema([]string{
		"invitationId", "namespaceId", "createdByPrincipalId", "expectedIdentity", "displayName",
		"onboarding", "expiresAt", "status", "revision", "createdAt", "updatedAt",
	}, map[string]JSONSchema{
		"invitationId": uuidSchema, "namespaceId": uuidSchema, "createdByPrincipalId": uuidSchema,
		"expectedIdentity": expectedIdentity, "displayName": stringSchema, "onboarding": snapshot,
		"expiresAt":           timestampSchema,
		"status":              {Type: "string", Enum: []string{"pending", "accepted", "expired", "revoked"}},
		"acceptedPrincipalId": uuidSchema, "acceptedUserId": uuidSchema, "acceptedAt": timestampSchema,
		"acceptedManagementSessionId": uuidSchema,
		"revision":                    {Type: "integer", Format: "int64"}, "createdAt": timestampSchema, "updatedAt": timestampSchema,
	})
	return map[string]JSONSchema{
		"InvitationExpectedIdentity":   expectedIdentity,
		"InvitationTeamAssignment":     team,
		"InvitationRoleGrantRequest":   requestedGrant,
		"InvitationRoleGrant":          pinnedGrant,
		"InvitationOnboardingSnapshot": snapshot,
		"InvitationCreateRequest": objectSchema([]string{
			"expectedIdentity", "displayName", "roleGrants", "expiresAt",
		}, map[string]JSONSchema{
			"expectedIdentity": expectedIdentity,
			"displayName":      {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"roleGrants":       arraySchema(requestedGrant), "team": team, "expiresAt": timestampSchema,
		}),
		"InvitationRotateTokenRequest": objectSchema(nil, map[string]JSONSchema{"expiresAt": timestampSchema}),
		"OnboardingCreateRequest": objectSchema([]string{
			"principalId", "email", "displayName", "roleGrants", "createFirstKey",
		}, map[string]JSONSchema{
			"principalId": uuidSchema, "email": {Type: "string", Format: "email", MaxLength: intPointer(320)},
			"displayName": {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"roleGrants":  arraySchema(requestedGrant), "team": team, "createFirstKey": {Type: "boolean"},
		}),
		"Invitation": invitation,
		"InvitationPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(invitation), "page": refSchema("PageInfo"),
		}),
		"InvitationDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": invitation}),
		"InvitationIssuedSecret": objectSchema([]string{"data", "token", "deliveryExpiresAt"}, map[string]JSONSchema{
			"data":              invitation,
			"token":             {Type: "string", Format: "password", Description: "One-time invitation URL token."},
			"deliveryExpiresAt": timestampSchema,
		}),
		"OnboardingResult": objectSchema([]string{"principalId", "userId", "deliveryExpiresAt"}, map[string]JSONSchema{
			"invitationId": uuidSchema, "principalId": uuidSchema, "userId": uuidSchema, "teamId": uuidSchema,
			"apiKeyId":          uuidSchema,
			"apiKey":            {Type: "string", Format: "password", Description: "One-time first inference API key."},
			"deliveryExpiresAt": timestampSchema,
		}),
	}
}

func invitationRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/invitations":
		return "InvitationCreateRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/invitations/{invitationId}:rotate-token":
		return "InvitationRotateTokenRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/onboarding":
		return "OnboardingCreateRequest", true
	default:
		return "", false
	}
}

func invitationResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/invitations":
		return refSchema("InvitationPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/invitations/{invitationId}":
		return refSchema("InvitationDetail"), true
	case contract.Method == MethodPOST && (contract.Path == BasePath+"/invitations" ||
		contract.Path == BasePath+"/invitations/{invitationId}:rotate-token"):
		return refSchema("InvitationIssuedSecret"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/onboarding":
		return refSchema("OnboardingResult"), true
	default:
		return JSONSchema{}, false
	}
}

func invitationParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET || contract.Path != BasePath+"/invitations" {
		return nil
	}
	return []OpenAPIParameter{{
		Name: "status", In: "query",
		Schema: JSONSchema{Type: "string", Enum: []string{"pending", "accepted", "expired", "revoked"}},
	}}
}
