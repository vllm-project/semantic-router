package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "identity", Schemas: identitySchemas,
		RequestSchema: identityRequestSchema, ResponseSchema: identityResponseSchema,
		ExtraParameters: identityParameters,
	})
}

func identityParameters(contract OperationContract) []OpenAPIParameter {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "GET " + BasePath + "/management-roles":
		return []OpenAPIParameter{{Name: "namespaceId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}}}
	case "GET " + BasePath + "/role-bindings":
		return []OpenAPIParameter{{Name: "principalId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}}}
	default:
		return nil
	}
}

func identityRequestSchema(contract OperationContract) (string, bool) {
	schemas := map[string]string{
		"POST " + BasePath + "/auth/bootstrap":                                                     "BootstrapRequest",
		"POST " + BasePath + "/auth/recovery":                                                      "RecoveryRequest",
		"POST " + BasePath + "/auth/exchange-challenges":                                           "ExchangeChallengeRequest",
		"POST " + BasePath + "/auth/token-exchange":                                                "TokenExchangeRequest",
		"POST " + BasePath + "/auth/backchannel-logout":                                            "BackchannelLogoutRequest",
		"POST " + BasePath + "/management-sessions/{sessionId}:revoke":                             "ManagementSessionRevokeRequest",
		"POST " + BasePath + "/management-principals/{principalId}/management-sessions:revoke-all": "ManagementSessionRevokeRequest",
		"POST " + BasePath + "/trusted-identity-issuers":                                           "TrustedIdentityIssuerCreateRequest",
		"PATCH " + BasePath + "/trusted-identity-issuers/{issuerId}":                               "TrustedIdentityIssuerPatchRequest",
		"POST " + BasePath + "/trusted-identity-issuers/{issuerId}:refresh-keys":                   "TrustedIdentityIssuerRefreshRequest",
		"POST " + BasePath + "/self/inference-sessions":                                            "DelegatedInferenceSessionCreateRequest",
		"POST " + BasePath + "/management-principals":                                              "ManagementPrincipalCreateRequest",
		"PATCH " + BasePath + "/management-principals/{principalId}":                               "ManagementPrincipalPatchRequest",
		"POST " + BasePath + "/management-roles":                                                   "ManagementRoleCreateRequest",
		"PATCH " + BasePath + "/management-roles/{roleId}":                                         "ManagementRolePatchRequest",
		"POST " + BasePath + "/role-bindings":                                                      "ManagementRoleBindingCreateRequest",
		"PATCH " + BasePath + "/role-bindings/{bindingId}":                                         "ManagementRoleBindingPatchRequest",
		"PATCH " + BasePath + "/management-session-policy":                                         "ManagementSessionPolicyPatchRequest",
	}
	value, found := schemas[string(contract.Method)+" "+contract.Path]
	return value, found
}

func identityResponseSchema(contract OperationContract) (JSONSchema, bool) {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "POST " + BasePath + "/auth/bootstrap":
		return refSchema("BootstrapResponse"), true
	case "POST " + BasePath + "/auth/recovery":
		return refSchema("RecoveryResponse"), true
	case "POST " + BasePath + "/auth/exchange-challenges":
		return refSchema("ExchangeChallengeResponse"), true
	case "POST " + BasePath + "/auth/token-exchange":
		return refSchema("TokenExchangeResponse"), true
	case "POST " + BasePath + "/auth/service-token":
		return refSchema("ManagementTokenEnvelope"), true
	case "POST " + BasePath + "/auth/backchannel-logout":
		return refSchema("BackchannelLogoutResponse"), true
	case "GET " + BasePath + "/me":
		return refSchema("Me"), true
	case "GET " + BasePath + "/self/management-sessions",
		"GET " + BasePath + "/management-principals/{principalId}/management-sessions":
		return refSchema("ManagementSessionPage"), true
	case "POST " + BasePath + "/management-sessions/{sessionId}:revoke":
		return refSchema("ManagementSessionRevocation"), true
	case "POST " + BasePath + "/management-principals/{principalId}/management-sessions:revoke-all":
		return refSchema("PrincipalManagementSessionsRevocation"), true
	case "GET " + BasePath + "/trusted-identity-issuers":
		return refSchema("TrustedIdentityIssuerPage"), true
	case "GET " + BasePath + "/trusted-identity-issuers/{issuerId}":
		return refSchema("TrustedIdentityIssuerDetail"), true
	case "GET " + BasePath + "/self/inference-keys":
		return refSchema("EligibleInferenceKeyPage"), true
	case "GET " + BasePath + "/self/inference-keys/{keyId}":
		return refSchema("EligibleInferenceKeyDetail"), true
	case "GET " + BasePath + "/self/inference-sessions":
		return refSchema("DelegatedInferenceSessionPage"), true
	case "GET " + BasePath + "/api-keys/{keyId}/inference-sessions":
		return refSchema("DelegatedInferenceSessionPage"), true
	case "GET " + BasePath + "/management-principals":
		return refSchema("ManagementPrincipalPage"), true
	case "GET " + BasePath + "/management-principals/{principalId}":
		return refSchema("ManagementPrincipalDetail"), true
	case "GET " + BasePath + "/management-roles":
		return refSchema("ManagementRolePage"), true
	case "GET " + BasePath + "/management-roles/{roleId}":
		return refSchema("ManagementRoleDetail"), true
	case "GET " + BasePath + "/role-bindings":
		return refSchema("ManagementRoleBindingPage"), true
	case "GET " + BasePath + "/role-bindings/{bindingId}":
		return refSchema("ManagementRoleBindingDetail"), true
	case "GET " + BasePath + "/management-session-policy":
		return refSchema("ManagementSessionPolicyDetail"), true
	}
	if (contract.Method == MethodPOST || contract.Method == MethodPATCH) &&
		(contract.Path == BasePath+"/management-principals" || contract.Path == BasePath+"/management-principals/{principalId}" ||
			contract.Path == BasePath+"/trusted-identity-issuers" || contract.Path == BasePath+"/trusted-identity-issuers/{issuerId}" ||
			contract.Path == BasePath+"/trusted-identity-issuers/{issuerId}:refresh-keys" ||
			contract.Path == BasePath+"/management-roles" || contract.Path == BasePath+"/management-roles/{roleId}" ||
			contract.Path == BasePath+"/role-bindings" || contract.Path == BasePath+"/role-bindings/{bindingId}" ||
			contract.Path == BasePath+"/management-session-policy") {
		return refSchema("MutationReceipt"), true
	}
	return JSONSchema{}, false
}

var (
	text        = JSONSchema{Type: "string"}
	uuid        = JSONSchema{Type: "string", Format: "uuid"}
	timestamp   = JSONSchema{Type: "string", Format: "date-time"}
	integer     = JSONSchema{Type: "integer", Format: "int64"}
	stringArray = arraySchema(text)
	stringMap   = JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}
	pageInfo    = refSchema("PageInfo")
	scope       = objectSchema([]string{"kind"}, map[string]JSONSchema{
		"kind":        {Type: "string", Enum: []string{"cluster", "namespace", "team", "user", "resource"}},
		"namespaceId": uuid, "teamId": uuid, "userId": uuid, "resourceType": text, "resourceId": text,
	})
	principal = objectSchema([]string{"principalId", "issuer", "subject", "displayName", "attributes", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"principalId": uuid, "issuer": text, "subject": text, "displayName": text, "verifiedEmail": text,
		"attributes": stringMap, "status": {Type: "string", Enum: []string{"active", "disabled"}},
		"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
	})
	role = objectSchema([]string{"roleId", "name", "displayName", "description", "permissions", "builtIn", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"roleId": uuid, "namespaceId": uuid, "name": text, "displayName": text, "description": text,
		"permissions": stringArray, "builtIn": {Type: "boolean"}, "status": {Type: "string", Enum: []string{"active", "disabled"}},
		"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
	})
	binding = objectSchema([]string{"bindingId", "principalId", "roleId", "scope", "delegationCeiling", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"bindingId": uuid, "principalId": uuid, "roleId": uuid, "scope": scope,
		"delegationCeiling": stringArray, "status": {Type: "string", Enum: []string{"active", "disabled"}},
		"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
	})
	bootstrapExternalIdentity = objectSchema(
		[]string{"issuerId", "issuer", "subject", "discoveryUrl", "audience"},
		map[string]JSONSchema{
			"issuerId": uuid, "issuer": {Type: "string", Format: "uri"},
			"subject":      {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(512)},
			"discoveryUrl": {Type: "string", Format: "uri"},
			"audience":     {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(512)},
		},
	)
	bootstrapServiceRequest = objectSchema(
		[]string{"kind", "displayName"},
		map[string]JSONSchema{
			"kind":        {Type: "string", Enum: []string{"service_account"}},
			"displayName": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)},
		},
	)
	bootstrapExternalRequest = objectSchema(
		[]string{"kind", "displayName", "external"},
		map[string]JSONSchema{
			"kind":        {Type: "string", Enum: []string{"external_principal"}},
			"displayName": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)},
			"external":    bootstrapExternalIdentity,
		},
	)
	bootstrapCredential = objectSchema(
		[]string{"resourceId", "kind", "secret", "expiresAt"},
		map[string]JSONSchema{
			"resourceId": uuid,
			"kind":       {Type: "string", Enum: []string{string(SecretKindServiceCredential)}},
			"secret":     {Type: "string", Format: "password"},
			"expiresAt":  timestamp,
		},
	)
	bootstrapExternalResponse = objectSchema(
		[]string{"principalId", "roleBindingId", "finalizationRequired"},
		map[string]JSONSchema{
			"principalId": uuid, "roleBindingId": uuid,
			"finalizationRequired": {Type: "boolean"},
		},
	)
	bootstrapServiceResponse = objectSchema(
		[]string{"principalId", "roleBindingId", "serviceAccountId", "serviceCredential", "finalizationRequired"},
		map[string]JSONSchema{
			"principalId": uuid, "roleBindingId": uuid, "serviceAccountId": uuid,
			"serviceCredential":    bootstrapCredential,
			"finalizationRequired": {Type: "boolean"},
		},
	)
	tokenExchangeProperties = map[string]JSONSchema{
		"issuerId": uuid, "exchangeChallengeId": uuid,
		"subjectToken":     {Type: "string", Format: "password", MinLength: int64Pointer(1), MaxLength: int64Pointer(65536)},
		"subjectTokenType": {Type: "string", Enum: []string{"oidc_id_token", "router_local_assertion"}},
	}
	standardTokenExchange = objectSchema(
		[]string{"issuerId", "exchangeChallengeId", "subjectToken", "subjectTokenType"},
		tokenExchangeProperties,
	)
	invitedTokenExchangeProperties = schemaPropertiesWith(tokenExchangeProperties, "invitationToken", JSONSchema{
		Type: "string", Format: "password", Pattern: `^vsi_[0-9a-f-]{36}_[A-Za-z0-9_-]{43}$`,
	})
	invitedTokenExchange = objectSchema(
		[]string{"issuerId", "exchangeChallengeId", "subjectToken", "subjectTokenType", "invitationToken"},
		invitedTokenExchangeProperties,
	)
	tokenResponseProperties = map[string]JSONSchema{
		"accessToken": {Type: "string", Format: "password"}, "tokenType": {Type: "string", Enum: []string{"Bearer"}},
		"expiresIn": integer, "managementSessionId": uuid,
	}
	standardTokenResponse = objectSchema(
		[]string{"accessToken", "tokenType", "expiresIn", "managementSessionId"}, tokenResponseProperties,
	)
	invitedTokenResponseProperties = schemaPropertiesWith(
		tokenResponseProperties, "onboarding", refSchema("OnboardingResult"),
	)
	invitedTokenResponse = objectSchema(
		[]string{"accessToken", "tokenType", "expiresIn", "managementSessionId", "onboarding"},
		invitedTokenResponseProperties,
	)
	apiKeyOwner = objectSchema([]string{"type", "id"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"user", "team"}}, "id": uuid,
	})
	eligibleKey = objectSchema([]string{"keyId", "name", "owner"}, map[string]JSONSchema{
		"keyId": uuid, "name": text, "owner": apiKeyOwner, "contextTeamId": uuid, "expiresAt": timestamp,
	})
	delegatedSession = objectSchema(
		[]string{"sessionId", "publicId", "keyId", "userId", "audience", "status", "notBefore", "expiresAt", "createdAt"},
		map[string]JSONSchema{
			"sessionId": uuid, "publicId": text, "keyId": uuid, "userId": uuid, "teamId": uuid,
			"audience": text, "status": {Type: "string", Enum: []string{"active", "revoked", "expired"}},
			"notBefore": timestamp, "expiresAt": timestamp, "createdAt": timestamp,
		},
	)
	managementSession = objectSchema(
		[]string{"sessionId", "principalId", "authSourceKind", "evidenceKind", "authenticatedAt", "expiresAt", "status", "createdAt"},
		map[string]JSONSchema{
			"sessionId": uuid, "principalId": uuid,
			"authSourceKind":  {Type: "string", Enum: []string{"issuer", "service_credential", "mtls"}},
			"evidenceKind":    {Type: "string", Enum: []string{"human", "workload"}},
			"authenticatedAt": timestamp, "expiresAt": timestamp,
			"status":    {Type: "string", Enum: []string{"active", "revoked", "expired"}},
			"revokedAt": timestamp, "createdAt": timestamp,
		},
	)
	trustedIssuer = objectSchema(
		[]string{"issuerId", "issuer", "kind", "audiences", "claimMapping", "assuranceMapping", "status", "revision", "createdAt", "updatedAt"},
		map[string]JSONSchema{
			"issuerId": uuid, "issuer": {Type: "string", Format: "uri"},
			"kind":         {Type: "string", Enum: []string{"oidc", "jwt"}},
			"discoveryUrl": {Type: "string", Format: "uri"}, "jwksUrl": {Type: "string", Format: "uri"},
			"audiences": stringArray, "claimMapping": stringMap, "assuranceMapping": stringMap,
			"status":   {Type: "string", Enum: []string{"active", "disabled"}},
			"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
		},
	)
	mePrincipal = objectSchema([]string{"principalId", "displayName", "kind", "status"}, map[string]JSONSchema{
		"principalId": uuid, "displayName": text,
		"kind":   {Type: "string", Enum: []string{"human", "workload"}},
		"status": {Type: "string", Enum: []string{"active", "disabled"}},
	})
	meSession = objectSchema([]string{"sessionId", "authenticatedAt", "expiresAt", "evidenceKind"}, map[string]JSONSchema{
		"sessionId": uuid, "authenticatedAt": timestamp, "expiresAt": timestamp,
		"evidenceKind": {Type: "string", Enum: []string{"human", "workload"}},
	})
	meNamespace = objectSchema([]string{"namespaceId", "name", "status", "desiredRevision", "appliedRevision"}, map[string]JSONSchema{
		"namespaceId": uuid, "name": text, "status": {Type: "string", Enum: []string{"active", "disabled"}},
		"desiredRevision": integer, "appliedRevision": integer,
	})
	meUser = objectSchema([]string{"userId", "email", "displayName", "status"}, map[string]JSONSchema{
		"userId": uuid, "email": {Type: "string", Format: "email"}, "displayName": text,
		"status": {Type: "string", Enum: []string{"active", "disabled"}},
	})
	meTeam = objectSchema([]string{"teamId", "name", "role", "status"}, map[string]JSONSchema{
		"teamId": uuid, "name": text, "role": {Type: "string", Enum: []string{"member", "admin"}},
		"status": {Type: "string", Enum: []string{"active", "disabled"}},
	})
	selfService = objectSchema(
		[]string{"maxKeysPerUser", "maxDelegatedSessions", "delegatedSessionTtlSeconds", "allowTeamKeyDelegation", "automaticFirstKey", "revision"},
		map[string]JSONSchema{
			"maxKeysPerUser": integer, "maxDelegatedSessions": integer, "delegatedSessionTtlSeconds": integer,
			"allowTeamKeyDelegation": {Type: "boolean"}, "automaticFirstKey": {Type: "boolean"}, "revision": integer,
		},
	)
	meScope = objectSchema(
		[]string{"namespace", "permissions", "roleBindings", "teams", "selfServicePolicy"},
		map[string]JSONSchema{
			"namespace": meNamespace, "permissions": stringArray, "roleBindings": arraySchema(binding),
			"user": meUser, "teams": arraySchema(meTeam), "selfServicePolicy": selfService,
		},
	)
)

var identitySchemaCatalog = map[string]JSONSchema{
	"BootstrapExternalIdentity":          bootstrapExternalIdentity,
	"BootstrapServiceAccountRequest":     bootstrapServiceRequest,
	"BootstrapExternalPrincipalRequest":  bootstrapExternalRequest,
	"BootstrapRequest":                   {OneOf: []JSONSchema{refSchema("BootstrapServiceAccountRequest"), refSchema("BootstrapExternalPrincipalRequest")}},
	"BootstrapServiceCredential":         bootstrapCredential,
	"BootstrapExternalPrincipalResponse": bootstrapExternalResponse,
	"BootstrapServiceAccountResponse":    bootstrapServiceResponse,
	"BootstrapResponse":                  {OneOf: []JSONSchema{refSchema("BootstrapExternalPrincipalResponse"), refSchema("BootstrapServiceAccountResponse")}},
	"RecoveryRequest": objectSchema([]string{"principalId", "reason"}, map[string]JSONSchema{
		"principalId": uuid,
		"reason":      {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(500)},
	}),
	"RecoveryResponse": objectSchema([]string{"principalId", "roleBindingId", "recoveryDisableRequired"}, map[string]JSONSchema{
		"principalId": uuid, "roleBindingId": uuid, "recoveryDisableRequired": {Type: "boolean"},
	}),
	"ExchangeChallengeRequest":     objectSchema([]string{"issuerId"}, map[string]JSONSchema{"issuerId": uuid}),
	"ExchangeChallengeResponse":    objectSchema([]string{"exchangeChallengeId", "nonce", "expiresAt"}, map[string]JSONSchema{"exchangeChallengeId": uuid, "nonce": text, "expiresAt": timestamp}),
	"StandardTokenExchangeRequest": standardTokenExchange,
	"InvitedTokenExchangeRequest":  invitedTokenExchange,
	"TokenExchangeRequest": {OneOf: []JSONSchema{
		refSchema("StandardTokenExchangeRequest"), refSchema("InvitedTokenExchangeRequest"),
	}},
	"StandardTokenExchangeResponse": standardTokenResponse,
	"InvitedTokenExchangeResponse":  invitedTokenResponse,
	"TokenExchangeResponse": {OneOf: []JSONSchema{
		refSchema("StandardTokenExchangeResponse"), refSchema("InvitedTokenExchangeResponse"),
	}},
	"BackchannelLogoutRequest": objectSchema([]string{"issuerId", "logoutToken"}, map[string]JSONSchema{
		"issuerId": uuid, "logoutToken": {Type: "string", Format: "password", MinLength: int64Pointer(1), MaxLength: int64Pointer(65536)},
	}),
	"BackchannelLogoutResponse": objectSchema([]string{"applied", "replayed"}, map[string]JSONSchema{
		"applied": {Type: "boolean"}, "replayed": {Type: "boolean"},
	}),
	"MePrincipal":                            mePrincipal,
	"MeSession":                              meSession,
	"MeNamespace":                            meNamespace,
	"MeUser":                                 meUser,
	"MeTeamMembership":                       meTeam,
	"MeSelfServicePolicy":                    selfService,
	"MeNamespaceScope":                       meScope,
	"Me":                                     objectSchema([]string{"principal", "session", "clusterPermissions", "namespaces"}, map[string]JSONSchema{"principal": mePrincipal, "session": meSession, "clusterPermissions": stringArray, "namespaces": arraySchema(meScope)}),
	"ManagementSession":                      managementSession,
	"ManagementSessionPage":                  objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(managementSession), "page": pageInfo}),
	"ManagementSessionRevokeRequest":         objectSchema([]string{"reason"}, map[string]JSONSchema{"reason": text}),
	"ManagementSessionRevocation":            objectSchema([]string{"sessionId", "status", "revokedAt", "changed"}, map[string]JSONSchema{"sessionId": uuid, "status": {Type: "string", Enum: []string{"revoked"}}, "revokedAt": timestamp, "changed": {Type: "boolean"}}),
	"PrincipalManagementSessionsRevocation":  objectSchema([]string{"principalId", "revokedCount", "alreadyRevoked"}, map[string]JSONSchema{"principalId": uuid, "revokedCount": integer, "alreadyRevoked": integer}),
	"TrustedIdentityIssuer":                  trustedIssuer,
	"TrustedIdentityIssuerPage":              objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(trustedIssuer), "page": pageInfo}),
	"TrustedIdentityIssuerDetail":            objectSchema([]string{"data"}, map[string]JSONSchema{"data": trustedIssuer}),
	"TrustedIdentityIssuerCreateRequest":     objectSchema([]string{"issuer", "kind", "audiences"}, map[string]JSONSchema{"issuer": {Type: "string", Format: "uri"}, "kind": {Type: "string", Enum: []string{"oidc", "jwt"}}, "discoveryUrl": {Type: "string", Format: "uri"}, "jwksUrl": {Type: "string", Format: "uri"}, "audiences": stringArray, "claimMapping": stringMap, "assuranceMapping": stringMap}),
	"TrustedIdentityIssuerPatchRequest":      objectSchema([]string{"reason"}, map[string]JSONSchema{"discoveryUrl": {Type: "string", Format: "uri"}, "jwksUrl": {Type: "string", Format: "uri"}, "audiences": stringArray, "claimMapping": stringMap, "assuranceMapping": stringMap, "status": {Type: "string", Enum: []string{"active", "disabled"}}, "reason": text}),
	"TrustedIdentityIssuerRefreshRequest":    objectSchema([]string{"reason"}, map[string]JSONSchema{"reason": text}),
	"EligibleInferenceKey":                   eligibleKey,
	"EligibleInferenceKeyPage":               objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(eligibleKey), "page": pageInfo}),
	"EligibleInferenceKeyDetail":             objectSchema([]string{"data"}, map[string]JSONSchema{"data": eligibleKey}),
	"DelegatedInferenceSessionCreateRequest": objectSchema([]string{"keyId"}, map[string]JSONSchema{"keyId": uuid}),
	"DelegatedInferenceSession":              delegatedSession,
	"DelegatedInferenceSessionPage":          objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(delegatedSession), "page": pageInfo}),
	"ManagementPrincipal":                    principal,
	"ManagementPrincipalPage":                objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(principal), "page": refSchema("PageInfo")}),
	"ManagementPrincipalDetail":              objectSchema([]string{"data"}, map[string]JSONSchema{"data": principal}),
	"ManagementPrincipalCreateRequest":       objectSchema([]string{"issuer", "subject", "displayName"}, map[string]JSONSchema{"issuer": text, "subject": text, "displayName": text, "verifiedEmail": text, "attributes": stringMap}),
	"ManagementPrincipalPatchRequest":        objectSchema([]string{"reason"}, map[string]JSONSchema{"displayName": text, "verifiedEmail": text, "status": {Type: "string", Enum: []string{"active", "disabled"}}, "reason": text}),
	"ManagementRole":                         role,
	"ManagementRolePage":                     objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(role), "page": refSchema("PageInfo")}),
	"ManagementRoleDetail":                   objectSchema([]string{"data"}, map[string]JSONSchema{"data": role}),
	"ManagementRoleCreateRequest":            objectSchema([]string{"namespaceId", "name", "displayName", "permissions"}, map[string]JSONSchema{"namespaceId": uuid, "name": text, "displayName": text, "description": text, "permissions": stringArray}),
	"ManagementRolePatchRequest":             objectSchema([]string{"reason"}, map[string]JSONSchema{"displayName": text, "description": text, "reason": text}),
	"ManagementScope":                        scope,
	"ManagementRoleBinding":                  binding,
	"ManagementRoleBindingPage":              objectSchema([]string{"data", "page"}, map[string]JSONSchema{"data": arraySchema(binding), "page": refSchema("PageInfo")}),
	"ManagementRoleBindingDetail":            objectSchema([]string{"data"}, map[string]JSONSchema{"data": binding}),
	"ManagementRoleBindingCreateRequest":     objectSchema([]string{"principalId", "roleId", "scope", "delegationCeiling"}, map[string]JSONSchema{"principalId": uuid, "roleId": uuid, "scope": scope, "delegationCeiling": stringArray}),
	"ManagementRoleBindingPatchRequest":      objectSchema([]string{"status", "reason"}, map[string]JSONSchema{"status": {Type: "string", Enum: []string{"active", "disabled"}}, "reason": text}),
	"ManagementSessionPolicy":                objectSchema([]string{"accessTokenTtlSeconds", "sessionTtlSeconds", "maxActiveSessions", "actionRequirements", "seedVersion", "revision", "updatedAt"}, map[string]JSONSchema{"accessTokenTtlSeconds": integer, "sessionTtlSeconds": integer, "maxActiveSessions": integer, "actionRequirements": stringMap, "seedVersion": integer, "revision": integer, "updatedAt": timestamp}),
	"ManagementSessionPolicyPatchRequest":    objectSchema([]string{"accessTokenTtlSeconds", "sessionTtlSeconds", "maxActiveSessions", "actionRequirements", "reason"}, map[string]JSONSchema{"accessTokenTtlSeconds": integer, "sessionTtlSeconds": integer, "maxActiveSessions": integer, "actionRequirements": stringMap, "reason": text}),
	"ManagementSessionPolicyDetail":          objectSchema([]string{"data"}, map[string]JSONSchema{"data": refSchema("ManagementSessionPolicy")}),
}

func identitySchemas() map[string]JSONSchema {
	return cloneSchemas(identitySchemaCatalog)
}

func schemaPropertiesWith(source map[string]JSONSchema, name string, schema JSONSchema) map[string]JSONSchema {
	result := make(map[string]JSONSchema, len(source)+1)
	for key, value := range source {
		result[key] = value
	}
	result[name] = schema
	return result
}

func int64Pointer(value int64) *int64 { return &value }
