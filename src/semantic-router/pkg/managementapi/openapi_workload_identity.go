package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "workload-identity", Schemas: workloadIdentitySchemas,
		RequestSchema: workloadIdentityRequestSchema, ResponseSchema: workloadIdentityResponseSchema,
		ExtraParameters: workloadIdentityParameters,
	})
}

func workloadIdentityParameters(contract OperationContract) []OpenAPIParameter {
	key := string(contract.Method) + " " + contract.Path
	status := OpenAPIParameter{Name: "status", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"active", "disabled"}}}
	switch key {
	case "GET " + BasePath + "/service-accounts":
		return []OpenAPIParameter{
			{Name: "namespaceId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}}, status,
		}
	case "GET " + BasePath + "/mtls-identity-mappings":
		return []OpenAPIParameter{status}
	default:
		return nil
	}
}

func workloadIdentityRequestSchema(contract OperationContract) (string, bool) {
	requests := map[string]string{
		"POST " + BasePath + "/service-accounts":                                       "ServiceAccountCreateRequest",
		"PATCH " + BasePath + "/service-accounts/{serviceAccountId}":                   "ServiceAccountPatchRequest",
		"POST " + BasePath + "/service-accounts/{serviceAccountId}/credentials:rotate": "ServiceCredentialRotateRequest",
		"POST " + BasePath + "/mtls-identity-mappings":                                 "MTLSIdentityMappingCreateRequest",
		"PATCH " + BasePath + "/mtls-identity-mappings/{mappingId}":                    "MTLSIdentityMappingPatchRequest",
	}
	value, found := requests[string(contract.Method)+" "+contract.Path]
	return value, found
}

func workloadIdentityResponseSchema(contract OperationContract) (JSONSchema, bool) {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "GET " + BasePath + "/service-accounts":
		return refSchema("ServiceAccountPage"), true
	case "POST " + BasePath + "/service-accounts",
		"POST " + BasePath + "/service-accounts/{serviceAccountId}/credentials:rotate":
		return refSchema("ServiceCredentialIssue"), true
	case "GET " + BasePath + "/service-accounts/{serviceAccountId}":
		return refSchema("ServiceAccountDetail"), true
	case "GET " + BasePath + "/service-accounts/{serviceAccountId}/credentials":
		return refSchema("ServiceCredentialPage"), true
	case "GET " + BasePath + "/mtls-identity-mappings":
		return refSchema("MTLSIdentityMappingPage"), true
	case "GET " + BasePath + "/mtls-identity-mappings/{mappingId}":
		return refSchema("MTLSIdentityMappingDetail"), true
	}
	if (contract.Method == MethodPOST || contract.Method == MethodPATCH) &&
		(contract.Path == BasePath+"/mtls-identity-mappings" ||
			contract.Path == BasePath+"/mtls-identity-mappings/{mappingId}" ||
			contract.Path == BasePath+"/service-accounts/{serviceAccountId}") {
		return refSchema("MutationReceipt"), true
	}
	return JSONSchema{}, false
}

func workloadIdentitySchemas() map[string]JSONSchema {
	uuid := JSONSchema{Type: "string", Format: "uuid"}
	timestamp := JSONSchema{Type: "string", Format: "date-time"}
	integer := JSONSchema{Type: "integer", Format: "int64"}
	ownerScope := JSONSchema{Type: "string", Enum: []string{"cluster", "namespace"}}
	status := JSONSchema{Type: "string", Enum: []string{"active", "disabled"}}
	workloadClass := JSONSchema{Type: "string", Enum: []string{"workload_standard", "workload_strong"}}
	serviceAccount := objectSchema(
		[]string{"serviceAccountId", "principalId", "displayName", "ownerScope", "status", "revision", "createdAt", "updatedAt"},
		map[string]JSONSchema{
			"serviceAccountId": uuid, "principalId": uuid,
			"displayName": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)},
			"ownerScope":  ownerScope, "namespaceId": uuid, "status": status,
			"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
		},
	)
	credential := objectSchema(
		[]string{"credentialId", "serviceAccountId", "publicId", "workloadClass", "sourceAssuredAt", "status", "notBefore", "expiresAt", "createdAt"},
		map[string]JSONSchema{
			"credentialId": uuid, "serviceAccountId": uuid, "publicId": uuid,
			"workloadClass": workloadClass, "sourceAssuredAt": timestamp,
			"status":    {Type: "string", Enum: []string{"active", "retiring", "revoked"}},
			"notBefore": timestamp, "expiresAt": timestamp, "revokedAt": timestamp, "createdAt": timestamp,
		},
	)
	mapping := objectSchema(
		[]string{"mappingId", "matcherKind", "matcherValue", "principalId", "workloadClass", "sourceAssuredAt", "status", "revision", "createdAt", "updatedAt"},
		map[string]JSONSchema{
			"mappingId":    uuid,
			"matcherKind":  {Type: "string", Enum: []string{"spiffe_id", "san_uri", "san_dns", "subject_dn_sha256"}},
			"matcherValue": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(2048)},
			"principalId":  uuid, "workloadClass": workloadClass, "sourceAssuredAt": timestamp,
			"status": status, "revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
		},
	)
	reason := JSONSchema{Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(1000)}
	return map[string]JSONSchema{
		"ServiceAccount": serviceAccount,
		"ServiceAccountPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(serviceAccount), "page": refSchema("PageInfo"),
		}),
		"ServiceAccountDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": serviceAccount}),
		"ServiceAccountCreateRequest": objectSchema(
			[]string{"displayName", "ownerScope", "credentialExpiresAt", "credentialClass", "reason"},
			map[string]JSONSchema{
				"displayName": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)},
				"ownerScope":  ownerScope, "namespaceId": uuid, "credentialExpiresAt": timestamp,
				"credentialClass": workloadClass, "reason": reason,
			},
		),
		"ServiceAccountPatchRequest": objectSchema([]string{"reason"}, map[string]JSONSchema{
			"displayName": {Type: "string", MinLength: int64Pointer(1), MaxLength: int64Pointer(200)},
			"status":      status, "reason": reason,
		}),
		"ServiceCredential": credential,
		"ServiceCredentialPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(credential), "page": refSchema("PageInfo"),
		}),
		"ServiceCredentialRotateRequest": objectSchema(
			[]string{"expiresAt", "workloadClass", "overlapSeconds", "reason"},
			map[string]JSONSchema{
				"expiresAt": timestamp, "workloadClass": workloadClass,
				"overlapSeconds": {Type: "integer", Format: "int64", Minimum: int64Pointer(0), Maximum: int64Pointer(86400)},
				"reason":         reason,
			},
		),
		"ServiceCredentialIssue": objectSchema(
			[]string{"serviceAccount", "credential", "secret", "deliveryExpiresAt"},
			map[string]JSONSchema{
				"serviceAccount": serviceAccount, "credential": credential,
				"secret":            {Type: "string", Format: "password", Pattern: `^vsm_[0-9a-f-]{36}_[A-Za-z0-9_-]{43}$`},
				"deliveryExpiresAt": timestamp,
			},
		),
		"MTLSIdentityMapping": mapping,
		"MTLSIdentityMappingPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(mapping), "page": refSchema("PageInfo"),
		}),
		"MTLSIdentityMappingDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": mapping}),
		"MTLSIdentityMappingCreateRequest": objectSchema(
			[]string{"matcherKind", "matcherValue", "principalId", "workloadClass", "reason"},
			map[string]JSONSchema{
				"matcherKind": mapping.Properties["matcherKind"], "matcherValue": mapping.Properties["matcherValue"],
				"principalId": uuid, "workloadClass": workloadClass, "reason": reason,
			},
		),
		"MTLSIdentityMappingPatchRequest": objectSchema([]string{"reason"}, map[string]JSONSchema{
			"status": status, "workloadClass": workloadClass, "reason": reason,
		}),
	}
}
