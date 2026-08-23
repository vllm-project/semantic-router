package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "principal-directory", Schemas: principalDirectorySchemas,
		RequestSchema:   principalDirectoryRequestSchema,
		ResponseSchema:  principalDirectoryResponseSchema,
		ExtraParameters: principalDirectoryParameters,
	})
}

func principalDirectorySchemas() map[string]JSONSchema {
	uuid := JSONSchema{Type: "string", Format: "uuid"}
	text := JSONSchema{Type: "string"}
	integer := JSONSchema{Type: "integer", Format: "int64"}
	timestamp := JSONSchema{Type: "string", Format: "date-time"}
	directory := objectSchema(
		[]string{"principalId", "displayName", "status", "linked"},
		map[string]JSONSchema{
			"principalId": uuid, "displayName": text,
			"verifiedEmail": {Type: "string", Format: "email"},
			"status":        {Type: "string", Enum: []string{"active", "disabled"}},
			"linked":        {Type: "boolean"}, "userId": uuid, "linkRevision": integer,
		},
	)
	link := objectSchema(
		[]string{"principalId", "namespaceId", "userId", "revision", "createdAt", "updatedAt"},
		map[string]JSONSchema{
			"principalId": uuid, "namespaceId": uuid, "userId": uuid,
			"revision": integer, "createdAt": timestamp, "updatedAt": timestamp,
		},
	)
	return map[string]JSONSchema{
		"PrincipalDirectoryEntry": directory,
		"PrincipalDirectoryDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": directory,
		}),
		"PrincipalDirectoryPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(directory), "page": refSchema("PageInfo"),
		}),
		"PrincipalUserLink": link,
		"PrincipalUserLinkDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": link,
		}),
		"PrincipalUserLinkPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(link), "page": refSchema("PageInfo"),
		}),
		"PrincipalUserLinkPutRequest": objectSchema([]string{"userId"}, map[string]JSONSchema{
			"userId": uuid,
		}),
	}
}

func principalDirectoryRequestSchema(contract OperationContract) (string, bool) {
	if contract.Method == MethodPUT &&
		contract.Path == BasePath+"/namespaces/{namespaceId}/principal-user-links/{principalId}" {
		return "PrincipalUserLinkPutRequest", true
	}
	return "", false
}

func principalDirectoryResponseSchema(contract OperationContract) (JSONSchema, bool) {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "GET " + BasePath + "/namespaces/{namespaceId}/principal-directory":
		return refSchema("PrincipalDirectoryPage"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}/principal-directory/{principalId}":
		return refSchema("PrincipalDirectoryDetail"), true
	case "GET " + BasePath + "/namespaces/{namespaceId}/principal-user-links",
		"GET " + BasePath + "/management-principals/{principalId}/user-links":
		return refSchema("PrincipalUserLinkPage"), true
	case "PUT " + BasePath + "/namespaces/{namespaceId}/principal-user-links/{principalId}":
		return refSchema("PrincipalUserLinkDetail"), true
	default:
		return JSONSchema{}, false
	}
}

func principalDirectoryParameters(contract OperationContract) []OpenAPIParameter {
	key := string(contract.Method) + " " + contract.Path
	switch key {
	case "GET " + BasePath + "/namespaces/{namespaceId}/principal-directory":
		return []OpenAPIParameter{{
			Name: "search", In: "query",
			Schema: JSONSchema{Type: "string", MaxLength: int64Pointer(128)},
		}}
	case "GET " + BasePath + "/namespaces/{namespaceId}/principal-user-links":
		return []OpenAPIParameter{
			{Name: "principalId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
			{Name: "userId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
		}
	case "PUT " + BasePath + "/namespaces/{namespaceId}/principal-user-links/{principalId}":
		return []OpenAPIParameter{{
			Name: HeaderIfMatch, In: "header", Required: false,
			Description: "Required when replacing an existing link; omit only when creating one.",
			Schema:      JSONSchema{Type: "string", Pattern: `^\"principal-user-link:[1-9][0-9]*\"$`},
		}}
	default:
		return nil
	}
}
