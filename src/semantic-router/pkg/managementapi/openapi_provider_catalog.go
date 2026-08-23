package managementapi

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name:           "provider-catalog-publication",
		Schemas:        providerCatalogPublicationSchemas,
		RequestSchema:  providerCatalogPublicationRequestSchema,
		ResponseSchema: providerCatalogPublicationResponseSchema,
	})
}

func providerCatalogPublicationSchemas() map[string]JSONSchema {
	positiveGeneration := JSONSchema{Type: "string", Pattern: `^[1-9][0-9]*$`}
	revision := JSONSchema{Type: "string", Pattern: `^sha256:[0-9a-f]{64}$`}
	return map[string]JSONSchema{
		"ProviderCatalogBootstrapRequest": objectSchema([]string{"expectedGeneration"}, map[string]JSONSchema{
			"expectedGeneration": positiveGeneration,
		}),
		"ProviderCatalogActivateRequest": objectSchema([]string{"revision", "expectedGeneration"}, map[string]JSONSchema{
			"revision": revision, "expectedGeneration": positiveGeneration,
		}),
		"ProviderCatalogPublication": objectSchema([]string{"desiredRevision", "generation", "updatedAt"}, map[string]JSONSchema{
			"desiredRevision": revision, "activeRevision": revision,
			"generation": positiveGeneration, "updatedAt": {Type: "string", Format: "date-time"},
		}),
	}
}

func providerCatalogPublicationRequestSchema(contract OperationContract) (string, bool) {
	if contract.Method != MethodPOST {
		return "", false
	}
	switch contract.Path {
	case BasePath + "/provider-catalog:bootstrap":
		return "ProviderCatalogBootstrapRequest", true
	case BasePath + "/provider-catalog:activate":
		return "ProviderCatalogActivateRequest", true
	default:
		return "", false
	}
}

func providerCatalogPublicationResponseSchema(contract OperationContract) (JSONSchema, bool) {
	if contract.Method == MethodPOST &&
		(contract.Path == BasePath+"/provider-catalog:bootstrap" ||
			contract.Path == BasePath+"/provider-catalog:activate") {
		return refSchema("ProviderCatalogPublication"), true
	}
	return JSONSchema{}, false
}
