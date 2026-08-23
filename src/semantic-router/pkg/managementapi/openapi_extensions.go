package managementapi

import "fmt"

// openAPIExtension keeps resource-family DTO knowledge out of the central
// protocol generator. Extensions are registered during package initialization
// and remain immutable for the lifetime of the process.
type openAPIExtension struct {
	Name            string
	Schemas         func() map[string]JSONSchema
	RequestSchema   func(OperationContract) (string, bool)
	ResponseSchema  func(OperationContract) (JSONSchema, bool)
	ExtraParameters func(OperationContract) []OpenAPIParameter
	AmendResponses  func(OperationContract, map[string]OpenAPIResponse)
}

func extensionAmendResponses(contract OperationContract, responses map[string]OpenAPIResponse) {
	for _, extension := range openAPIExtensions {
		if extension.AmendResponses != nil {
			extension.AmendResponses(contract, responses)
		}
	}
}

var openAPIExtensions []openAPIExtension

func registerOpenAPIExtension(extension openAPIExtension) {
	if extension.Name == "" || extension.Schemas == nil {
		panic("Management OpenAPI extension requires a name and schemas")
	}
	for _, existing := range openAPIExtensions {
		if existing.Name == extension.Name {
			panic(fmt.Sprintf("duplicate Management OpenAPI extension %q", extension.Name))
		}
	}
	openAPIExtensions = append(openAPIExtensions, extension)
}

func mergeOpenAPIExtensionSchemas(base map[string]JSONSchema) map[string]JSONSchema {
	for _, extension := range openAPIExtensions {
		for name, schema := range extension.Schemas() {
			if _, duplicate := base[name]; duplicate {
				panic(fmt.Sprintf("duplicate Management OpenAPI schema %q", name))
			}
			base[name] = schema
		}
	}
	return base
}

func extensionRequestSchema(contract OperationContract) (string, bool) {
	for _, extension := range openAPIExtensions {
		if extension.RequestSchema == nil {
			continue
		}
		if schema, found := extension.RequestSchema(contract); found {
			return schema, true
		}
	}
	return "", false
}

func extensionResponseSchema(contract OperationContract) (JSONSchema, bool) {
	for _, extension := range openAPIExtensions {
		if extension.ResponseSchema == nil {
			continue
		}
		if schema, found := extension.ResponseSchema(contract); found {
			return schema, true
		}
	}
	return JSONSchema{}, false
}

func extensionParameters(contract OperationContract) []OpenAPIParameter {
	var parameters []OpenAPIParameter
	for _, extension := range openAPIExtensions {
		if extension.ExtraParameters != nil {
			parameters = append(parameters, extension.ExtraParameters(contract)...)
		}
	}
	return parameters
}
