package managementapi

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"

func collectionSearchParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET {
		return nil
	}
	switch contract.Path {
	case BasePath + "/users", BasePath + "/teams", BasePath + "/api-keys",
		BasePath + "/access-policies", BasePath + "/rate-limit-policies":
		return []OpenAPIParameter{{
			Name: "search", In: "query",
			Description: "Case-insensitive literal prefix matched only against public resource identity fields.",
			Schema:      JSONSchema{Type: "string", MaxLength: intPointer(managementsearch.MaximumRunes)},
		}}
	default:
		return nil
	}
}
