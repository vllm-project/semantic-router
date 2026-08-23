package managementapi

import "testing"

func TestOpenAPICollectionSearchIsBoundedToPublicDirectories(t *testing.T) {
	document := GenerateOpenAPI()
	searchable := []string{
		BasePath + "/users",
		BasePath + "/teams",
		BasePath + "/api-keys",
		BasePath + "/access-policies",
		BasePath + "/rate-limit-policies",
		BasePath + "/agent-profiles",
		BasePath + "/agent-skills",
		BasePath + "/agent-tools",
		BasePath + "/agent-tool-credentials",
		BasePath + "/agent-tool-sources",
		BasePath + "/agent-sessions",
	}
	for _, path := range searchable {
		operation := document.Paths[path]["get"]
		var search *OpenAPIParameter
		for index := range operation.Parameters {
			if operation.Parameters[index].Name == "search" && operation.Parameters[index].In == "query" {
				search = &operation.Parameters[index]
				break
			}
		}
		if search == nil || search.Schema.MaxLength == nil || *search.Schema.MaxLength != 200 {
			t.Fatalf("%s search parameter = %#v", path, search)
		}
	}
	credentials := document.Paths[BasePath+"/api-keys/{keyId}/credentials"]["get"]
	for _, parameter := range credentials.Parameters {
		if parameter.Name == "search" {
			t.Fatal("credential collections must not expose search")
		}
	}
}
