package managementapi

import "testing"

func TestPrincipalDirectoryOpenAPIIsNamespaceScopedAndTyped(t *testing.T) {
	document := GenerateOpenAPI()
	for _, name := range []string{
		"PrincipalDirectoryEntry", "PrincipalDirectoryDetail", "PrincipalDirectoryPage",
		"PrincipalUserLink", "PrincipalUserLinkDetail", "PrincipalUserLinkPage", "PrincipalUserLinkPutRequest",
	} {
		if _, found := document.Components.Schemas[name]; !found {
			t.Fatalf("schema %q is missing", name)
		}
	}
	directory := document.Paths[BasePath+"/namespaces/{namespaceId}/principal-directory"]["get"]
	if directory.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/PrincipalDirectoryPage" {
		t.Fatalf("directory response schema = %#v", directory.Responses["200"])
	}
	for _, parameter := range []string{"namespaceId", "cursor", "pageSize", "search"} {
		if !openAPIHasParameter(directory.Parameters, parameter, "") {
			t.Fatalf("directory operation omits %q", parameter)
		}
	}
	schema := document.Components.Schemas["PrincipalDirectoryEntry"]
	for _, forbidden := range []string{"issuer", "subject", "attributes", "sessions", "roleBindings"} {
		if _, found := schema.Properties[forbidden]; found {
			t.Fatalf("directory schema exposes %q", forbidden)
		}
	}
}

func TestPrincipalUserLinkOpenAPIModelsCreateOrCASReplace(t *testing.T) {
	document := GenerateOpenAPI()
	path := BasePath + "/namespaces/{namespaceId}/principal-user-links/{principalId}"
	put := document.Paths[path]["put"]
	if put.RequestBody == nil ||
		put.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/PrincipalUserLinkPutRequest" ||
		put.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/PrincipalUserLinkDetail" {
		t.Fatalf("principal link PUT contract = %#v", put)
	}
	if !openAPIHasParameter(put.Parameters, HeaderIdempotencyKey, "header") {
		t.Fatal("principal link PUT omits required Idempotency-Key")
	}
	if parameter, found := openAPIParameter(put.Parameters, HeaderIfMatch, "header"); !found || parameter.Required {
		t.Fatalf("principal link PUT conditional If-Match = %#v, %v", parameter, found)
	}
	remove := document.Paths[path]["delete"]
	if parameter, found := openAPIParameter(remove.Parameters, HeaderIfMatch, "header"); !found || !parameter.Required {
		t.Fatalf("principal link DELETE If-Match = %#v, %v", parameter, found)
	}
}

func openAPIHasParameter(parameters []OpenAPIParameter, name, location string) bool {
	_, found := openAPIParameter(parameters, name, location)
	return found
}

func openAPIParameter(parameters []OpenAPIParameter, name, location string) (OpenAPIParameter, bool) {
	for _, parameter := range parameters {
		if parameter.Name == name && (location == "" || parameter.In == location) {
			return parameter, true
		}
	}
	return OpenAPIParameter{}, false
}
