package managementapi

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

func TestProviderCatalogPublicationContractsAreTypedClusterOperations(t *testing.T) {
	document := GenerateOpenAPI()
	tests := []struct {
		path    string
		request string
	}{
		{BasePath + "/provider-catalog:bootstrap", "ProviderCatalogBootstrapRequest"},
		{BasePath + "/provider-catalog:activate", "ProviderCatalogActivateRequest"},
	}
	for _, test := range tests {
		operation, found := LookupOperation(MethodPOST, test.path)
		if !found {
			t.Fatalf("missing operation POST %s", test.path)
		}
		if operation.Scope != ScopeCluster || operation.Permission.Canonical() != "cluster.manage@cluster" ||
			operation.Idempotency != IdempotencyNone || operation.Revision != RevisionNone {
			t.Errorf("POST %s contract = %+v", test.path, operation)
		}
		openAPI := document.Paths[test.path]["post"]
		if openAPI.RequestBody == nil ||
			openAPI.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/"+test.request ||
			openAPI.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/ProviderCatalogPublication" {
			t.Errorf("POST %s OpenAPI = %+v", test.path, openAPI)
		}
		for _, parameter := range openAPI.Parameters {
			if parameter.Name == HeaderNamespaceID {
				t.Errorf("cluster operation POST %s requires a namespace", test.path)
			}
		}
	}
	for _, schema := range []string{
		"ProviderCatalogBootstrapRequest", "ProviderCatalogActivateRequest", "ProviderCatalogPublication",
	} {
		if _, found := document.Components.Schemas[schema]; !found {
			t.Errorf("OpenAPI components omitted %s", schema)
		}
	}
}

func TestProviderCatalogPublicationGenerationUsesExactDecimalStrings(t *testing.T) {
	bootstrap := ProviderCatalogBootstrapRequest{ExpectedGeneration: "9223372036854775807"}
	if generation, err := bootstrap.Generation(); err != nil || generation != math.MaxInt64 {
		t.Fatalf("bootstrap generation = %d, %v", generation, err)
	}
	bootstrap.ExpectedGeneration = "9223372036854775808"
	if _, err := bootstrap.Generation(); err == nil {
		t.Fatal("generation larger than PostgreSQL BIGINT was accepted")
	}
	publication := ProviderCatalogPublication{
		DesiredRevision: "sha256:" + strings.Repeat("a", 64), Generation: WholeQuantity("9223372036854775807"),
		UpdatedAt: time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC),
	}
	encoded, err := json.Marshal(publication)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"generation":"9223372036854775807"`) ||
		strings.Contains(string(encoded), `"generation":9223372036854775807`) {
		t.Fatalf("publication generation lost string encoding: %s", encoded)
	}
}

func TestProviderCatalogOpenAPIHasTypedReadAndDiscoveryContracts(t *testing.T) {
	document := GenerateOpenAPI()
	list := document.Paths[BasePath+"/providers"]["get"]
	if got := list.Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/ProviderCatalogPage" {
		t.Fatalf("provider list response schema = %q", got)
	}
	parameterNames := make(map[string]bool)
	for _, parameter := range list.Parameters {
		parameterNames[parameter.Name] = true
	}
	for _, name := range []string{"cursor", "pageSize", "search", "category", "capability"} {
		if !parameterNames[name] {
			t.Errorf("provider list OpenAPI omitted %q", name)
		}
	}
	detail := document.Paths[BasePath+"/providers/{providerId}"]["get"]
	if got := detail.Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/ProviderCatalogDetail" {
		t.Fatalf("provider detail response schema = %q", got)
	}
	discover := document.Paths[BasePath+"/providers/{providerId}:discover-models"]["post"]
	if discover.RequestBody == nil ||
		discover.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/DiscoverModelsRequest" {
		t.Fatalf("discover request body = %#v", discover.RequestBody)
	}
	if got := discover.Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/DiscoverModelsPage" {
		t.Fatalf("discover response schema = %q", got)
	}
	for _, name := range []string{
		"ProviderCatalogIcon", "ProviderInterface", "ProviderCatalogItem", "ProviderCatalogPage", "ProviderCatalogDetail",
		"DiscoverModelsRequest", "DiscoveredModel", "DiscoverModelsPage",
	} {
		if _, found := document.Components.Schemas[name]; !found {
			t.Errorf("OpenAPI components omitted %s", name)
		}
	}
	item := document.Components.Schemas["ProviderCatalogItem"]
	if _, exposed := item.Properties["apiDialect"]; exposed {
		t.Fatal("ProviderCatalogItem OpenAPI exposed internal wire metadata")
	}
	if item.Properties["interfaces"].Items == nil ||
		item.Properties["interfaces"].Items.Ref != "#/components/schemas/ProviderInterface" {
		t.Fatal("ProviderCatalogItem OpenAPI omitted safe Provider interfaces")
	}
}
