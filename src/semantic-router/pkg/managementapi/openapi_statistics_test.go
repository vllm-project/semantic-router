package managementapi

import "testing"

func TestStatisticsContractUsesExactOptionalCounts(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/statistics"]["get"]
	if operation.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/AccessStatistics" {
		t.Fatal("statistics response schema is not published")
	}
	if operation.RouterPermissionCanonical != "usage.read@all_returned_resources" {
		t.Fatalf("statistics entry permission = %q", operation.RouterPermissionCanonical)
	}
	schema := document.Components.Schemas["AccessStatistics"]
	for _, name := range []string{
		"users", "teams", "activeApiKeys", "expiringApiKeys", "accessPolicies", "activeRatePolicies",
	} {
		field := schema.Properties[name]
		if field.Type != "string" || field.Pattern == "" {
			t.Errorf("AccessStatistics.%s = %#v, want exact decimal string", name, field)
		}
		for _, required := range schema.Required {
			if required == name {
				t.Errorf("AccessStatistics.%s must remain permission-optional", name)
			}
		}
	}
}
