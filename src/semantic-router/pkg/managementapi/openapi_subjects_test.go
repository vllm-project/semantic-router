package managementapi

import "testing"

func TestTeamCreateSchemaExposesAtomicPolicySelection(t *testing.T) {
	schema, found := GenerateOpenAPI().Components.Schemas["TeamCreateRequest"]
	if !found {
		t.Fatal("TeamCreateRequest schema is unavailable")
	}
	access, found := schema.Properties["accessPolicyIds"]
	if !found || access.Type != "array" || access.Items == nil || access.Items.Type != "string" ||
		access.Items.Format != "uuid" || access.MinItems == nil || *access.MinItems != 1 || !access.UniqueItems {
		t.Fatalf("accessPolicyIds schema = %#v", access)
	}
	rate, found := schema.Properties["rateLimitPolicyId"]
	if !found || rate.Type != "string" || rate.Format != "uuid" {
		t.Fatalf("rateLimitPolicyId schema = %#v", rate)
	}
}
