package managementapi

import "testing"

func TestOpenAPIRelationshipCollectionsOfferOptInAuthoritativeTotals(t *testing.T) {
	document := GenerateOpenAPI()
	for _, path := range []string{
		BasePath + "/users/{userId}/memberships",
		BasePath + "/teams/{teamId}/members",
		BasePath + "/api-keys",
		BasePath + "/access-policy-bindings",
		BasePath + "/rate-limit-bindings",
	} {
		operation := document.Paths[path]["get"]
		if !openAPIHasParameter(operation.Parameters, "includeTotal", "query") {
			t.Errorf("GET %s does not expose includeTotal", path)
		}
	}
	if openAPIHasParameter(document.Paths[BasePath+"/users"]["get"].Parameters, "includeTotal", "query") {
		t.Fatal("ordinary User collection unexpectedly pays for an exact total")
	}
	page := document.Components.Schemas["PageInfo"]
	if page.Properties["totalCount"].Type != "string" {
		t.Fatalf("PageInfo.totalCount = %#v", page.Properties["totalCount"])
	}
	member := document.Components.Schemas["TeamMember"]
	if member.Properties["email"].Format != "email" {
		t.Fatalf("TeamMember.email = %#v", member.Properties["email"])
	}
}
