package managementapi

import "testing"

func TestAccessReadOpenAPIUsesTypedSchemasAndRoutingContextCAS(t *testing.T) {
	document := GenerateOpenAPI()
	for _, name := range []string{"EffectivePolicy", "EffectiveQuota", "RoutingContext", "AccessCheckRequest", "AccessCheckResponse"} {
		if _, found := document.Components.Schemas[name]; !found {
			t.Fatalf("%s schema is missing", name)
		}
	}
	claim := document.Components.Schemas["RoutingClaimValue"]
	if len(claim.OneOf) != 3 {
		t.Fatalf("RoutingClaimValue schema = %#v", claim)
	}
	check := document.Components.Schemas["AccessCheckRequest"]
	if _, found := check.Properties["credential"]; found {
		t.Fatal("AccessCheckRequest must not represent raw credentials")
	}
	for _, path := range []string{
		BasePath + "/users/{userId}/routing-context",
		BasePath + "/teams/{teamId}/routing-context",
		BasePath + "/api-keys/{keyId}/routing-context",
	} {
		contract, found := LookupOperation(MethodPUT, path)
		if !found || contract.Revision != RevisionCAS {
			t.Fatalf("PUT %s contract=%#v found=%t", path, contract, found)
		}
	}
}
