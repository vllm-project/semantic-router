package managementapi

import (
	"strings"
	"testing"
)

func TestAPIKeyCreateOpenAPIExposesAtomicPolicyOverrides(t *testing.T) {
	document := GenerateOpenAPI()
	request, found := document.Components.Schemas["APIKeyCreateRequest"]
	if !found {
		t.Fatal("APIKeyCreateRequest schema is missing")
	}
	access := request.Properties["accessPolicyIds"]
	if access.Type != "array" || access.Items == nil || access.Items.Format != "uuid" ||
		access.MaxItems == nil || *access.MaxItems != 12 {
		t.Fatalf("accessPolicyIds schema = %#v", access)
	}
	override := request.Properties["rateLimitOverride"]
	if len(override.OneOf) != 2 || len(override.OneOf[0].Required) != 1 ||
		override.OneOf[0].Required[0] != "policyId" || len(override.OneOf[1].Required) != 1 ||
		override.OneOf[1].Required[0] != "inlinePolicy" {
		t.Fatalf("rateLimitOverride schema = %#v", override)
	}
	issued := document.Components.Schemas["APIKeyIssuedSecret"]
	if _, found := issued.Properties["accessPolicyBindings"]; !found {
		t.Fatal("issued secret omits access-policy binding receipts")
	}
	if _, found := issued.Properties["rateLimitOverride"]; !found {
		t.Fatal("issued secret omits rate-limit override receipt")
	}
}

func TestAPIKeyCreateAuthorizationSeparatesPolicyKinds(t *testing.T) {
	contract, found := LookupOperation(MethodPOST, BasePath+"/api-keys")
	if !found {
		t.Fatal("API-key create contract is missing")
	}
	canonical := contract.Permission.Canonical()
	for _, expected := range []string{
		"access_policy_binding_requested",
		"rate_policy_binding_requested",
		"inline_rate_policy_requested",
		"access_policy.manage",
		"rate_policy.manage",
	} {
		if !strings.Contains(canonical, expected) {
			t.Fatalf("API-key create permission %q omits %q", canonical, expected)
		}
	}
	if strings.Contains(canonical, "explicit_policy_binding_requested") {
		t.Fatalf("API-key create retained ambiguous policy condition: %q", canonical)
	}
}
