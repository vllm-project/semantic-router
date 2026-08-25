package managementapi

import "testing"

func TestPolicyOperationContractsRequireCASAndTypedAsyncBulk(t *testing.T) {
	casPaths := []string{
		BasePath + "/access-policies/{policyId}",
		BasePath + "/rate-limit-policies/{policyId}",
		BasePath + "/access-policy-bindings/{bindingId}",
		BasePath + "/rate-limit-bindings/{bindingId}",
	}
	for _, path := range casPaths {
		contract, found := LookupOperation(MethodDELETE, path)
		if !found || contract.Revision != RevisionCAS {
			t.Errorf("DELETE %s contract = %#v, found=%t", path, contract, found)
		}
	}
	for _, path := range []string{
		BasePath + "/access-policy-bindings:bulk-apply",
		BasePath + "/rate-limit-bindings:bulk-apply",
	} {
		contract, found := LookupOperation(MethodPOST, path)
		if !found || contract.Async != AsyncOperation || contract.Idempotency != IdempotencyRequired ||
			contract.Revision != RevisionNone {
			t.Errorf("POST %s contract = %#v, found=%t", path, contract, found)
		}
	}
}

func TestPolicyOpenAPIUsesOneOfForExistingOrInlineRateBinding(t *testing.T) {
	document := GenerateOpenAPI()
	schema, found := document.Components.Schemas["RateLimitBindingCreateRequest"]
	if !found || len(schema.OneOf) != 2 {
		t.Fatalf("RateLimitBindingCreateRequest = %#v, found=%t", schema, found)
	}
	bulk, found := document.Components.Schemas["RateLimitBindingBulkApplyRequest"]
	if !found || bulk.Properties["items"].MinItems == nil || *bulk.Properties["items"].MinItems != 1 ||
		bulk.Properties["items"].MaxItems == nil || *bulk.Properties["items"].MaxItems != 1000 {
		t.Fatalf("RateLimitBindingBulkApplyRequest = %#v, found=%t", bulk, found)
	}
	rules := document.Components.Schemas["RateLimitRuleInput"]
	if len(rules.OneOf) != 9 {
		t.Fatalf("RateLimitRuleInput variants = %d, want 9", len(rules.OneOf))
	}
	for _, variant := range rules.OneOf {
		algorithm := variant.Properties["algorithm"]
		if len(algorithm.Enum) != 1 || variant.AdditionalProperties == nil || *variant.AdditionalProperties {
			t.Fatalf("RateLimitRuleInput variant is not closed and discriminated: %#v", variant)
		}
		if window, found := variant.Properties["window"]; found &&
			(window.Type != "string" || window.Pattern != canonicalISODurationPattern) {
			t.Fatalf("RateLimitRuleInput.window = %#v", window)
		}
	}
	inline := document.Components.Schemas["APIKeyInlineRateLimitPolicy"].Properties["rules"].Items
	if inline == nil || len(inline.OneOf) != len(rules.OneOf) {
		t.Fatalf("API-key inline rules do not share the policy union: %#v", inline)
	}
}
