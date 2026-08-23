package managementapi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
)

func TestRegistryIsValidAndOperationIDsAreUnique(t *testing.T) {
	operations := Operations()
	if len(operations) == 0 {
		t.Fatal("Operations() returned an empty registry")
	}
	if err := ValidateRegistry(operations); err != nil {
		t.Fatalf("ValidateRegistry() error = %v", err)
	}

	ids := make(map[string]string, len(operations))
	for _, operation := range operations {
		route := string(operation.Method) + " " + operation.Path
		if previous, found := ids[operation.OperationID]; found {
			t.Fatalf("operation ID %q is shared by %s and %s", operation.OperationID, previous, route)
		}
		ids[operation.OperationID] = route
	}
}

func TestRegistryReturnsDefensiveCopies(t *testing.T) {
	first := Operations()
	if len(first) == 0 || len(first[0].Permission.Operands) == 0 {
		// Locate a compound expression rather than relying on sort order.
		for i := range first {
			if len(first[i].Permission.Operands) > 0 {
				first[0], first[i] = first[i], first[0]
				break
			}
		}
	}
	if len(first[0].Permission.Operands) == 0 {
		t.Fatal("registry contains no compound permission expression")
	}
	original := first[0].Permission.Operands[0].Canonical()
	first[0].Permission.Operands[0] = Require("cluster.manage", "cluster")
	second, found := LookupOperation(first[0].Method, first[0].Path)
	if !found {
		t.Fatalf("LookupOperation(%s, %s) failed", first[0].Method, first[0].Path)
	}
	if second.Permission.Operands[0].Canonical() != original {
		t.Fatal("caller mutation escaped into the canonical operation registry")
	}
}

func TestRegistryMatchesNormativeEndpointInventory(t *testing.T) {
	const (
		coreManagementEndpointCount = 192
		agentEndpointCount          = 36
	)
	documentPath := filepath.Join("..", "..", "..", "..", "website", "docs", "proposals", "router-native-access-control-management-api.md")
	contents, err := os.ReadFile(documentPath)
	if err != nil {
		t.Fatalf("read endpoint inventory: %v", err)
	}

	documented := extractDocumentedOperations(string(contents))
	registered := make([]string, 0, len(Operations()))
	for _, operation := range Operations() {
		registered = append(registered, string(operation.Method)+" "+operation.Path)
	}
	sort.Strings(registered)

	documentedAgentCount := 0
	for _, endpoint := range documented {
		if strings.Contains(endpoint, " /management/v1/agent-") ||
			strings.HasSuffix(endpoint, " /management/v1/publication-plans/{plan}:commit") {
			documentedAgentCount++
		}
	}
	if documentedAgentCount != agentEndpointCount || len(documented)-documentedAgentCount != coreManagementEndpointCount {
		t.Fatalf("normative document contains %d core and %d Agent endpoints, want %d and %d",
			len(documented)-documentedAgentCount, documentedAgentCount,
			coreManagementEndpointCount, agentEndpointCount)
	}
	if !reflect.DeepEqual(registered, documented) {
		t.Fatalf("registry does not match normative endpoint inventory\nmissing from registry: %v\nextra in registry: %v", difference(documented, registered), difference(registered, documented))
	}
}

func TestCrossFamilyPermissionExpressions(t *testing.T) {
	tests := []struct {
		method HTTPMethod
		path   string
		want   []string
	}{
		{MethodGET, BasePath + "/users/{userId}/effective-policy", []string{"user.read@user", "access_policy.read@user", "rate_policy.read@user"}},
		{MethodGET, BasePath + "/api-keys/{keyId}/quota", []string{"key.read@key", "quota.read@all_returned_bindings"}},
		{MethodPOST, BasePath + "/providers/{providerId}:discover-models", []string{"provider_catalog.read@request_namespace", "provider_credential.read@credential", "provider_credential.use@credential", "routing.manage@request_namespace"}},
		{MethodPOST, BasePath + "/routing/entrypoints/{entrypointId}:resolve", []string{"routing.read@target", "routing.read@all_dependencies", "routing_context.read@subject", "routing_context.manage@subject"}},
		{MethodPOST, BasePath + "/operations/{operationId}:cancel", []string{"self.manage@intrinsic_self", "operation.manage@operation_targets", "RECORDED(original_domain_mutation)"}},
	}

	for _, test := range tests {
		operation, found := LookupOperation(test.method, test.path)
		if !found {
			t.Fatalf("missing operation %s %s", test.method, test.path)
		}
		canonical := operation.Permission.Canonical()
		if strings.Contains(canonical, " ALL ") || strings.Contains(canonical, " ANY ") {
			t.Errorf("permission canonical form is not boolean DSL: %q", canonical)
		}
		for _, fragment := range test.want {
			if !strings.Contains(canonical, fragment) {
				t.Errorf("%s %s permission %q does not contain %q", test.method, test.path, canonical, fragment)
			}
		}
	}
}

func TestProviderDiscoveryPermissionBranchesAreConjunctive(t *testing.T) {
	operation, found := LookupOperation(MethodPOST, BasePath+"/providers/{providerId}:discover-models")
	if !found {
		t.Fatal("provider discovery operation is not registered")
	}
	canonical := operation.Permission.Canonical()
	if strings.Contains(canonical, " OR ") {
		t.Fatalf("credential and credential-free discovery branches must not be alternatives: %s", canonical)
	}
	for _, condition := range []string{"provider_credential_supplied", "no_provider_credential_supplied"} {
		if !strings.Contains(canonical, "WHEN("+condition+",") {
			t.Fatalf("provider discovery permission omits %q: %s", condition, canonical)
		}
	}
	if operation.Idempotency != IdempotencyNone || operation.Revision != RevisionNone {
		t.Fatalf("provider discovery is a stateless read and must not require mutation metadata: %+v", operation)
	}
}

func TestCanonicalPaginationAndStringQuantities(t *testing.T) {
	page := Page[string]{Data: []string{"one"}, Page: PageInfo{NextCursor: "opaque", HasMore: true, PageSize: 1}}
	encoded, err := json.Marshal(page)
	if err != nil {
		t.Fatalf("marshal page: %v", err)
	}
	if strings.Contains(string(encoded), "offset") || !strings.Contains(string(encoded), `"nextCursor":"opaque"`) {
		t.Fatalf("page is not a keyset envelope: %s", encoded)
	}

	validWhole := []string{"0", "1", "999999999999999999999999999999999999999999"}
	for _, value := range validWhole {
		if _, err := ParseWholeQuantity(value); err != nil {
			t.Errorf("ParseWholeQuantity(%q) error = %v", value, err)
		}
	}
	invalidWhole := []string{"", "-1", "+1", "01", "1.0", "1e3"}
	for _, value := range invalidWhole {
		if _, err := ParseWholeQuantity(value); err == nil {
			t.Errorf("ParseWholeQuantity(%q) unexpectedly succeeded", value)
		}
	}

	validCurrency := []string{"0", "0.0", "12.000000000000001", "999999999999999999"}
	for _, value := range validCurrency {
		if _, err := ParseCurrencyDecimal(value); err != nil {
			t.Errorf("ParseCurrencyDecimal(%q) error = %v", value, err)
		}
	}
	invalidCurrency := []string{"", "-0.1", ".1", "01", "1e3", "1.0000000000000001"}
	for _, value := range invalidCurrency {
		if _, err := ParseCurrencyDecimal(value); err == nil {
			t.Errorf("ParseCurrencyDecimal(%q) unexpectedly succeeded", value)
		}
	}

	assertNoFloatKinds(t, reflect.TypeOf(QuotaMeter{}), make(map[reflect.Type]bool))
	assertNoFloatKinds(t, reflect.TypeOf(CostSummary{}), make(map[reflect.Type]bool))
	assertNoFloatKinds(t, reflect.TypeOf(EffectivePolicy{}), make(map[reflect.Type]bool))
	assertNoFloatKinds(t, reflect.TypeOf(OperationProgress{}), make(map[reflect.Type]bool))
}

func TestQuantitiesRejectNumericJSONAndInvalidMarshalValues(t *testing.T) {
	var quantity WholeQuantity
	if err := json.Unmarshal([]byte(`12`), &quantity); err == nil {
		t.Fatal("numeric JSON unexpectedly decoded as WholeQuantity")
	}
	if err := json.Unmarshal([]byte(`"12"`), &quantity); err != nil || quantity != "12" {
		t.Fatalf("string WholeQuantity decode = %q, %v", quantity, err)
	}
	if _, err := json.Marshal(WholeQuantity("01")); err == nil {
		t.Fatal("invalid WholeQuantity unexpectedly marshaled")
	}
	if _, err := json.Marshal(CurrencyDecimal("1e3")); err == nil {
		t.Fatal("exponent CurrencyDecimal unexpectedly marshaled")
	}
}

func TestIdempotencyKeysAreOpaqueAndBounded(t *testing.T) {
	if _, err := ParseIdempotencyKey("0123456789abcdef"); err != nil {
		t.Fatalf("valid idempotency key rejected: %v", err)
	}
	for _, value := range []string{"short", "contains whitespace", strings.Repeat("x", MaximumIdempotencyKeyLength+1)} {
		if _, err := ParseIdempotencyKey(value); err == nil {
			t.Errorf("invalid idempotency key %q unexpectedly accepted", value)
		}
	}
}

func TestQuotaAndCostSemanticValidation(t *testing.T) {
	wholeRemaining := DecimalQuantity("10")
	wholeMeter := QuotaMeter{Metric: "requests", Algorithm: "sliding_log", Accounting: "request", Enforcement: "enforce", Limit: "12", Used: "2", Remaining: &wholeRemaining, Completeness: "complete", KnownDispatches: "2", IncompleteDispatches: "0", CapacityState: "available"}
	if err := wholeMeter.Validate(); err != nil {
		t.Fatalf("whole quota meter validation error = %v", err)
	}
	wholeMeter.Used = "2.5"
	if err := wholeMeter.Validate(); err == nil {
		t.Fatal("fractional request quota unexpectedly validated")
	}
	costRemaining := DecimalQuantity("10.25")
	costMeter := QuotaMeter{Metric: "cost", Algorithm: "calendar_window", Accounting: "response_actual", Enforcement: "enforce", Currency: "USD", Limit: "12.50", Used: "2.25", Remaining: &costRemaining, Completeness: "complete", KnownDispatches: "2", IncompleteDispatches: "0", CapacityState: "available"}
	if err := costMeter.Validate(); err != nil {
		t.Fatalf("cost quota meter validation error = %v", err)
	}
	overage := DecimalQuantity("3")
	zeroRemaining := DecimalQuantity("0")
	overLimit := QuotaMeter{Metric: "requests", Algorithm: "sliding_log", Accounting: "request", Enforcement: "enforce", Limit: "12", Used: "15", Remaining: &zeroRemaining, Overage: &overage, Completeness: "complete", KnownDispatches: "15", IncompleteDispatches: "0", CapacityState: "over_limit"}
	if err := overLimit.Validate(); err != nil {
		t.Fatalf("over-limit quota validation error = %v", err)
	}
	partial := QuotaMeter{Metric: "total_tokens", Algorithm: "sliding_log", Accounting: "response_actual", Enforcement: "enforce", Limit: "100", Used: "12", Remaining: nil, Completeness: "partial", KnownDispatches: "1", IncompleteDispatches: "1", CapacityState: "fenced", ActiveFenceIDs: []string{"fence-1"}}
	if err := partial.Validate(); err != nil {
		t.Fatalf("partial quota validation error = %v", err)
	}
	partial.Remaining = &zeroRemaining
	if err := partial.Validate(); err == nil {
		t.Fatal("partial quota meter unexpectedly claimed remaining capacity")
	}

	summary := CostSummary{Currency: "USD", KnownAmount: "1.25", Completeness: CostPartial, KnownDispatches: "2", IncompleteDispatches: "1"}
	if err := summary.Validate(); err != nil {
		t.Fatalf("partial cost validation error = %v", err)
	}
	summary.Completeness = CostComplete
	if err := summary.Validate(); err == nil {
		t.Fatal("inconsistent complete cost unexpectedly validated")
	}

	costMeter.Limit = "1.0000000000000001"
	if _, err := json.Marshal(costMeter); err == nil {
		t.Fatal("over-precision cost quota unexpectedly marshaled")
	}
}

func TestOrdinaryWireTypesHaveNoSecretBearingFields(t *testing.T) {
	ordinary := []reflect.Type{
		reflect.TypeOf(ErrorResponse{}),
		reflect.TypeOf(PageInfo{}),
		reflect.TypeOf(IdempotencyMetadata{}),
		reflect.TypeOf(RevisionState{}),
		reflect.TypeOf(Operation{}),
		reflect.TypeOf(EffectivePolicy{}),
		reflect.TypeOf(CostSummary{}),
	}
	for _, wireType := range ordinary {
		assertNoSecretFields(t, wireType, make(map[reflect.Type]bool))
	}
}

var documentedOperationPattern = regexp.MustCompile(`(?m)^\s*(GET|POST|PUT|PATCH|DELETE)\s+(/management/v1/\S+)`)

func extractDocumentedOperations(document string) []string {
	unique := make(map[string]bool)
	for _, match := range documentedOperationPattern.FindAllStringSubmatch(document, -1) {
		unique[match[1]+" "+match[2]] = true
	}
	result := make([]string, 0, len(unique))
	for operation := range unique {
		result = append(result, operation)
	}
	sort.Strings(result)
	return result
}

func difference(left, right []string) []string {
	rightSet := make(map[string]bool, len(right))
	for _, value := range right {
		rightSet[value] = true
	}
	var result []string
	for _, value := range left {
		if !rightSet[value] {
			result = append(result, value)
		}
	}
	return result
}

func assertNoFloatKinds(t *testing.T, wireType reflect.Type, seen map[reflect.Type]bool) {
	t.Helper()
	for wireType.Kind() == reflect.Pointer || wireType.Kind() == reflect.Slice || wireType.Kind() == reflect.Array {
		wireType = wireType.Elem()
	}
	if seen[wireType] {
		return
	}
	seen[wireType] = true
	if wireType.Kind() == reflect.Float32 || wireType.Kind() == reflect.Float64 {
		t.Errorf("wire type %s contains floating-point quantity", wireType)
		return
	}
	if wireType.Kind() != reflect.Struct {
		return
	}
	for i := 0; i < wireType.NumField(); i++ {
		assertNoFloatKinds(t, wireType.Field(i).Type, seen)
	}
}

func assertNoSecretFields(t *testing.T, wireType reflect.Type, seen map[reflect.Type]bool) {
	t.Helper()
	for wireType.Kind() == reflect.Pointer || wireType.Kind() == reflect.Slice || wireType.Kind() == reflect.Array {
		wireType = wireType.Elem()
	}
	if seen[wireType] || wireType.PkgPath() != "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi" {
		return
	}
	seen[wireType] = true
	if wireType.Kind() != reflect.Struct {
		return
	}
	for i := 0; i < wireType.NumField(); i++ {
		field := wireType.Field(i)
		jsonName := strings.Split(field.Tag.Get("json"), ",")[0]
		lower := strings.ToLower(jsonName)
		for _, forbidden := range []string{"secret", "token", "credential", "claim"} {
			if strings.Contains(lower, forbidden) {
				t.Errorf("ordinary wire type %s contains secret-bearing JSON field %q", wireType, jsonName)
			}
		}
		assertNoSecretFields(t, field.Type, seen)
	}
}
