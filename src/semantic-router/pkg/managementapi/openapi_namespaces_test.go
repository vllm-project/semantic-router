package managementapi

import (
	"strings"
	"testing"
)

func TestNamespaceOperationsUseRegisteredExactAuthorizationOperands(t *testing.T) {
	path := BasePath + "/namespaces/{namespaceId}/self-service-policy"
	contract, found := LookupOperation(MethodPATCH, path)
	if !found {
		t.Fatalf("PATCH %s is not registered", path)
	}
	if err := contract.Permission.Validate(); err != nil {
		t.Fatalf("Namespace self-service authorization contract is invalid: %v", err)
	}
	canonical := contract.Permission.Canonical()
	for _, fragment := range []string{
		"namespace.manage@path_namespace",
		"WHEN(current_access_policy_default_present,access_policy.manage@current_access_policy_default)",
		"WHEN(current_rate_policy_default_present,rate_policy.manage@current_rate_policy_default)",
		"WHEN(target_access_policy_default_present,access_policy.manage@target_access_policy_default)",
		"WHEN(target_rate_policy_default_present,rate_policy.manage@target_rate_policy_default)",
	} {
		if !strings.Contains(canonical, fragment) {
			t.Fatalf("authorization contract %q omits %q", canonical, fragment)
		}
	}
}

func TestNamespaceOpenAPIHasTypedLifecycleAndCASContracts(t *testing.T) {
	document := GenerateOpenAPI()
	base := BasePath + "/namespaces"

	list := document.Paths[base]["get"]
	for _, parameter := range []struct {
		name     string
		location string
	}{
		{"status", "query"},
		{"cursor", "query"},
		{"pageSize", "query"},
	} {
		if !openAPIHasParameter(list.Parameters, parameter.name, parameter.location) {
			t.Fatalf("Namespace list omitted %s %s", parameter.location, parameter.name)
		}
	}

	create := document.Paths[base]["post"]
	if create.RequestBody == nil ||
		create.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/NamespaceCreateRequest" ||
		!openAPIHasParameter(create.Parameters, HeaderIdempotencyKey, "header") {
		t.Fatalf("Namespace create contract = %#v", create)
	}

	resource := base + "/{namespaceId}"
	for _, method := range []string{"patch", "delete"} {
		operation := document.Paths[resource][method]
		if !openAPIHasParameter(operation.Parameters, HeaderIfMatch, "header") {
			t.Fatalf("Namespace %s omitted If-Match", method)
		}
		if openAPIHasParameter(operation.Parameters, HeaderNamespaceID, "header") {
			t.Fatalf("Namespace %s redundantly accepts %s", method, HeaderNamespaceID)
		}
	}

	for _, suffix := range []string{
		"/self-service-policy",
		"/management-security-policy",
		"/routing-claim-schema",
	} {
		path := resource + suffix
		patch := document.Paths[path]["patch"]
		if !openAPIHasParameter(patch.Parameters, HeaderIfMatch, "header") {
			t.Fatalf("PATCH %s omitted If-Match", path)
		}
	}

	for _, schemaName := range []string{
		"Namespace",
		"NamespacePage",
		"NamespaceCreateRequest",
		"SelfServicePolicy",
		"NamespaceManagementSecurityPolicy",
		"RoutingClaimSchema",
	} {
		if _, found := document.Components.Schemas[schemaName]; !found {
			t.Fatalf("OpenAPI omitted schema %s", schemaName)
		}
	}
}
