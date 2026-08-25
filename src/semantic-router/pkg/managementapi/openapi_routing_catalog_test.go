package managementapi

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

func TestRoutingCatalogOpenAPIIsKeyScopedAndCannotExposeCredentialsOrRecipeSource(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/api-keys/{keyId}/routing-catalog"]["get"]
	response := operation.Responses["200"]
	if response.Content[JSONMediaType].Schema.Ref != "#/components/schemas/RoutingCatalog" {
		t.Fatalf("routing catalog response schema = %#v", response)
	}
	model := document.Components.Schemas["RoutingCatalogModel"]
	if _, exposed := model.Properties["backends"]; exposed {
		t.Fatal("consumer routing catalog exposes Provider backends")
	}
	recipe := document.Components.Schemas["RoutingCatalogRecipe"]
	if _, exposed := recipe.Properties["document"]; exposed {
		t.Fatal("consumer routing catalog exposes Recipe source document")
	}
	contract, found := LookupOperation(MethodGET, BasePath+"/api-keys/{keyId}/routing-catalog")
	if !found || contract.Permission.Canonical() !=
		"(key.read@key AND access_policy.read@key AND routing_context.read@key)" {
		t.Fatalf("routing catalog permission contract = %#v", contract.Permission)
	}
}

func TestRoutingCatalogAllowsConsumerOnlyForAnOwnedKey(t *testing.T) {
	namespaceID := accesscontrol.NamespaceID("namespace-1")
	userID := accesscontrol.UserID("user-1")
	role, found := accesscontrol.BuiltInRole(accesscontrol.BuiltInRoleConsumer)
	if !found {
		t.Fatal("consumer role is unavailable")
	}
	binding := accesscontrol.ManagementRoleBinding{
		ID: "binding-1", PrincipalID: "principal-1", RoleID: role.ID,
		Scope:  accesscontrol.UserScope(namespaceID, userID),
		Status: accesscontrol.BindingStatusActive, Revision: 1,
	}
	contract, found := LookupOperation(MethodGET, BasePath+"/api-keys/{keyId}/routing-catalog")
	if !found {
		t.Fatal("routing catalog operation is unavailable")
	}
	keyTarget := func(keyID string, owner accesscontrol.UserID) accesscontrol.ScopedTarget {
		return accesscontrol.ScopedTarget{
			Scope:     accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceAPIKey, accesscontrol.ResourceID(keyID)),
			Ancestors: []accesscontrol.Scope{accesscontrol.UserScope(namespaceID, owner)},
		}
	}
	context := managementauthorization.EvaluationContext{
		Authenticated: true,
		RoleGrants:    []managementauthorization.RoleGrant{{Binding: binding, Role: role}},
		Targets: map[string][]accesscontrol.ScopedTarget{
			"key": {keyTarget("key-owned", userID)},
		},
	}
	if err := managementauthorization.Evaluate(contract.Permission, context); err != nil {
		t.Fatalf("consumer-owned key authorization error = %v", err)
	}
	context.Targets["key"] = []accesscontrol.ScopedTarget{keyTarget("key-other", "user-2")}
	if err := managementauthorization.Evaluate(contract.Permission, context); !errors.Is(err, managementauthorization.ErrDenied) {
		t.Fatalf("cross-user key authorization error = %v, want ErrDenied", err)
	}
}
