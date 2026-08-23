package managementapi

import "testing"

func TestOperationContractsUseConjunctiveOriginAuthorizationAndCAS(t *testing.T) {
	detail, found := LookupOperation(MethodGET, BasePath+"/operations/{operationId}")
	if !found || detail.Revision != RevisionReturns || detail.Permission.Operator != PermissionAll ||
		len(detail.Permission.Operands) != 2 || detail.Permission.Operands[0].Operator != PermissionAll {
		t.Fatalf("operation detail contract = %#v, found=%t", detail, found)
	}
	cancel, found := LookupOperation(MethodPOST, BasePath+"/operations/{operationId}:cancel")
	if !found || cancel.Revision != RevisionCAS || cancel.Idempotency != IdempotencyRequired ||
		cancel.Permission.Operator != PermissionAll || cancel.Permission.Operands[0].Operator != PermissionAll {
		t.Fatalf("operation cancel contract = %#v, found=%t", cancel, found)
	}
}

func TestOperationOpenAPIHasTypedPageFiltersNamespaceAndETags(t *testing.T) {
	document := GenerateOpenAPI()
	page, found := document.Components.Schemas["OperationPage"]
	if !found || page.Properties["data"].Items == nil ||
		page.Properties["data"].Items.Ref != "#/components/schemas/Operation" {
		t.Fatalf("OperationPage = %#v, found=%t", page, found)
	}
	list := document.Paths[BasePath+"/operations"]["get"]
	parameters := make(map[string]OpenAPIParameter, len(list.Parameters))
	for _, parameter := range list.Parameters {
		parameters[parameter.Name] = parameter
	}
	for _, name := range []string{"cursor", "pageSize", "kind", "state", "originPrincipalId", HeaderNamespaceID} {
		if _, found := parameters[name]; !found {
			t.Errorf("operation list is missing %s", name)
		}
	}
	if list.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/OperationPage" {
		t.Fatalf("operation list response = %#v", list.Responses["200"])
	}
	detail := document.Paths[BasePath+"/operations/{operationId}"]["get"]
	if _, found := detail.Responses["200"].Headers[HeaderETag]; !found {
		t.Fatalf("operation detail response has no ETag: %#v", detail.Responses["200"])
	}
	cancel := document.Paths[BasePath+"/operations/{operationId}:cancel"]["post"]
	if parameters := parameterNames(cancel.Parameters); !parameters[HeaderIfMatch] || !parameters[HeaderIdempotencyKey] {
		t.Fatalf("operation cancel parameters = %#v", cancel.Parameters)
	}
}

func parameterNames(parameters []OpenAPIParameter) map[string]bool {
	result := make(map[string]bool, len(parameters))
	for _, parameter := range parameters {
		result[parameter.Name] = true
	}
	return result
}
