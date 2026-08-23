package managementapi

import "testing"

func TestWorkloadIdentityOpenAPIUsesExplicitCredentialOrMTLSAuthentication(t *testing.T) {
	document := GenerateOpenAPI()
	operation, found := document.Paths[BasePath+"/auth/service-token"]["post"]
	if !found {
		t.Fatal("service-token operation is missing")
	}
	if len(operation.Security) != 2 {
		t.Fatalf("service-token security = %#v", operation.Security)
	}
	_, credential := operation.Security[0]["serviceCredential"]
	_, mtls := operation.Security[1]["mutualTLS"]
	if !credential || !mtls {
		t.Fatalf("service-token security = %#v", operation.Security)
	}
	if document.Components.SecuritySchemes["mutualTLS"].Type != "mutualTLS" {
		t.Fatalf("mutualTLS security scheme = %#v", document.Components.SecuritySchemes["mutualTLS"])
	}
}

func TestWorkloadIdentityOpenAPIDoesNotExposeCredentialStorageFields(t *testing.T) {
	document := GenerateOpenAPI()
	credential := document.Components.Schemas["ServiceCredential"]
	for _, field := range []string{"secret", "secretHmac", "pepperVersion", "ciphertext", "nonce", "kekVersion"} {
		if _, found := credential.Properties[field]; found {
			t.Fatalf("ServiceCredential exposes storage field %q", field)
		}
	}
	issue := document.Components.Schemas["ServiceCredentialIssue"]
	if _, found := issue.Properties["secret"]; !found {
		t.Fatal("one-time credential issue response is missing secret")
	}
}
