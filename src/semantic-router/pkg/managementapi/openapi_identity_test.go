package managementapi

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBootstrapOpenAPIIsIdempotentSecretAndRevisionFree(t *testing.T) {
	contract, found := LookupOperation(MethodPOST, BasePath+"/auth/bootstrap")
	if !found {
		t.Fatal("bootstrap operation is absent")
	}
	if contract.Idempotency != IdempotencyRequired || contract.Revision != RevisionNone ||
		contract.Secret.Input != SecretInputAuthorization || contract.Secret.Output != SecretOutputOneTime ||
		!contract.Secret.NoStore || contract.Secret.Authenticated {
		t.Fatalf("unexpected bootstrap contract: %+v", contract)
	}
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/auth/bootstrap"]["post"]
	if operation.RequestBody == nil ||
		operation.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/BootstrapRequest" {
		t.Fatalf("unexpected bootstrap request schema: %+v", operation.RequestBody)
	}
	response := operation.Responses["201"]
	if response.Content[JSONMediaType].Schema.Ref != "#/components/schemas/BootstrapResponse" {
		t.Fatalf("unexpected bootstrap response schema: %+v", response)
	}
	var idempotency bool
	for _, parameter := range operation.Parameters {
		if parameter.Name == HeaderIdempotencyKey && parameter.In == "header" && parameter.Required {
			idempotency = true
		}
	}
	if !idempotency {
		t.Fatal("bootstrap OpenAPI omits required Idempotency-Key")
	}
	request := document.Components.Schemas["BootstrapRequest"]
	responseSchema := document.Components.Schemas["BootstrapResponse"]
	if len(request.OneOf) != 2 || len(responseSchema.OneOf) != 2 {
		t.Fatal("bootstrap request/response schemas do not model both exact variants")
	}
}

func TestBootstrapOpenAPINeverExposesStoredSecretMetadata(t *testing.T) {
	document := GenerateOpenAPI()
	encoded, err := json.Marshal(map[string]JSONSchema{
		"request":    document.Components.Schemas["BootstrapRequest"],
		"response":   document.Components.Schemas["BootstrapResponse"],
		"credential": document.Components.Schemas["BootstrapServiceCredential"],
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"ciphertext", "nonce", "kekVersion", "secretHmac", "pepperVersion", "idempotencyKeyDigest", "requestDigest"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("bootstrap public schemas expose %q", forbidden)
		}
	}
}

func TestRecoveryOpenAPIIsNarrowAndDoesNotIssueCredentials(t *testing.T) {
	contract, found := LookupOperation(MethodPOST, BasePath+"/auth/recovery")
	if !found {
		t.Fatal("recovery operation is absent")
	}
	if contract.Idempotency != IdempotencyRequired || contract.Revision != RevisionNone ||
		contract.Secret.Input != SecretInputAuthorization || contract.Secret.Output != SecretOutputNone ||
		!contract.Secret.NoStore {
		t.Fatalf("unexpected recovery contract: %+v", contract)
	}
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/auth/recovery"]["post"]
	if operation.RequestBody == nil ||
		operation.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/RecoveryRequest" ||
		operation.Responses["201"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/RecoveryResponse" {
		t.Fatalf("recovery schemas = %+v / %+v", operation.RequestBody, operation.Responses["201"])
	}
	encoded, err := json.Marshal(map[string]JSONSchema{
		"request":  document.Components.Schemas["RecoveryRequest"],
		"response": document.Components.Schemas["RecoveryResponse"],
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"secret", "credential", "displayName", "createServiceAccount"} {
		if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
			t.Fatalf("recovery schema exposes unsupported field %q: %s", forbidden, encoded)
		}
	}
}

func TestTokenExchangeOpenAPIModelsExactInvitationVariant(t *testing.T) {
	document := GenerateOpenAPI()
	request := document.Components.Schemas["TokenExchangeRequest"]
	response := document.Components.Schemas["TokenExchangeResponse"]
	if len(request.OneOf) != 2 || len(response.OneOf) != 2 {
		t.Fatalf("token exchange oneOf = request:%d response:%d", len(request.OneOf), len(response.OneOf))
	}
	standard := document.Components.Schemas["StandardTokenExchangeRequest"]
	invited := document.Components.Schemas["InvitedTokenExchangeRequest"]
	if _, found := standard.Properties["invitationToken"]; found {
		t.Fatal("standard token exchange accepts invitationToken")
	}
	if _, found := invited.Properties["invitationToken"]; !found ||
		!containsString(invited.Required, "invitationToken") {
		t.Fatal("invited token exchange does not require invitationToken")
	}
	operation := document.Paths[BasePath+"/auth/token-exchange"]["post"]
	if operation.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/TokenExchangeRequest" ||
		operation.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/TokenExchangeResponse" {
		t.Fatalf("token exchange schemas = %#v / %#v", operation.RequestBody, operation.Responses["200"])
	}
}

func TestSelfDelegationOpenAPIUsesTypedNamespaceScopedContracts(t *testing.T) {
	document := GenerateOpenAPI()
	keys := document.Paths[BasePath+"/self/inference-keys"]["get"]
	if keys.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/EligibleInferenceKeyPage" {
		t.Fatalf("self key response = %+v", keys.Responses["200"])
	}
	if !openAPIHasParameter(keys.Parameters, "search", "query") {
		t.Fatalf("self key search parameter = %+v", keys.Parameters)
	}
	detail := document.Paths[BasePath+"/self/inference-keys/{keyId}"]["get"]
	if detail.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/EligibleInferenceKeyDetail" ||
		!openAPIHasParameter(detail.Parameters, HeaderNamespaceID, "header") {
		t.Fatalf("self key detail contract = %+v", detail)
	}
	create := document.Paths[BasePath+"/self/inference-sessions"]["post"]
	if create.RequestBody == nil ||
		create.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/DelegatedInferenceSessionCreateRequest" ||
		create.Responses["201"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/SecretEnvelope" {
		t.Fatalf("self delegation create contract = %+v / %+v", create.RequestBody, create.Responses["201"])
	}
	for _, operation := range []OpenAPIOperation{keys, create} {
		foundNamespace := false
		for _, parameter := range operation.Parameters {
			if parameter.Name == HeaderNamespaceID && parameter.In == "header" && parameter.Required {
				foundNamespace = true
			}
		}
		if !foundNamespace {
			t.Fatal("self delegation operation omits required namespace")
		}
	}
	if create.RouterSecret.Output != SecretOutputOneTime || !create.RouterSecret.NoStore {
		t.Fatalf("self delegation secret metadata = %+v", create.RouterSecret)
	}
}

func TestIdentityLifecycleOpenAPIUsesExactTypedContracts(t *testing.T) {
	document := GenerateOpenAPI()
	for path, methodAndSchema := range map[string][2]string{
		BasePath + "/me":                       {"get", "Me"},
		BasePath + "/self/management-sessions": {"get", "ManagementSessionPage"},
		BasePath + "/management-principals/{principalId}/management-sessions": {"get", "ManagementSessionPage"},
		BasePath + "/trusted-identity-issuers":                                {"get", "TrustedIdentityIssuerPage"},
		BasePath + "/trusted-identity-issuers/{issuerId}":                     {"get", "TrustedIdentityIssuerDetail"},
	} {
		operation := document.Paths[path][methodAndSchema[0]]
		if operation.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/"+methodAndSchema[1] {
			t.Fatalf("%s response schema = %+v", path, operation.Responses["200"])
		}
	}
	logout := document.Paths[BasePath+"/auth/backchannel-logout"]["post"]
	if logout.RequestBody == nil ||
		logout.RequestBody.Content[JSONMediaType].Schema.Ref != "#/components/schemas/BackchannelLogoutRequest" ||
		logout.Responses["200"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/BackchannelLogoutResponse" ||
		logout.RouterSecret.Input != SecretInputBody || !logout.RouterSecret.NoStore {
		t.Fatalf("back-channel logout contract = %+v", logout)
	}
	issuerDelete, found := LookupOperation(MethodDELETE, BasePath+"/trusted-identity-issuers/{issuerId}")
	if !found || issuerDelete.Revision != RevisionCAS {
		t.Fatalf("trusted issuer delete contract = %+v", issuerDelete)
	}
	managementSession := document.Components.Schemas["ManagementSession"]
	for _, forbidden := range []string{"tokenId", "assurance", "issuerSessionId"} {
		if _, found := managementSession.Properties[forbidden]; found {
			t.Fatalf("ManagementSession exposes %q", forbidden)
		}
	}
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}
