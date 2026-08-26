package managementapi

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestGenerateOpenAPI31IsDeterministicAndRegistryDriven(t *testing.T) {
	first, err := GenerateOpenAPIJSON()
	if err != nil {
		t.Fatalf("GenerateOpenAPIJSON() error = %v", err)
	}
	second, err := GenerateOpenAPIJSON()
	if err != nil {
		t.Fatalf("second GenerateOpenAPIJSON() error = %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("generated OpenAPI JSON is not deterministic")
	}

	var document OpenAPIDocument
	if err := json.Unmarshal(first, &document); err != nil {
		t.Fatalf("generated OpenAPI is invalid JSON: %v", err)
	}
	if document.OpenAPI != "3.1.0" || document.JSONSchemaDialect != openAPIJSONSchemaDialect {
		t.Fatalf("unexpected OpenAPI contract header: %#v", document)
	}
	if len(document.Servers) != 1 || document.Servers[0].URL != "https://{managementAddress}" {
		t.Fatalf("managed OpenAPI must advertise Router-terminated HTTPS: %#v", document.Servers)
	}

	operationCount := 0
	for _, path := range document.Paths {
		operationCount += len(path)
	}
	if operationCount != len(Operations()) {
		t.Fatalf("OpenAPI operation count = %d, registry count = %d", operationCount, len(Operations()))
	}
	for _, contract := range Operations() {
		operation, found := document.Paths[contract.Path][strings.ToLower(string(contract.Method))]
		if !found {
			t.Fatalf("OpenAPI omitted %s %s", contract.Method, contract.Path)
		}
		if operation.OperationID != contract.OperationID || operation.RouterPermissionCanonical != contract.Permission.Canonical() {
			t.Errorf("OpenAPI metadata drift for %s %s", contract.Method, contract.Path)
		}
		accept, found := openAPIParameter(operation.Parameters, "Accept", "header")
		if !found || !accept.Required {
			t.Errorf("%s %s does not require an explicit Accept header", contract.Method, contract.Path)
			continue
		}
		wantMedia := []string{JSONMediaType}
		if contract.Method == MethodGET && contract.Path == BasePath+"/routing/exports/current" {
			wantMedia = []string{YAMLMediaType}
		}
		if contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions/{session}/events" {
			wantMedia = append(wantMedia, EventStreamMediaType)
		}
		if strings.Join(accept.Schema.Enum, "\n") != strings.Join(wantMedia, "\n") {
			t.Errorf("%s %s Accept media = %v, want %v", contract.Method, contract.Path, accept.Schema.Enum, wantMedia)
		}
		seenParameters := make(map[string]struct{}, len(operation.Parameters))
		for _, parameter := range operation.Parameters {
			key := parameter.In + "\x00" + parameter.Name
			if _, duplicate := seenParameters[key]; duplicate {
				t.Errorf("%s %s publishes duplicate %s parameter %q", contract.Method, contract.Path, parameter.In, parameter.Name)
			}
			seenParameters[key] = struct{}{}
		}
	}
}

func TestOpenAPIUsesStringSchemasForQuotaAndCost(t *testing.T) {
	document := GenerateOpenAPI()
	quota := document.Components.Schemas["QuotaMeter"]
	for _, field := range []string{"limit", "used"} {
		schema := quota.Properties[field]
		if schema.Type != "string" || schema.Pattern == "" {
			t.Errorf("QuotaMeter.%s schema = %#v, want patterned string", field, schema)
		}
	}
	remaining := quota.Properties["remaining"]
	if len(remaining.OneOf) != 2 || remaining.OneOf[0].Type != "string" || remaining.OneOf[0].Pattern == "" || remaining.OneOf[1].Type != "null" {
		t.Errorf("QuotaMeter.remaining schema = %#v, want patterned string or null", remaining)
	}
	cost := document.Components.Schemas["CostSummary"]
	for _, field := range []string{"knownAmount", "knownDispatches", "incompleteDispatches"} {
		schema := cost.Properties[field]
		if schema.Type != "string" || schema.Pattern == "" {
			t.Errorf("CostSummary.%s schema = %#v, want patterned string", field, schema)
		}
	}
}

func TestSecretOperationsGenerateNoStoreResponses(t *testing.T) {
	document := GenerateOpenAPI()
	for _, contract := range Operations() {
		if !contract.Secret.NoStore {
			continue
		}
		operation := document.Paths[contract.Path][strings.ToLower(string(contract.Method))]
		for _, response := range operation.Responses {
			header, found := response.Headers["Cache-Control"]
			if !found || len(header.Schema.Enum) != 1 || header.Schema.Enum[0] != "no-store" {
				t.Errorf("%s %s secret-bearing response has no no-store header", contract.Method, contract.Path)
			}
		}
	}
}

func TestOpenAPIRequiresExplicitNamespaceOnlyForUnscopedNamespaceResources(t *testing.T) {
	document := GenerateOpenAPI()
	tests := []struct {
		method HTTPMethod
		path   string
		want   bool
	}{
		{MethodGET, BasePath + "/providers", true},
		{MethodPOST, BasePath + "/provider-credentials", true},
		{MethodGET, BasePath + "/routing/models", true},
		{MethodGET, BasePath + "/routing/model-cards", true},
		{MethodGET, BasePath + "/users", true},
		{MethodGET, BasePath + "/usage/series", true},
		{MethodGET, BasePath + "/self/inference-keys", true},
		{MethodPOST, BasePath + "/self/inference-sessions", true},
		{MethodGET, BasePath + "/self/management-sessions", false},
		{MethodGET, BasePath + "/namespaces/{namespaceId}", false},
		{MethodPOST, BasePath + "/auth/token-exchange", false},
		{MethodGET, BasePath + "/runtime-diagnostics", false},
	}
	for _, test := range tests {
		operation := document.Paths[test.path][strings.ToLower(string(test.method))]
		found := false
		for _, parameter := range operation.Parameters {
			if parameter.In == "header" && parameter.Name == HeaderNamespaceID {
				found = parameter.Required
			}
		}
		if found != test.want {
			t.Errorf("%s %s namespace header = %v, want %v", test.method, test.path, found, test.want)
		}
	}
}
