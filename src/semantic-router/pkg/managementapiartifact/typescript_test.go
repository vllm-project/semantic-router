package managementapiartifact

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func TestTypeScriptContractIsDeterministicAndRegistryComplete(t *testing.T) {
	first := RenderTypeScriptContract()
	second := RenderTypeScriptContract()
	if !bytes.Equal(first, second) {
		t.Fatal("TypeScript Management contract is not deterministic")
	}

	contract := string(first)
	for _, operation := range managementapi.Operations() {
		if !strings.Contains(contract, "  "+operation.OperationID+": {\n") {
			t.Errorf("TypeScript contract omits operation %q", operation.OperationID)
		}
	}
	if strings.Count(contract, "    method: ") != len(managementapi.Operations()) {
		t.Fatalf("TypeScript contract operation count does not match registry")
	}
}

func TestTypeScriptContractCarriesCanonicalTransportMetadata(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, value := range []string{
		managementapi.APIVersion,
		managementapi.BasePath,
		managementapi.JSONMediaType,
		managementapi.HeaderNamespaceID,
		managementapi.HeaderIdempotencyKey,
		managementapi.HeaderIfMatch,
	} {
		if !strings.Contains(contract, tsString(value)) {
			t.Errorf("TypeScript contract omits transport value %q", value)
		}
	}
	if !strings.Contains(contract, "export function managementApiPath") {
		t.Fatal("TypeScript contract omits the typed path builder")
	}
}

func TestTypeScriptContractGeneratesTypedManagementClientFromOpenAPI(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, expected := range []string{
		"export type AgentProfile = {",
		"export type AgentEvent<EventType extends AgentEventType = AgentEventType>",
		"export type APIKey = {",
		"export type RoutingModelView = {",
		"export type UsageSummary = {",
		"export interface ManagementApiOperationTypes",
		"export type ManagementApiRequestHeaders",
		"status: number",
		"export function createManagementApiClient",
		"export function assertManagementApiSchema",
		"export function assertManagementApiOperationResponse",
	} {
		if !strings.Contains(contract, expected) {
			t.Errorf("TypeScript contract omits generated Management client fragment %q", expected)
		}
	}

	for _, operation := range managementapi.Operations() {
		if !strings.Contains(contract, "  "+operation.OperationID+"(") {
			t.Errorf("TypeScript Management client omits operation %q", operation.OperationID)
		}
	}
}

func TestTypeScriptContractDerivesConcurrencyAndReplayHeadersFromOpenAPI(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, expected := range []string{
		"'Idempotency-Key': string",
		"'If-Match': string",
		"'Last-Event-ID'?: string",
		"response.data, response.status",
	} {
		if !strings.Contains(contract, expected) {
			t.Errorf("TypeScript Management client omits OpenAPI-derived contract %q", expected)
		}
	}
}

func TestTypeScriptContractCoversConsoleManagementDomains(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, operationID := range []string{
		// API keys.
		"getApiKeys",
		"getApiKeysByKeyId",
		"postApiKeys",
		"patchApiKeysByKeyId",
		"deleteApiKeysByKeyId",
		// Users and teams.
		"getUsers",
		"getUsersByUserId",
		"postUsers",
		"patchUsersByUserId",
		"deleteUsersByUserId",
		"getTeams",
		"getTeamsByTeamId",
		"postTeams",
		"patchTeamsByTeamId",
		"deleteTeamsByTeamId",
		// Access policy and quota resources.
		"getAccessPolicies",
		"getRateLimitPolicies",
		// Routing models, recipes, and entrypoints.
		"getRoutingModels",
		"getRoutingModelsByModelId",
		"postRoutingModels",
		"patchRoutingModelsByModelId",
		"deleteRoutingModelsByModelId",
		"getRoutingRecipes",
		"getRoutingRecipesByRecipeId",
		"postRoutingRecipes",
		"patchRoutingRecipesByRecipeId",
		"deleteRoutingRecipesByRecipeId",
		"getRoutingEntrypoints",
		"getRoutingEntrypointsByEntrypointId",
		"postRoutingEntrypoints",
		"patchRoutingEntrypointsByEntrypointId",
		"deleteRoutingEntrypointsByEntrypointId",
		// Provider credentials.
		"getProviderCredentials",
		"getProviderCredentialsByCredentialId",
		"postProviderCredentials",
		"patchProviderCredentialsByCredentialId",
		"deleteProviderCredentialsByCredentialId",
		// Statistics and usage.
		"getStatistics",
		"getUsage",
		"getUsageSeries",
		"getUsageBreakdowns",
	} {
		if !strings.Contains(contract, "  "+operationID+"(") {
			t.Errorf("TypeScript Management client omits required console operation %q", operationID)
		}
	}
	for _, schemaName := range []string{
		"APIKey",
		"User",
		"Team",
		"AccessPolicy",
		"RateLimitPolicy",
		"EffectiveQuota",
		"RoutingModelView",
		"RoutingRecipeView",
		"RoutingEntrypointView",
		"ProviderCredential",
		"AccessStatistics",
		"UsageSummary",
	} {
		if !strings.Contains(contract, "export type "+schemaName+" = ") {
			t.Errorf("TypeScript Management contract omits schema %q", schemaName)
		}
	}
}

func TestTypeScriptContractDistinguishesJSONStreamsAndEmptyResponses(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, expected := range []string{
		"successStatuses: [204] as const",
		"successResponses: { 204: [] } as const",
		"responseMode: 'empty'",
		"responseMode: 'json_or_event_stream'",
		"responseMode: 'yaml'",
		"successMediaTypes: ['application/yaml'] as const",
		"secret: { input: 'none', output: 'one_time_secret', noStore: true",
	} {
		if !strings.Contains(contract, expected) {
			t.Errorf("TypeScript Management transport metadata omits %q", expected)
		}
	}
}

func TestTypeScriptContractKeepsSuccessfulStatusAndMediaContractsPaired(t *testing.T) {
	operation := managementapi.OpenAPIOperation{
		Responses: map[string]managementapi.OpenAPIResponse{
			"200": {
				Content: map[string]managementapi.OpenAPIMedia{
					managementapi.JSONMediaType: {Schema: managementapi.JSONSchema{Type: "string"}},
				},
			},
			"204": {},
		},
	}
	if got, want := renderSuccessResponses(operation),
		"{ 200: ["+tsString(managementapi.JSONMediaType)+"], 204: [] } as const"; got != want {
		t.Fatalf("renderSuccessResponses() = %q, want %q", got, want)
	}
	if got, want := operationResponseType(operation), "string | void"; got != want {
		t.Fatalf("operationResponseType() = %q, want %q", got, want)
	}
}

func TestTypeScriptContractGeneratesTextResponseFromOpenAPI(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, expected := range []string{
		"getRoutingExportsCurrent: {",
		"response: string",
		"getRoutingExportsCurrent(",
		"response.data, response.status, response.mediaType",
	} {
		if !strings.Contains(contract, expected) {
			t.Errorf("TypeScript Management text client omits %q", expected)
		}
	}
}

func TestManagementSchemaRendererPreservesRequiredAndClosedProperties(t *testing.T) {
	closed := false
	got := renderTypeScriptSchema(managementapi.JSONSchema{
		Type:                 "object",
		Required:             []string{"state"},
		AdditionalProperties: &closed,
		Properties: map[string]managementapi.JSONSchema{
			"state": {Type: "string", Enum: []string{"ready", "disabled"}},
			"note":  {Type: "string"},
		},
	}, "")
	want := "{\n  note?: string\n  state: 'ready' | 'disabled'\n}"
	if got != want {
		t.Fatalf("renderTypeScriptSchema() = %q, want %q", got, want)
	}
}

func TestTypeScriptStringEscapesExecutableDelimiters(t *testing.T) {
	got := tsString("one\\two'three\nfour")
	want := `'one\\two\'three\nfour'`
	if got != want {
		t.Fatalf("tsString() = %q, want %q", got, want)
	}
}
