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
		managementapi.ContractVersion,
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

func TestTypeScriptContractGeneratesTypedAgentClientFromOpenAPI(t *testing.T) {
	contract := string(RenderTypeScriptContract())
	for _, expected := range []string{
		"export type AgentProfile = {",
		"export type AgentEvent<EventType extends AgentEventType = AgentEventType>",
		"export interface ManagementApiAgentOperationTypes",
		"export function createManagementApiAgentClient",
		"export function assertManagementApiAgentSchema",
	} {
		if !strings.Contains(contract, expected) {
			t.Errorf("TypeScript contract omits generated Agent client fragment %q", expected)
		}
	}

	for _, operation := range managementapi.Operations() {
		if !isAgentClientOperation(operation) {
			continue
		}
		if !strings.Contains(contract, "  "+operation.OperationID+"(") {
			t.Errorf("TypeScript Agent client omits operation %q", operation.OperationID)
		}
	}
}

func TestAgentSchemaRendererPreservesRequiredAndClosedProperties(t *testing.T) {
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
