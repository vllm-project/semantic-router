package protocolcodec

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

type providerSimulatorContractDocument struct {
	Protocols map[string]providerSimulatorContract `json:"protocols"`
}

type providerSimulatorContract struct {
	SchemaRevision          string   `json:"schema_revision"`
	OfficialRequestFields   []string `json:"official_request_fields"`
	ExtensionRequestFields  []string `json:"extension_request_fields"`
	RequiredRequestFields   []string `json:"required_request_fields"`
	OfficialResponseFields  []string `json:"official_response_fields"`
	ExtensionResponseFields []string `json:"extension_response_fields"`
	OfficialUsageFields     []string `json:"official_usage_fields"`
}

func TestProviderSimulatorContractsTrackCodecInventories(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	openAI := readProviderSimulatorContracts(
		t,
		filepath.Join(root, "tools", "mock-vllm", "schema_contract.json"),
	)
	anthropic := readProviderSimulatorContracts(
		t,
		filepath.Join(root, "e2e", "testing", "anthropic-shim", "schema_contract.json"),
	)

	tests := []struct {
		name           string
		contract       providerSimulatorContract
		revision       string
		requestWire    any
		responseWire   any
		usageWire      any
		requiredFields []string
	}{
		{
			name: "OpenAI Chat Completions", contract: openAI["openai_chat_completions"],
			revision:    "690521b1753dce0c6d6b275f583d22537679cff9",
			requestWire: chatRequestWire{}, responseWire: chatResponseWire{}, usageWire: chatUsageWire{},
			requiredFields: fields("messages", "model"),
		},
		{
			name: "OpenAI Responses", contract: openAI["openai_responses"],
			revision:    "690521b1753dce0c6d6b275f583d22537679cff9",
			requestWire: responsesRequestWire{}, responseWire: responsesResponseWire{}, usageWire: responsesUsageWire{},
			requiredFields: fields("input"),
		},
		{
			name: "Anthropic Messages", contract: anthropic["anthropic_messages"],
			revision:    "d19dea9ed85bbb5fdb2d6f20fb6f903920ed23fa",
			requestWire: anthropicRequestWire{}, responseWire: anthropicResponseWire{}, usageWire: anthropicUsageWire{},
			requiredFields: fields("max_tokens", "messages", "model"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.contract.SchemaRevision != test.revision {
				t.Fatalf("simulator schema revision = %q, want %q", test.contract.SchemaRevision, test.revision)
			}
			assertSimulatorFields(t, "request", test.contract.OfficialRequestFields, test.contract.ExtensionRequestFields, jsonFieldNames(reflect.TypeOf(test.requestWire)))
			assertSimulatorFields(t, "response", test.contract.OfficialResponseFields, test.contract.ExtensionResponseFields, jsonFieldNames(reflect.TypeOf(test.responseWire)))
			assertSimulatorFields(t, "usage", test.contract.OfficialUsageFields, nil, jsonFieldNames(reflect.TypeOf(test.usageWire)))
			if !reflect.DeepEqual(sortedStrings(test.contract.RequiredRequestFields), test.requiredFields) {
				t.Fatalf("required request fields = %v, want %v", test.contract.RequiredRequestFields, test.requiredFields)
			}
		})
	}
}

func readProviderSimulatorContracts(t *testing.T, path string) map[string]providerSimulatorContract {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document providerSimulatorContractDocument
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatalf("decode simulator contract %s: %v", path, err)
	}
	return document.Protocols
}

func assertSimulatorFields(t *testing.T, kind string, official, extensions, want []string) {
	t.Helper()
	got := append(append([]string(nil), official...), extensions...)
	sort.Strings(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("simulator %s inventory drifted\n got: %v\nwant: %v", kind, got, want)
	}
}

func sortedStrings(values []string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
}
