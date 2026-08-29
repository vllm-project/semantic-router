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
	SchemaRevision          string              `json:"schema_revision"`
	OfficialRequestFields   []string            `json:"official_request_fields"`
	ExtensionRequestFields  []string            `json:"extension_request_fields"`
	ProviderSchemaRevision  string              `json:"provider_schema_revision"`
	ProviderRequestFields   map[string][]string `json:"provider_request_fields"`
	RequiredRequestFields   []string            `json:"required_request_fields"`
	OfficialResponseFields  []string            `json:"official_response_fields"`
	ExtensionResponseFields []string            `json:"extension_response_fields"`
	OfficialUsageFields     []string            `json:"official_usage_fields"`
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
		name             string
		contract         providerSimulatorContract
		revision         string
		requestWire      any
		responseWire     any
		usageWire        any
		requiredFields   []string
		providerRevision string
		providerFields   []string
	}{
		{
			name: "OpenAI Chat Completions", contract: openAI["openai_chat_completions"],
			revision:    "690521b1753dce0c6d6b275f583d22537679cff9",
			requestWire: chatRequestWire{}, responseWire: chatResponseWire{}, usageWire: chatUsageWire{},
			requiredFields:   fields("messages", "model"),
			providerRevision: "6cddad414ee46796f21aaf7b8643a6e7a00c09b5",
			providerFields: fields(
				"add_generation_prompt", "add_special_tokens", "allowed_token_ids", "bad_words", "cache_salt",
				"chat_template", "chat_template_kwargs", "continue_final_message", "documents", "echo",
				"ec_transfer_params", "ignore_eos", "include_reasoning", "include_stop_str_in_output",
				"kv_transfer_params", "length_penalty", "logprob_token_ids", "media_io_kwargs", "min_p",
				"min_tokens", "mm_processor_kwargs", "priority", "prompt_logprobs", "repetition_detection",
				"repetition_penalty", "request_id", "return_assistant_tokens_mask", "return_prompt_text",
				"return_token_ids", "return_token_offsets", "return_tokens_as_token_ids",
				"routed_experts_prompt_start", "session_id", "skip_special_tokens", "spaces_between_special_tokens",
				"stop_token_ids", "stream_interval", "structured_outputs", "thinking_token_budget", "top_k",
				"truncate_prompt_tokens", "truncation_side", "use_beam_search", "vllm_xargs",
			),
		},
		{
			name: "OpenAI Responses", contract: openAI["openai_responses"],
			revision:    "690521b1753dce0c6d6b275f583d22537679cff9",
			requestWire: responsesRequestWire{}, responseWire: responsesResponseWire{}, usageWire: responsesUsageWire{},
			requiredFields:   fields("input"),
			providerRevision: "6cddad414ee46796f21aaf7b8643a6e7a00c09b5",
			providerFields: fields(
				"cache_salt", "chat_template_kwargs", "ec_transfer_params", "enable_response_messages",
				"frequency_penalty", "ignore_eos", "include_reasoning", "include_stop_str_in_output",
				"kv_transfer_params", "logit_bias", "media_io_kwargs", "mm_processor_kwargs", "presence_penalty",
				"previous_input_messages", "priority", "repetition_penalty", "request_id", "seed", "session_id",
				"skip_special_tokens", "stop", "structured_outputs", "top_k", "vllm_xargs",
			),
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
			if test.contract.ProviderSchemaRevision != test.providerRevision {
				t.Fatalf("provider schema revision = %q, want %q", test.contract.ProviderSchemaRevision, test.providerRevision)
			}
			if got := sortedMapKeys(test.contract.ProviderRequestFields); !reflect.DeepEqual(got, test.providerFields) {
				t.Fatalf("provider request inventory drifted\n got: %v\nwant: %v", got, test.providerFields)
			}
			assertProviderFieldTypes(t, test.contract.ProviderRequestFields)
			if !reflect.DeepEqual(sortedStrings(test.contract.RequiredRequestFields), test.requiredFields) {
				t.Fatalf("required request fields = %v, want %v", test.contract.RequiredRequestFields, test.requiredFields)
			}
		})
	}
}

func assertProviderFieldTypes(t *testing.T, fields map[string][]string) {
	t.Helper()
	allowed := map[string]struct{}{
		"array": {}, "boolean": {}, "integer": {}, "null": {},
		"number": {}, "object": {}, "string": {},
	}
	for field, types := range fields {
		if len(types) == 0 {
			t.Fatalf("provider request field %q has no JSON types", field)
		}
		seen := make(map[string]struct{}, len(types))
		for _, jsonType := range types {
			if _, ok := allowed[jsonType]; !ok {
				t.Fatalf("provider request field %q has unknown JSON type %q", field, jsonType)
			}
			if _, duplicate := seen[jsonType]; duplicate {
				t.Fatalf("provider request field %q repeats JSON type %q", field, jsonType)
			}
			seen[jsonType] = struct{}{}
		}
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

func sortedMapKeys[V any](values map[string]V) []string {
	var result []string
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
