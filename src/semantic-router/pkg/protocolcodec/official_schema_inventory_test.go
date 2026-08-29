package protocolcodec

import (
	"reflect"
	"sort"
	"testing"
)

// These inventories are the top-level request fields published by the
// OpenAI OpenAPI contract at 690521b1753dce0c6d6b275f583d22537679cff9 and the
// generated Anthropic Messages API types at
// d19dea9ed85bbb5fdb2d6f20fb6f903920ed23fa.
// Every field is either represented semantically or decoded into an explicit
// unsupported_feature error; adding a silent JSON sink is not allowed.
func TestOfficialRequestFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name       string
		wire       any
		official   []string
		extensions []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatRequestWire{},
			official: fields(
				"audio", "frequency_penalty", "function_call", "functions", "logit_bias", "logprobs",
				"max_completion_tokens", "max_tokens", "messages", "metadata", "modalities", "model",
				"moderation", "n", "parallel_tool_calls", "prediction", "presence_penalty",
				"prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning_effort",
				"response_format", "safety_identifier", "seed", "service_tier", "stop", "store", "stream",
				"stream_options", "temperature", "tool_choice", "tools", "top_logprobs", "top_p", "user",
				"verbosity", "web_search_options",
			),
			extensions: fields("reasoning_budget_tokens"),
		},
		{
			name: "OpenAI Responses",
			wire: responsesRequestWire{},
			official: fields(
				"background", "context_management", "conversation", "include", "input", "instructions",
				"max_output_tokens", "max_tool_calls", "metadata", "model", "moderation", "parallel_tool_calls",
				"previous_response_id", "prompt", "prompt_cache_key", "prompt_cache_options",
				"prompt_cache_retention", "reasoning", "safety_identifier", "service_tier", "store", "stream",
				"stream_options", "temperature", "text", "tool_choice", "tools", "top_logprobs", "top_p",
				"truncation", "user",
			),
			extensions: fields("auto_store"),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicRequestWire{},
			official: fields(
				"cache_control", "container", "inference_geo", "max_tokens", "messages", "metadata", "model",
				"output_config", "service_tier", "stop_sequences", "stream", "system", "temperature", "thinking",
				"tool_choice", "tools", "top_k", "top_p",
			),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			want := append(append([]string(nil), test.official...), test.extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, want) {
				t.Fatalf("wire field inventory drifted\n got: %v\nwant: %v", got, want)
			}
		})
	}
}

func TestOfficialRequestFieldDispositionsAreClosed(t *testing.T) {
	tests := []struct {
		name        string
		wire        any
		semantic    []string
		transport   []string
		unsupported []string
		extensions  []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatRequestWire{},
			semantic: fields(
				"frequency_penalty", "max_completion_tokens", "max_tokens", "messages", "metadata", "model",
				"n", "parallel_tool_calls", "presence_penalty", "reasoning_effort", "response_format", "seed",
				"stop", "store", "stream", "temperature", "tool_choice", "tools", "top_p", "user",
			),
			unsupported: fields(
				"audio", "function_call", "functions", "logit_bias", "logprobs", "modalities", "moderation",
				"prediction", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
				"safety_identifier", "service_tier", "top_logprobs", "verbosity", "web_search_options",
			),
			extensions: fields("reasoning_budget_tokens"),
			transport:  fields("stream_options"),
		},
		{
			name: "OpenAI Responses",
			wire: responsesRequestWire{},
			semantic: fields(
				"conversation", "input", "instructions", "max_output_tokens", "metadata", "model",
				"parallel_tool_calls", "previous_response_id", "reasoning", "store", "stream", "temperature",
				"text", "tool_choice", "tools", "top_p", "truncation", "user",
			),
			unsupported: fields(
				"background", "context_management", "include", "max_tool_calls", "moderation", "prompt",
				"prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "safety_identifier",
				"service_tier", "top_logprobs",
			),
			transport:  fields("stream_options"),
			extensions: fields("auto_store"),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicRequestWire{},
			semantic: fields(
				"max_tokens", "messages", "metadata", "model", "output_config", "stop_sequences", "stream",
				"system", "temperature", "thinking", "tool_choice", "tools", "top_k", "top_p",
			),
			unsupported: fields("cache_control", "container", "inference_geo", "service_tier"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertClosedFieldDisposition(t, test.name, jsonFieldNames(reflect.TypeOf(test.wire)), map[string][]string{
				"semantic": test.semantic, "transport": test.transport,
				"unsupported": test.unsupported, "extension": test.extensions,
			})
		})
	}
}

func TestOfficialResponseFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name       string
		wire       any
		official   []string
		extensions []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatResponseWire{},
			official: fields(
				"choices", "created", "id", "metadata", "model", "moderation", "object", "service_tier", "system_fingerprint", "usage",
			),
			extensions: fields(
				"do_remote_decode", "do_remote_prefill", "ec_transfer_params", "error", "kv_transfer_params", "metrics",
				"prompt_logprobs", "prompt_text", "prompt_token_ids", "remote_block_ids", "remote_engine_id",
				"remote_host", "remote_port",
			),
		},
		{
			name: "OpenAI Responses",
			wire: responsesResponseWire{},
			official: fields(
				"background", "completed_at", "conversation", "created_at", "error", "id",
				"incomplete_details", "instructions", "max_output_tokens", "max_tool_calls", "metadata",
				"model", "moderation", "object", "output", "output_text", "parallel_tool_calls", "previous_response_id",
				"prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning",
				"safety_identifier", "service_tier", "status", "temperature", "text", "tool_choice",
				"tools", "top_logprobs", "top_p", "truncation", "usage", "user",
			),
			extensions: fields("conversation_id", "store"),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicResponseWire{},
			official: fields(
				"container", "content", "id", "model", "role", "stop_details", "stop_reason",
				"stop_sequence", "type", "usage",
			),
			extensions: fields("error"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			want := append(append([]string(nil), test.official...), test.extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, want) {
				t.Fatalf("wire field inventory drifted\n got: %v\nwant: %v", got, want)
			}
		})
	}
}

func TestOfficialUsageFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name     string
		wire     any
		official []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatUsageWire{},
			official: fields(
				"completion_tokens", "completion_tokens_details", "compute_units", "prompt_tokens",
				"prompt_tokens_details", "total_tokens",
			),
		},
		{
			name: "OpenAI Responses",
			wire: responsesUsageWire{},
			official: fields(
				"compute_units", "input_tokens", "input_tokens_details", "output_tokens",
				"output_tokens_details", "total_tokens",
			),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicUsageWire{},
			official: fields(
				"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
				"input_tokens", "output_tokens", "output_tokens_details", "server_tool_use", "service_tier",
			),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, test.official) {
				t.Fatalf("usage field inventory drifted\n got: %v\nwant: %v", got, test.official)
			}
		})
	}
}
