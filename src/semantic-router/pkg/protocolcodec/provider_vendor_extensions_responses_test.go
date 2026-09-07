package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAzureResponsesResponseReportsDroppedVendorExtensions(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorAzure
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"completed","output":[],"azure_trace":{"region":"eastus"}}`)

	response, _, diagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse(body, policy)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.ID != "resp_1" {
		t.Fatalf("response ID = %q, want resp_1", response.ID)
	}
	assertDroppedResponseExtensions(t, diagnostics, "azure_trace")
}

func TestAzureResponsesStreamReportsDroppedVendorExtensions(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorAzure
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}

	frame := []byte("event: response.created\n" +
		`data: {"type":"response.created","sequence_number":0,"azure_trace":{"region":"eastus"},"response":{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"in_progress","output":[]}}` + "\n\n")
	_, events, diagnostics, err := stream.Push(frame)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	if len(events) == 0 {
		t.Fatal("Push() returned no events")
	}
	assertDroppedResponseExtensions(t, diagnostics, "azure_trace")
}

func TestAzureResponsesResponseReportsNestedVendorExtensions(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorAzure
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[],"azure_content":{"region":"eastus"}}],"azure_item":{"region":"eastus"}}]}`)

	response, _, diagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse(body, policy)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(response.Output) != 1 || response.Output[0].Content[0].Text != "hello" {
		t.Fatalf("response output = %+v, want nested assistant text", response.Output)
	}
	assertDroppedResponseExtensions(t, diagnostics,
		"output[].azure_item",
		"output[].content[].azure_content",
	)
}

func TestAzureResponsesStreamReportsNestedItemVendorExtensions(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorAzure
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	start := []byte("event: response.created\n" +
		`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"in_progress","output":[]}}` + "\n\n")
	if _, _, _, err := stream.Push(start); err != nil {
		t.Fatalf("Push(start) error = %v", err)
	}

	frame := []byte("event: response.output_item.added\n" +
		`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[],"azure_item":{"region":"eastus"}}}` + "\n\n")
	_, events, diagnostics, err := stream.Push(frame)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	if len(events) == 0 {
		t.Fatal("Push() returned no events")
	}
	assertDroppedResponseExtensions(t, diagnostics, "output[].azure_item")
}

func TestResponsesNestedVendorExtensionsRemainStrictWithoutAzure(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[],"azure_item":{}}]}`)

	_, _, _, err := (OpenAIResponsesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err == nil || !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Fatalf("DecodeResponse() error = %v, want invalid_upstream_json", err)
	}
}

func TestResponsesStreamNestedVendorExtensionsRemainStrictWithoutAzure(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	start := []byte("event: response.created\n" +
		`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"model":"provider-model","status":"in_progress","output":[]}}` + "\n\n")
	if _, _, _, err := stream.Push(start); err != nil {
		t.Fatalf("Push(start) error = %v", err)
	}
	frame := []byte("event: response.output_item.added\n" +
		`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[],"azure_item":{}}}` + "\n\n")

	_, _, _, err = stream.Push(frame)
	if err == nil || !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Fatalf("Push() error = %v, want invalid_upstream_json", err)
	}
}

func assertDroppedResponseExtensions(t *testing.T, diagnostics llmprotocol.Diagnostics, fields ...string) {
	t.Helper()
	dropped := make(map[string]bool, len(fields))
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped &&
			diagnostic.Source == llmprotocol.OpenAIResponsesV1 &&
			diagnostic.Reason == vendorExtensionReason {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range fields {
		if !dropped[field] {
			t.Errorf("no dropped diagnostic for %q; diagnostics = %+v", field, diagnostics)
		}
	}
	if len(dropped) != len(fields) {
		t.Errorf("vendor diagnostics = %v, want %v", dropped, fields)
	}
}
