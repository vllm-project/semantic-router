package protocolcodec

import (
	"context"
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
	assertDroppedResponseExtension(t, diagnostics, "azure_trace")
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
	assertDroppedResponseExtension(t, diagnostics, "azure_trace")
}

func assertDroppedResponseExtension(t *testing.T, diagnostics llmprotocol.Diagnostics, field string) {
	t.Helper()
	if len(diagnostics) != 1 {
		t.Fatalf("diagnostics count = %d, want 1: %+v", len(diagnostics), diagnostics)
	}
	diagnostic := diagnostics[0]
	if diagnostic.Field != field {
		t.Errorf("diagnostic field = %q, want %q", diagnostic.Field, field)
	}
	if diagnostic.Action != llmprotocol.DiagnosticDropped {
		t.Errorf("diagnostic action = %q, want %q", diagnostic.Action, llmprotocol.DiagnosticDropped)
	}
	if diagnostic.Source != llmprotocol.OpenAIResponsesV1 {
		t.Errorf("diagnostic source = %q, want %q", diagnostic.Source, llmprotocol.OpenAIResponsesV1)
	}
}
