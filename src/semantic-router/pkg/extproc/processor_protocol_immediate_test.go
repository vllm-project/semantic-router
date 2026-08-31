package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestCreateErrorResponseTypeMatchesStatusClass(t *testing.T) {
	router := &OpenAIRouter{}
	for status, wantType := range map[int]string{403: "invalid_request_error", 503: "api_error"} {
		body := router.createErrorResponse(status, "boom").GetImmediateResponse().GetBody()
		var parsed struct {
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatal(err)
		}
		if parsed.Error.Type != wantType {
			t.Fatalf("status %d type = %q, want %q", status, parsed.Error.Type, wantType)
		}
	}
}

func TestFastResponseUsesClientProtocolAcrossEveryBackendAndMode(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.FastResponsePluginConfig{Message: "policy response"})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "fast",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginFastResponse, Configuration: payload,
		}},
	}
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	forEachExtProcMatrixMode(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat, streaming bool) {
		assertFastImmediateResponse(t, router, engine, decision, clientFormat, backendFormat, streaming)
	})
}

func assertFastImmediateResponse(
	t *testing.T,
	router *OpenAIRouter,
	engine *protocolcodec.Engine,
	decision *config.Decision,
	clientFormat, backendFormat llmprotocol.WireFormat,
	streaming bool,
) {
	t.Helper()
	ctx := immediateResponseContext(t, clientFormat, backendFormat, streaming)
	ctx.VSRSelectedDecision = decision
	response := router.handleFastResponse(ctx, decision.Name)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("fast response did not return an immediate response")
	}
	body := response.GetImmediateResponse().Body
	if streaming {
		assertImmediateStream(t, engine, clientFormat, response, body)
		return
	}
	assertImmediateJSON(t, response)
	decoded, _, _, err := engine.DecodeResponse(clientFormat, body)
	if err != nil {
		t.Fatalf("fast response is not valid %s: %v\n%s", clientFormat, err, body)
	}
	if decoded.Model != "public-model" || responseText(decoded) != "policy response" {
		t.Fatalf("fast response semantics changed: %+v", decoded)
	}
}

func responseText(response llmprotocol.Response) string {
	var text string
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentText {
				text += content.Text
			}
		}
	}
	return text
}

func TestCacheHitUsesClientProtocolAcrossEveryBackendAndMode(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	forEachExtProcMatrixMode(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat, streaming bool) {
		assertCacheImmediateResponse(t, router, engine, clientFormat, backendFormat, streaming)
	})
}

func assertCacheImmediateResponse(
	t *testing.T,
	router *OpenAIRouter,
	engine *protocolcodec.Engine,
	clientFormat, backendFormat llmprotocol.WireFormat,
	streaming bool,
) {
	t.Helper()
	ctx := immediateResponseContext(t, clientFormat, backendFormat, streaming)
	response := router.createCacheHitResponse(ctx, extProcResponseFixture(clientFormat), "", "", nil, 0)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("cache hit did not return an immediate response")
	}
	body := response.GetImmediateResponse().Body
	if streaming {
		assertImmediateStream(t, engine, clientFormat, response, body)
		return
	}
	assertImmediateJSON(t, response)
	if _, _, _, err := engine.DecodeResponse(clientFormat, body); err != nil {
		t.Fatalf("cache response is not valid %s: %v\n%s", clientFormat, err, body)
	}
}

func immediateResponseContext(
	t *testing.T,
	clientFormat, backendFormat llmprotocol.WireFormat,
	streaming bool,
) *RequestContext {
	t.Helper()
	return &RequestContext{
		SourceFormat: clientFormat, TargetFormat: backendFormat,
		RequestID: "request_1", RequestModel: "public-model",
		ExpectStreamingResponse: streaming, TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{Model: "public-model", Stream: streaming},
	}
}

func assertImmediateStream(t *testing.T, engine *protocolcodec.Engine, format llmprotocol.WireFormat, response *ext_proc.ProcessingResponse, body []byte) {
	t.Helper()
	if got := immediateHeaderValue(response, "content-type"); got != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", got)
	}
	assertClientStreamDecodes(t, engine, format, body)
}

func assertImmediateJSON(t *testing.T, response *ext_proc.ProcessingResponse) {
	t.Helper()
	if got := immediateHeaderValue(response, "content-type"); got != "application/json" {
		t.Fatalf("content-type = %q, want application/json", got)
	}
}

func forEachExtProcMatrixMode(
	t *testing.T,
	run func(*testing.T, llmprotocol.WireFormat, llmprotocol.WireFormat, bool),
) {
	t.Helper()
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			for _, streaming := range []bool{false, true} {
				mode := "buffered"
				if streaming {
					mode = "stream"
				}
				name := string(clientFormat) + "_client_" + string(backendFormat) + "_backend_" + mode
				t.Run(name, func(t *testing.T) { run(t, clientFormat, backendFormat, streaming) })
			}
		}
	}
}
