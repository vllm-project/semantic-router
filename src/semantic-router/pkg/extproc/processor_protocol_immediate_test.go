package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

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
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			for _, streaming := range []bool{false, true} {
				name := string(clientFormat) + "_client_" + string(backendFormat) + "_backend"
				if streaming {
					name += "_stream"
				} else {
					name += "_buffered"
				}
				t.Run(name, func(t *testing.T) {
					ctx := &RequestContext{
						SourceFormat: clientFormat, TargetFormat: backendFormat,
						RequestID: "request_1", RequestModel: "public-model",
						ExpectStreamingResponse: streaming, TraceContext: t.Context(),
						VSRSelectedDecision: decision,
						SemanticRequest: &llmprotocol.Request{
							Model: "public-model", Stream: streaming,
						},
					}
					response := router.handleFastResponse(ctx, decision.Name)
					if response == nil || response.GetImmediateResponse() == nil {
						t.Fatal("fast response did not return an immediate response")
					}
					body := response.GetImmediateResponse().Body
					if streaming {
						if got := immediateHeaderValue(response, "content-type"); got != "text/event-stream" {
							t.Fatalf("content-type = %q, want text/event-stream", got)
						}
						assertClientStreamDecodes(t, engine, clientFormat, body)
						return
					}
					if got := immediateHeaderValue(response, "content-type"); got != "application/json" {
						t.Fatalf("content-type = %q, want application/json", got)
					}
					decoded, _, _, err := engine.DecodeResponse(clientFormat, body)
					if err != nil {
						t.Fatalf("fast response is not valid %s: %v\n%s", clientFormat, err, body)
					}
					if decoded.Model != "public-model" || responseText(decoded) != "policy response" {
						t.Fatalf("fast response semantics changed: %+v", decoded)
					}
				})
			}
		}
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
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			for _, streaming := range []bool{false, true} {
				name := string(clientFormat) + "_client_" + string(backendFormat) + "_cache"
				if streaming {
					name += "_stream"
				} else {
					name += "_buffered"
				}
				t.Run(name, func(t *testing.T) {
					ctx := &RequestContext{
						SourceFormat: clientFormat, TargetFormat: backendFormat,
						RequestID: "request_1", RequestModel: "public-model",
						ExpectStreamingResponse: streaming, TraceContext: t.Context(),
						SemanticRequest: &llmprotocol.Request{
							Model: "public-model", Stream: streaming,
						},
					}
					response := router.createCacheHitResponse(
						ctx, extProcResponseFixture(clientFormat), "", "", nil, 0,
					)
					if response == nil || response.GetImmediateResponse() == nil {
						t.Fatal("cache hit did not return an immediate response")
					}
					body := response.GetImmediateResponse().Body
					if streaming {
						if got := immediateHeaderValue(response, "content-type"); got != "text/event-stream" {
							t.Fatalf("content-type = %q, want text/event-stream", got)
						}
						assertClientStreamDecodes(t, engine, clientFormat, body)
						return
					}
					if got := immediateHeaderValue(response, "content-type"); got != "application/json" {
						t.Fatalf("content-type = %q, want application/json", got)
					}
					if _, _, _, err := engine.DecodeResponse(clientFormat, body); err != nil {
						t.Fatalf("cache response is not valid %s: %v\n%s", clientFormat, err, body)
					}
				})
			}
		}
	}
}
