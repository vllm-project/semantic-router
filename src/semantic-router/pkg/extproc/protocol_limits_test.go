package extproc

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Claude Code's Artifact tool description is about 20 KiB, over the 16 KiB
// default. A raised limit admits it at ingress and on the backend re-encode.
func TestProtocolLimitsAdmitConfiguredToolDescriptionAcrossFormats(t *testing.T) {
	description := strings.Repeat("d", 20_225)
	bodies := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"model":"%s","messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"function","function":{"name":"Artifact","description":"` + description + `","parameters":{"type":"object"}}}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"%s","max_tokens":64,"messages":[{"role":"user","content":"hi"}],
			"tools":[{"name":"Artifact","description":"` + description + `","input_schema":{"type":"object"}}]}`,
	}
	for format, template := range bodies {
		for _, raised := range []bool{false, true} {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			if raised {
				router.Config.ProtocolLimits = config.ProtocolLimitsConfig{ToolDescriptionBytes: 64 << 10}
			}
			ctx := routingTestContext(format, nil)
			request, rejection := router.prepareProtocolRequest([]byte(strings.Replace(template, "%s", model, 1)), ctx)
			if !raised {
				if rejection == nil || ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "tool_text_limit" {
					t.Fatalf("%s default ingress = %+v, want tool_text_limit", format, ctx.ImmediateProtocolError)
				}
				want := "tools[0].description is 20225 bytes and exceeds the configured limit of 16384 bytes"
				if ctx.ImmediateProtocolError.Message != want {
					t.Fatalf("%s default message = %q, want %q", format, ctx.ImmediateProtocolError.Message, want)
				}
				continue
			}
			if rejection != nil {
				t.Fatalf("%s raised ingress rejected the tool: %+v", format, ctx.ImmediateProtocolError)
			}
			if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
				t.Fatal(err)
			}
			encoded, err := router.encodeDispatchRequest(ctx)
			if err != nil || !bytes.Contains(encoded, []byte(description)) {
				t.Fatalf("%s raised dispatch lost the description: error=%v", format, err)
			}
		}
	}
}

func TestProtocolLimitsReachEveryRouterEngine(t *testing.T) {
	router, _ := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.ProtocolLimits = config.ProtocolLimitsConfig{ToolNameBytes: 512, ToolDescriptionBytes: 64 << 10}

	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"` +
		strings.Repeat("n", 300) + `","description":"` + strings.Repeat("d", 20_225) + `","parameters":{"type":"object"}}}]}`)
	engines := map[string]func() (*protocolcodec.Engine, error){
		"ingress": router.protocolEngine,
		"backend": func() (*protocolcodec.Engine, error) {
			return router.protocolEngineForVendor(llmprotocol.ResponseVendorCloudflare)
		},
		"detached": func() (*protocolcodec.Engine, error) {
			return protocolEngineFor(nil, router.Config.ProtocolPolicy())
		},
	}
	for name, build := range engines {
		engine, err := build()
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body); err != nil {
			t.Fatalf("%s engine ignored the configured limits: %v", name, err)
		}
	}
}
