package extproc

import (
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A coding agent's history counts one estimated token per byte of tool-call
// arguments, so a request well inside a model's window can be estimated above
// it. Without that reserve the request fits, so the Router still sends it.
func TestProcessRequestBodySendsRequestsTheEstimatePutsAboveEveryWindow(t *testing.T) {
	for _, format := range extProcMatrixFormats {
		t.Run(string(format), func(t *testing.T) {
			router := routingTestRouter("worker")
			params := router.Config.ModelConfig["worker"]
			params.ContextWindowSize = 3000
			router.Config.ModelConfig["worker"] = params
			router.Config.Decisions = []config.Decision{{
				Name: "default", ModelRefs: []config.ModelRef{{Model: "worker"}},
			}}
			classifier, err := classification.NewClassifier(router.Config)
			require.NoError(t, err)
			t.Cleanup(func() { _ = classifier.Close() })
			router.Classifier = classifier

			request := testNeutralRequest("vllm-sr/auto", "write the file")
			arguments := `{"content":"` + strings.Repeat("x", 6000) + `"}`
			request.Messages = append(request.Messages,
				llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "write", Arguments: arguments},
				}}},
				llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
						CallID: "call_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "written"}},
					},
				}}},
			)
			ctx := routingTestContext(format, request)
			encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(format, *request, llmprotocol.Envelope{})
			require.NoError(t, err)
			stream := NewMockStream(nil)
			err = router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{Body: encoded.Body, EndOfStream: true},
			}, ctx)
			require.NoError(t, err)
			require.Len(t, stream.Responses, 1)
			require.Nil(t, stream.Responses[0].GetImmediateResponse(), "the request must reach the backend")
			require.Equal(t, "worker", ctx.VSRSelectedModel)
			require.LessOrEqual(t, ctx.VSRContextTokenCount, params.ContextWindowSize, "the request continues without its reserves")
			require.Zero(t, ctx.VSRContextReserveTokens)
		})
	}
}
