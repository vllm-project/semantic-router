package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func prepareNativeLooperStream(
	engine *protocolcodec.Engine, response *looper.Response, ctx *RequestContext,
) (*llmprotocol.Response, []byte, error) {
	// Ratings synthesizes a native Chat stream with alternative choices after
	// every provider stream has completed successfully. Its buffered snapshot
	// retains those alternatives for validation and semantic accounting.
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, response.BufferedBody, nil)
	if err != nil {
		return nil, nil, err
	}
	semantic := &translated.Response
	if semantic.Model != response.Model {
		semantic.Model = response.Model
		semantic.Generation++
	}
	body := response.Body
	if !streamUsageRequestedByClient(ctx) {
		filter := protocolcodec.NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
		body, err = filter.Push(body)
		if err != nil {
			return nil, nil, err
		}
		final, finalErr := filter.Finalize()
		if finalErr != nil {
			return nil, nil, finalErr
		}
		body = append(body, final...)
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, translated.Diagnostics...)
	return semantic, body, nil
}
