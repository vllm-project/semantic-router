package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Internal aggregation does not choose the client's transport. Project through
// the existing target codec first, preserving its loss diagnostics and the full
// neutral result for Replay, usage and response-object persistence. A requested
// stream renders that same public projection as events without widening target
// capabilities or multi-choice streaming. Unsupported alternatives still fail.
func prepareBufferedLooperResponse(
	engine *protocolcodec.Engine,
	resp *looper.Response,
	ctx *RequestContext,
	target llmprotocol.WireFormat,
) (*llmprotocol.Response, []byte, error) {
	response, envelope, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, resp.ProtocolBody())
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	if err != nil {
		return nil, nil, err
	}
	response.Model = resp.Model
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		response.ID = responseID
	}
	response.Generation++
	envelope.ResponseRender.PreviousResponseID = responseObjectPreviousID(ctx)
	encoded, err := engine.EncodeResponse(target, response, envelope)
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
	if err != nil {
		return nil, nil, err
	}
	ctx.ResponseEnvelope = encoded.Envelope
	if !ctx.ExpectStreamingResponse {
		return &response, encoded.Body, nil
	}
	projected, _, diagnostics, err := engine.DecodeResponse(target, encoded.Body)
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	if err != nil {
		return nil, nil, err
	}
	body, diagnostics, err := engine.EncodeResponseStream(target, projected, llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: target, Target: target,
		Options: clientStreamOptions(ctx), PreviousResponseID: responseObjectPreviousID(ctx),
	})
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	return &response, body, err
}
