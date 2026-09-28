package extproc

import (
	"encoding/json"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Confidence requests native Chat logprobs on authenticated internal calls.
// They are transport evidence, not a public neutral protocol capability.
type looperLogprobOptions struct {
	Logprobs    bool `json:"logprobs"`
	TopLogprobs int  `json:"top_logprobs"`
}

func decodeRequestWithLooperEvidence(
	engine *protocolcodec.Engine, body []byte, ctx *RequestContext,
) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	request, envelope, diagnostics, err := engine.DecodeRequestForMutation(ctx.SourceFormat, body)
	var protocolErr *llmprotocol.ProtocolError
	if !ctx.LooperRequest || ctx.SourceFormat != llmprotocol.OpenAIChatV1 ||
		!errors.As(err, &protocolErr) || (protocolErr.Code != "unsupported_logprobs" && protocolErr.Code != "unsupported_top_logprobs") {
		return request, envelope, diagnostics, err
	}
	// Decode with the ordinary codec first: size/depth limits, duplicate fields,
	// exact field casing, and JSON shape must pass before any field is removed.
	cleaned, options, err := extractLooperEvidence(body)
	if err != nil {
		return request, envelope, diagnostics, err
	}
	request, envelope, diagnostics, err = engine.DecodeRequestForMutation(ctx.SourceFormat, cleaned)
	if err == nil {
		ctx.LooperLogprobs = options
	}
	return request, envelope, diagnostics, err
}

func extractLooperEvidence(body []byte) ([]byte, *looperLogprobOptions, error) {
	var options looperLogprobOptions
	if err := json.Unmarshal(body, &options); err != nil || !options.Logprobs || options.TopLogprobs < 1 || options.TopLogprobs > 5 {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest,
			"invalid_looper_logprobs", "internal confidence requests require logprobs=true and top_logprobs between 1 and 5", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, nil, err
	}
	delete(fields, "logprobs")
	delete(fields, "top_logprobs")
	cleaned, err := json.Marshal(fields)
	return cleaned, &options, err
}

func encodeLooperEvidence(body []byte, format llmprotocol.WireFormat, ctx *RequestContext) ([]byte, error) {
	if ctx.LooperLogprobs == nil {
		return body, nil
	}
	if !ctx.LooperRequest || ctx.SourceFormat != llmprotocol.OpenAIChatV1 || format != llmprotocol.OpenAIChatV1 {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_logprobs",
			"internal confidence logprobs require a Chat Completions backend", nil)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	fields["logprobs"], _ = json.Marshal(ctx.LooperLogprobs.Logprobs)
	fields["top_logprobs"], _ = json.Marshal(ctx.LooperLogprobs.TopLogprobs)
	return json.Marshal(fields)
}
