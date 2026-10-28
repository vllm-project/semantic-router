package looper

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// validateLooperStageCandidate runs after all stage and client wire mutations,
// before the connector can send any request. It preserves legacy calls without
// an explicit recipe policy, including their existing context-growth check.
func validateLooperStageCandidate(base *Request, body []byte, model string, logprobs *LogprobsConfig) error {
	if base == nil || !selection.CandidateRequirementsEnabled(base.CandidateRequirements) {
		return nil
	}
	if !slices.Contains(base.PermittedModels, model) {
		return fmt.Errorf("%w: looper model %q is outside the permitted worker and helper set", selection.ErrNoEligibleCandidates, model)
	}
	params, ok := selection.CandidateModelParams(base.ModelParams, base.ModelRefs, model)
	if !ok {
		return fmt.Errorf("%w: looper model %q has no declared metadata", selection.ErrNoEligibleCandidates, model)
	}
	codec := protocolcodec.OpenAIChatCodec{}
	if logprobs != nil && logprobs.Enabled {
		// The client itself generated bounded native Chat logprob evidence.
		// Authenticated internal dispatch preserves these two fields separately
		// from the neutral protocol; they change neither input nor output size.
		// Keep the actual wire body untouched and decode all semantic fields.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return fmt.Errorf("%w: looper stage wire is invalid", selection.ErrNoEligibleCandidates)
		}
		delete(fields, "logprobs")
		delete(fields, "top_logprobs")
		var err error
		body, err = json.Marshal(fields)
		if err != nil {
			return fmt.Errorf("%w: looper stage wire is invalid", selection.ErrNoEligibleCandidates)
		}
	}
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		// Codec diagnostics may include field values; admission errors expose
		// only the stage model and never its prompt or tool definitions.
		return fmt.Errorf("%w: looper model %q stage cannot be represented by its Chat codec", selection.ErrNoEligibleCandidates, model)
	}
	demand := selection.DemandForRequest(&request)
	if err := selection.ValidateCandidateRequirements(base.CandidateRequirements, model, params, demand); err != nil {
		return err
	}
	return selection.ValidateCandidateCodec(base.CandidateRequirements, model, codec.Capabilities(), demand)
}

func capLooperStageOutput(base *Request, request *openai.ChatCompletionNewParams) error {
	if base == nil || base.MaxTokensLimit == nil {
		return nil
	}
	if request.MaxTokens.Valid() && request.MaxCompletionTokens.Valid() && request.MaxTokens.Value != request.MaxCompletionTokens.Value {
		return fmt.Errorf("%w: looper stage has conflicting output limits", selection.ErrNoEligibleCandidates)
	}
	view := llmprotocol.Request{}
	if request.MaxCompletionTokens.Valid() {
		view.Sampling.MaxOutputTokens = llmprotocol.Int64(request.MaxCompletionTokens.Value)
	} else if request.MaxTokens.Valid() {
		view.Sampling.MaxOutputTokens = llmprotocol.Int64(request.MaxTokens.Value)
	}
	if llmprotocol.CapOutputTokens(&view, base.MaxTokensLimit) {
		if request.MaxCompletionTokens.Valid() {
			request.MaxCompletionTokens = openai.Int(*view.Sampling.MaxOutputTokens)
		}
		if request.MaxTokens.Valid() {
			request.MaxTokens = openai.Int(*view.Sampling.MaxOutputTokens)
		}
	}
	return nil
}
