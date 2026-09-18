package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// automaticOutputUnsupported never includes provider bodies, credentials, or
// request contents. An unavailable renderer is not evidence of input overflow.
func automaticOutputUnsupported(reason string) error {
	return fmt.Errorf("%w: automatic output requires %s", selection.ErrNoEligibleCandidates, reason)
}

// resolveAutomaticOutput renders the same adapted Chat payload the backend
// receives. Render is preprocessing only: it never invokes model generation.
func (r *OpenAIRouter) resolveAutomaticOutput(request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) error {
	if request == nil || !request.Sampling.AutomaticOutput {
		return nil
	}
	if dispatch == nil || dispatch.profile == nil || dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		return automaticOutputUnsupported("a vLLM Chat backend with the render API enabled")
	}
	provider, err := dispatch.profile.ProviderType()
	if err != nil || provider != "vllm" {
		return automaticOutputUnsupported("an explicit vllm provider")
	}
	params, ok := r.Config.ModelConfig[dispatch.logicalModel]
	if !ok || params.ContextWindowSize <= 0 || params.MaxOutputTokens <= 0 || len(params.PreferredEndpoints) != 1 {
		return automaticOutputUnsupported("known model limits and exactly one configured backend")
	}
	if request.Truncation != "" && request.Truncation != "disabled" {
		return automaticOutputUnsupported("router-managed input compression instead of provider truncation")
	}
	if r.isLooperRequest(ctx) {
		return automaticOutputUnsupported("single-dispatch routing; Looper stages are not supported")
	}
	// A detached request retires ingress replay and previously materialized auto
	// limits. Provider generation defaults are included in render's returned cap.
	view := *request
	view.Model = dispatch.upstreamModel
	view.Sampling.MaxOutputTokens = nil
	view.Sampling.AutomaticInputTokens = nil
	view.Generation++
	renderCtx := *ctx
	renderCtx.SemanticRequest = &view
	renderCtx.TargetFormat = dispatch.targetFormat
	body, err := r.encodeDispatchRequest(&renderCtx)
	if err != nil {
		return err
	}
	body, err = r.adaptProviderRequest(body, dispatch, &renderCtx)
	if err != nil {
		return err
	}
	result, err := r.renderAutomaticOutput(body, dispatch, ctx)
	if err != nil {
		return err
	}
	if result.Model != dispatch.upstreamModel || len(result.TokenIDs) == 0 || result.Sampling == nil || result.Sampling.MaxTokens == nil || *result.Sampling.MaxTokens <= 0 || len(result.Features) != 0 && string(result.Features) != "null" {
		return automaticOutputUnsupported("a valid text render result with a positive output budget")
	}
	for _, token := range result.TokenIDs {
		if token < 0 {
			return automaticOutputUnsupported("nonnegative rendered token IDs")
		}
	}
	input := len(result.TokenIDs)
	remaining := params.ContextWindowSize - input
	if remaining <= 0 {
		return overflowBudgetError("The rendered input leaves no output capacity in the configured model context window")
	}
	limit := min(params.MaxOutputTokens, remaining)
	if *result.Sampling.MaxTokens < int64(limit) {
		return automaticOutputUnsupported("provider output limits aligned with the configured model limits; the render API reported a smaller output capacity")
	}
	if cap := request.Sampling.AutomaticOutputCap; cap != nil {
		limit = min(limit, int(*cap))
	}
	if limit <= 0 {
		return automaticOutputUnsupported("a positive configured output cap")
	}
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(int64(limit))
	request.Sampling.AutomaticInputTokens = llmprotocol.Int64(int64(input))
	request.Generation++
	return nil
}

type automaticRenderResult struct {
	Model    string                   `json:"model"`
	TokenIDs []int64                  `json:"token_ids"`
	Features json.RawMessage          `json:"features"`
	Sampling *automaticRenderSampling `json:"sampling_params"`
}

// vLLM serializes SamplingParams with omit_defaults. A present sampling object
// without max_tokens therefore means its schema default (16), not unknown.
// Missing/null sampling objects and explicit null max_tokens remain invalid.
type automaticRenderSampling struct {
	MaxTokens *int64 `json:"max_tokens"`
}

func (value *automaticRenderSampling) UnmarshalJSON(body []byte) error {
	type wire automaticRenderSampling
	decoded := wire{MaxTokens: llmprotocol.Int64(16)}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return err
	}
	*value = automaticRenderSampling(decoded)
	return nil
}

func (r *OpenAIRouter) renderAutomaticOutput(body []byte, dispatch *providerDispatch, ctx *RequestContext) (*automaticRenderResult, error) {
	endpoint, query, err := splitProviderEndpoint(providerEndpointPath(dispatch.profile, dispatch.targetFormat))
	if err != nil {
		return nil, err
	}
	authorize, err := configuredProviderAuthorizer(r.Config, dispatch.profile, dispatch.logicalModel)
	if err != nil {
		return nil, err
	}
	base := providerEndpointScheme(r.Config, dispatch.backendName, dispatch.profile) + "://" + dispatch.backendAddress
	client, err := connector.New(base, authorize, connector.Options{AttemptTimeout: 30 * time.Second, MaxRequestBytes: 16 << 20, MaxResponseBytes: 32 << 20, MaxErrorBytes: 8 << 10})
	if err != nil {
		return nil, err
	}
	defer client.Close()
	callCtx := ctx.TraceContext
	if callCtx == nil {
		callCtx = context.Background()
	}
	operation := connector.Operation{Name: "automatic_output_render", Method: http.MethodPost, Path: strings.TrimRight(endpoint, "/") + "/render", Query: query, SuccessStatusCode: http.StatusOK}
	response, err := client.DoRequest(callCtx, operation, connector.Request{Body: body, Headers: dispatch.profile.ExtraHeaders})
	if err != nil {
		if automaticRenderOverflow(err) {
			return nil, overflowBudgetError("The provider-rendered input exceeds the model context window")
		}
		// An exactly full window can fail sampling validation with an untyped
		// max_tokens=0 error. A single one-token render probe can establish
		// typed input overflow; its result is never used as a dispatch budget.
		var failure *connector.Error
		if errors.As(err, &failure) && failure.Kind == connector.KindStatus && failure.StatusCode == http.StatusBadRequest {
			var probe map[string]json.RawMessage
			if json.Unmarshal(body, &probe) == nil {
				delete(probe, "max_completion_tokens")
				probe["max_tokens"] = json.RawMessage("1")
				probeBody, encodeErr := json.Marshal(probe)
				if encodeErr == nil {
					_, probeErr := client.DoRequest(callCtx, operation, connector.Request{Body: probeBody, Headers: dispatch.profile.ExtraHeaders})
					if automaticRenderOverflow(probeErr) {
						return nil, overflowBudgetError("The provider-rendered input leaves no room for even one output token")
					}
				}
			}
		}
		return nil, automaticOutputUnsupported("an available vLLM Chat render API (enable --enable-scale-out on supported vLLM versions)")
	}
	var result automaticRenderResult
	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, automaticOutputUnsupported("a valid vLLM Chat render response")
	}
	return &result, nil
}

func automaticRenderOverflow(err error) bool {
	var transport *connector.Error
	if !errors.As(err, &transport) || transport.Kind != connector.KindStatus || transport.StatusCode != http.StatusBadRequest {
		return false
	}
	body, truncated := transport.ResponseBody()
	if truncated {
		return false
	}
	var result struct {
		Error struct {
			Param string `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &result) != nil {
		return false
	}
	return result.Error.Param == "input_tokens" || result.Error.Param == "input_text"
}
