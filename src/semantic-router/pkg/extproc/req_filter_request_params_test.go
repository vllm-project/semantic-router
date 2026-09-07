//go:build !windows && cgo

package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestApplySemanticRequestParamsNilDecision(t *testing.T) {
	r := &OpenAIRouter{}
	request := llmprotocol.Request{Model: "x"}
	changed, err := r.applySemanticRequestParams(nil, &request, config.DefaultRecipeName)
	if err != nil || changed || request.Model != "x" {
		t.Fatalf("nil decision changed request: changed=%v request=%+v err=%v", changed, request, err)
	}
}

func TestApplySemanticRequestParamsBlocksAndCapsNeutralFields(t *testing.T) {
	r := &OpenAIRouter{}
	payload, err := config.NewStructuredPayload(map[string]interface{}{
		"blocked_params":   []string{"frequency_penalty", "custom_evil_field"},
		"max_tokens_limit": 500,
		"max_n":            1,
		"strip_unknown":    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "tier_a",
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: payload},
		},
	}
	request := llmprotocol.Request{
		Model:          "m",
		CandidateCount: llmprotocol.Int64(5),
		Sampling: llmprotocol.Sampling{
			MaxOutputTokens:  llmprotocol.Int64(9000),
			FrequencyPenalty: llmprotocol.Float64(0.5),
		},
	}
	changed, err := r.applySemanticRequestParams(decision, &request, config.DefaultRecipeName)
	if err != nil {
		t.Fatal(err)
	}
	if !changed || request.Sampling.FrequencyPenalty != nil || request.Sampling.MaxOutputTokens == nil ||
		*request.Sampling.MaxOutputTokens != 500 || request.CandidateCount == nil || *request.CandidateCount != 1 {
		t.Fatalf("semantic request params = %+v, changed=%v", request, changed)
	}
}

func TestProtocolCodecRejectsUnknownFieldsBeforeRequestParamPlugins(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	_, _, _, err := engine.DecodeRequest(
		llmprotocol.OpenAIChatV1,
		[]byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"future_field":true}`),
	)
	if err == nil {
		t.Fatal("unknown wire field reached semantic request plugins")
	}
}
