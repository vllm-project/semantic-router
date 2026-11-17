package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type maskingKeywordSpans struct {
	classifier *classification.KeywordClassifier
}

func (s maskingKeywordSpans) RuleSpans(_ context.Context, _, rule, text string) ([]modelservice.Span, error) {
	return s.classifier.RuleSpans(rule, text)
}

// Configuration validation accepts these different templates. The real keyword
// matcher and final dispatch must still keep their values distinct on the wire.
func TestMaskingDispatchIntersectingTemplatesKeepValuesDistinct(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  models:
    - name: m
      backend_refs: [{endpoint: "127.0.0.1:8000", provider: vllm}]
entrypoints:
  - {model_names: ["vllm-sr/masking"], recipe: default}
recipes:
  - name: default
    signals:
      keywords:
        - {name: first, operator: OR, keywords: [alpha]}
        - {name: second, operator: OR, keywords: [beta]}
    decisions:
      - name: mask
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
        plugins:
          - type: masking
            configuration:
              enabled: true
              rules: [{type: keyword, name: first}, {type: keyword, name: second}]
              placeholders: {first: "[V{index}X0]", second: "[V0X{index}]"}
`))
	if err != nil {
		t.Fatal(err)
	}
	classifier, err := classification.NewKeywordClassifier(cfg.KeywordRules)
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			router := &OpenAIRouter{Config: cfg}
			useMaskingDetector(t, maskingKeywordSpans{classifier: classifier})
			body := `{"model":"vllm-sr/masking","max_tokens":64,"messages":[{"role":"user","content":"alpha beta alpha beta"}]}`
			ctx := newMaskingContextOf(t, router, target, &cfg.Decisions[0], body)
			encoded, err := finalizeMaskingDispatch(router, ctx)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			dispatched, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(target, encoded)
			if err != nil {
				t.Fatalf("invalid %s request: %v", target, err)
			}
			if got, want := dispatched.Messages[0].Content[0].Text, "[V0X0] [V0X1] [V0X0] [V0X1]"; got != want {
				t.Fatalf("provider receives %q, want %q", got, want)
			}
		})
	}
}
