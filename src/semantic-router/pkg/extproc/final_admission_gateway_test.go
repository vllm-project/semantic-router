package extproc

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const finalAdmissionGatewayConfig = `version: v0.5
listeners: [{name: public, address: 127.0.0.1, port: 8899}]
providers:
  defaults: {model: small}
  models:
    - name: small
      context_window_size: 96
      pricing: {currency: USD, prompt_per_1m: 1, completion_per_1m: 1}
      backend_refs: [{name: primary, endpoint: "%s", protocol: http, provider: vllm}]
    - name: wide
      context_window_size: 4096
      pricing: {currency: USD, prompt_per_1m: 3, completion_per_1m: 3}
      backend_refs: [{name: alternate, endpoint: "%s", protocol: http, provider: vllm}]
    - name: preferred
      context_window_size: 4096
      pricing: {currency: USD, prompt_per_1m: 2, completion_per_1m: 2}
      backend_refs: [{name: preferred, endpoint: "%s", protocol: http, provider: vllm}]
recipes:
  - name: default
    signals:
      keywords:
        - {name: redact, operator: OR, method: regex, keywords: ["x"]}
    decisions:
      - name: redact
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [%s]
        algorithm: %s
        plugins:
          - type: masking
            configuration:
              enabled: true
              rules: [{type: keyword, name: redact}]
              placeholders: {redact: "[MASKED_VALUE_{index}]"}
entrypoints: [{model_names: [auto], recipe: default, default: true}]
`

func TestFinalAdmissionAtGateway(t *testing.T) {
	small, first := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
	wide, second := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
	preferred, third := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
	const static = `{type: static}`
	const multiFactor = `{type: multi_factor, multi_factor: {weights: {quality: 0, latency: 0, cost: 1, load: 0}}}`
	const pair = `{model: small}, {model: wide}`
	for _, test := range []struct {
		name, refs, algorithm string
		winner                string
		limit                 *ratelimit.RuleMatch
	}{
		{"only small is eligible", "{model: small}", static, "", nil},
		{"same decision has room", pair, static, "wide", nil},
		{"uses selector scores", pair + ", {model: preferred}", multiFactor, "preferred", nil},
		{"request quota precedes masking", pair, static, "", &ratelimit.RuleMatch{}},
		{"model quota checks the winner", pair, static, "", &ratelimit.RuleMatch{Model: "wide"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(finalAdmissionGatewayConfig, strings.TrimPrefix(first.URL, "http://"), strings.TrimPrefix(second.URL, "http://"), strings.TrimPrefix(third.URL, "http://"), test.refs, test.algorithm)))
			require.NoError(t, err)
			pools, err := upstream.Build(cfg, upstream.Options{})
			require.NoError(t, err)
			t.Cleanup(func() { _ = pools.Close(context.Background()) })
			router := newParityRouter(t, cfg)
			detector := &stubPIIDetector{found: map[string][]string{"redact": {"x"}}}
			useMaskingDetector(t, detector)
			if test.limit != nil {
				router.RateLimiter = ratelimit.NewRateLimitResolver(ratelimit.NewLocalLimiter([]ratelimit.Rule{{
					Name: "exhausted", Match: *test.limit, RequestsPerUnit: 1, Unit: time.Hour,
				}}))
				model := firstNonEmpty(test.limit.Model, "small")
				decision, checkErr := router.RateLimiter.Check(ratelimit.Context{Model: model})
				require.NoError(t, checkErr)
				require.True(t, decision.Allowed)
			}
			for mode, adapter := range looperAdapters {
				t.Run(mode, func(t *testing.T) {
					handler, err := gateway.NewHandler(gateway.Options{Listener: "public", Serving: gateway.Static(gateway.Serving{Engine: adapter(router), Upstream: pools})})
					require.NoError(t, err)
					for _, wire := range []struct{ path, body string }{
						{"/v1/chat/completions", `{"model":"auto","max_tokens":16,"messages":[{"role":"user","content":"%s"}]}`},
						{"/v1/messages", `{"model":"auto","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"%s"}]}`},
						{"/v1/responses", `{"model":"auto","max_output_tokens":16,"input":"%s"}`},
					} {
						req := httptest.NewRequest(http.MethodPost, wire.path, strings.NewReader(fmt.Sprintf(wire.body, strings.Repeat("x ", 30))))
						req.Header.Set("Content-Type", "application/json")
						w := httptest.NewRecorder()
						beforeMasking := detector.calls
						handler.ServeHTTP(w, req)
						require.Empty(t, small.ownCalls(), wire.path)
						calls := append(wide.ownCalls(), preferred.ownCalls()...)
						if test.limit != nil {
							require.Equal(t, http.StatusTooManyRequests, w.Code, "%s: %s", wire.path, w.Body)
							require.Empty(t, calls)
							if test.limit.Model == "" {
								require.Equal(t, beforeMasking, detector.calls)
							} else {
								require.Greater(t, detector.calls, beforeMasking)
							}
							continue
						}
						if test.winner == "" {
							require.Equal(t, 400, w.Code, "%s: %s", wire.path, w.Body)
							require.Empty(t, calls)
							if wire.path == "/v1/messages" {
								require.Contains(t, w.Body.String(), "invalid_request_error")
							} else {
								require.Contains(t, w.Body.String(), "context_length_exceeded")
							}
							continue
						}
						require.Equal(t, 200, w.Code, "%s: %s", wire.path, w.Body)
						require.Len(t, calls, 1)
						require.Equal(t, test.winner, calls[0].Model)
						require.Contains(t, calls[0].Messages[0].Content[0].Text, "[MASKED_VALUE_0]")
					}
				})
			}
		})
	}
}
