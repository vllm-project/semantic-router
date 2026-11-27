package extproc

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestFinalAdmissionExplicitCompressionParity(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			for _, tc := range []struct {
				name, prompt               string
				narrowWindow, reserve, rag int
				fits, compresses           bool
			}{
				{"preserved_rag_over_budget", "What are the refund terms?", 1500, 8000, 290, false, false},
				{"compressible_user", strings.Repeat("Refund terms apply. ", 170), 1500, 8000, 210, true, true},
				{"discard_narrow_compression", strings.Repeat("x ", 900) + "MIDDLE_VALUE " + strings.Repeat("x ", 899) + "TAIL_VALUE", 3500, 0, 2, true, false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var directPrompt string
					for _, route := range []string{"direct", "reselection", "fallback"} {
						t.Run(route, func(t *testing.T) {
							refs, smallWindow := finalAdmissionModelPair, tc.narrowWindow
							switch route {
							case "direct":
								refs = "{model: wide}"
							case "fallback":
								smallWindow = 40000
							}
							g, lookups := newCompressionParityGateway(t, adapter, refs, smallWindow, tc.reserve, tc.rag, route == "fallback")
							detector := &stubPIIDetector{found: map[string][]string{"redact": {"x", "MIDDLE_VALUE", "TAIL_VALUE"}}}
							useMaskingDetector(t, detector)
							if route == "fallback" {
								g.small.status = http.StatusServiceUnavailable
							}
							compressions := metrics.PluginExecutionTotal.WithLabelValues("context_compression", config.RoutingDecisionKey("default", "redact"), "applied")
							before := testutil.ToFloat64(compressions)
							response := g.post("/v1/chat/completions", finalAdmissionChatBody(tc.prompt), "")
							require.EqualValues(t, 1, lookups.Load())
							if tc.name == "discard_narrow_compression" {
								wantScans := 2
								if route == "reselection" {
									wantScans++ // The two user prompts differ. The RAG block is unchanged.
								}
								require.Equal(t, wantScans, detector.calls)
								require.Equal(t, before, testutil.ToFloat64(compressions))
							}
							small, wide := g.small.ownCalls(), g.wide.ownCalls()
							if route == "fallback" {
								require.Len(t, small, 1)
							} else {
								require.Empty(t, small)
							}
							if !tc.fits {
								require.Empty(t, wide)
								if route == "fallback" {
									require.Equal(t, http.StatusServiceUnavailable, response.Code, response.Body.String())
								} else {
									require.Equal(t, http.StatusBadRequest, response.Code, response.Body.String())
									require.Contains(t, response.Body.String(), "context_length_exceeded")
								}
								return
							}
							require.Equal(t, http.StatusOK, response.Code, response.Body.String())
							require.Len(t, wide, 1)
							require.Equal(t, "wide", wide[0].Model)
							require.Equal(t, llmprotocol.Int64(16), wide[0].Sampling.MaxOutputTokens)
							prompt := dispatchedUserText(wide[0])
							require.NotEmpty(t, prompt)
							if route == "direct" {
								if tc.compresses {
									require.Less(t, len(prompt), len(tc.prompt))
								} else {
									masked := strings.NewReplacer("x", "[MASKED_VALUE_0]", "MIDDLE_VALUE", "[MASKED_VALUE_1]", "TAIL_VALUE", "[MASKED_VALUE_2]")
									require.Equal(t, masked.Replace(tc.prompt), prompt)
								}
								directPrompt = prompt
							} else {
								require.Len(t, prompt, len(directPrompt))
								require.Equal(t, directPrompt, prompt)
							}
						})
					}
				})
			}
			t.Run("fallback_cannot_dispatch_excluded_model", func(t *testing.T) {
				g, lookups := newCompressionParityGateway(t, adapter, finalAdmissionModelPair, 1500, 1500, 2, true)
				g.wide.status = http.StatusServiceUnavailable
				response := g.post("/v1/chat/completions", finalAdmissionChatBody("What are the refund terms?"), "")
				require.EqualValues(t, 1, lookups.Load())
				require.Len(t, g.wide.ownCalls(), 1)
				require.Empty(t, g.small.ownCalls(), "the reserve exceeds the smaller model's input budget")
				require.Equal(t, http.StatusServiceUnavailable, response.Code, response.Body.String())
			})
		})
	}
}

func TestFinalAdmissionSkipsCandidateWithoutCompressionLimits(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			for _, refs := range []string{finalAdmissionModelPair, "{model: small}"} {
				t.Run(refs, func(t *testing.T) {
					g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, refs, func(document string) string {
						document = strings.Replace(document, "context_window_size: 96", "context_window_size: 1700", 1)
						document = strings.Replace(document, "context_window_size: 4096", "context_window_size: 4096\n      max_output_tokens: 512", 1)
						return strings.Replace(document, "entrypoints:", `          - type: context
            configuration:
              enabled: true
              compress:
                match: {recall: {method: bm25, top_k: 3}}
                targets: {current_user: {mode: truncate}}
entrypoints:`, 1)
					})
					response := g.post("/v1/chat/completions", finalAdmissionChatBody("hello"), "")
					require.Empty(t, g.small.ownCalls())
					if refs == finalAdmissionModelPair {
						require.Equal(t, http.StatusOK, response.Code, response.Body.String())
						require.Len(t, g.wide.ownCalls(), 1)
						return
					}
					require.Equal(t, http.StatusServiceUnavailable, response.Code, response.Body.String())
					require.Contains(t, response.Body.String(), "no_eligible_model")
					require.Empty(t, g.wide.ownCalls())
				})
			}
		})
	}
}

func newCompressionParityGateway(t *testing.T, adapter func(*OpenAIRouter) routing.Engine, refs string, smallWindow, reserve, repeats int, fallback bool) (*finalAdmissionGateway, *atomic.Int32) {
	t.Helper()
	var lookups atomic.Int32
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		lookups.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"content": strings.Repeat("Audit report confirms the office opens Monday. ", repeats),
		})
	}))
	t.Cleanup(rag.Close)
	g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, refs, func(document string) string {
		document = strings.Replace(document, "context_window_size: 96", fmt.Sprintf("context_window_size: %d\n      max_output_tokens: 512", smallWindow), 1)
		document = strings.Replace(document, "context_window_size: 4096", "context_window_size: 20000\n      max_output_tokens: 512", 1)
		if fallback {
			document = enableParityFallback(document)
		}
		return strings.Replace(document, "entrypoints:", fmt.Sprintf(`          - type: context
            configuration:
              enabled: true
              sources:
                - type: rag
                  backend: external_api
                  max_context_length: 24000
                  on_failure: block
                  match: {recall: {method: embedding, top_k: 3}}
                  backend_config:
                    endpoint: "%s"
                    request_format: custom
                    request_template: '{"query":"{{.Query}}"}'
              compress:
                match: {recall: {method: bm25, top_k: 3}}
                targets:
                  current_user: {mode: truncate}
                  rag: {mode: preserve}
              budget: {reserve_output_tokens: %d}
entrypoints:`, rag.URL, reserve), 1)
	})
	g.router.Embeddings = ragBlockEmbeddings()
	return g, &lookups
}

func enableParityFallback(document string) string {
	return strings.Replace(document, "        plugins:", "        fallback: {enabled: true, max_attempts: 2}\n        plugins:", 1)
}

func dispatchedUserText(request llmprotocol.Request) string {
	var prompt strings.Builder
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleUser {
			for _, content := range message.Content {
				prompt.WriteString(content.Text)
			}
		}
	}
	return prompt.String()
}

func TestFinalAdmissionChecksConfiguredReserveAfterMasking(t *testing.T) {
	prompt := "HEAD\n" + strings.Repeat("x ", 2600) + "\nTAIL"
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			for _, tc := range []struct {
				name, refs string
				reserve    int
			}{
				{"direct", "{model: wide}", 8000},
				{"reselection", finalAdmissionModelPair, 8000},
				{"without_configured_reserve", "{model: wide}", 0},
			} {
				t.Run(tc.name, func(t *testing.T) {
					g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, tc.refs, func(document string) string {
						document = strings.Replace(document, "context_window_size: 96", "context_window_size: 8000\n      max_output_tokens: 512", 1)
						document = strings.Replace(document, "context_window_size: 4096", "context_window_size: 20000\n      max_output_tokens: 512", 1)
						return strings.Replace(document, "entrypoints:", fmt.Sprintf(`          - type: context
            configuration:
              enabled: true
              compress:
                match: {recall: {method: bm25, top_k: 3}}
                targets: {current_user: {mode: truncate}}
              budget: {reserve_output_tokens: %d}
entrypoints:`, tc.reserve), 1)
					})
					response := g.post("/v1/chat/completions", finalAdmissionChatBody(prompt), "")
					require.Empty(t, g.small.ownCalls())
					wide := g.wide.ownCalls()
					if tc.reserve != 0 {
						require.Equal(t, http.StatusBadRequest, response.Code, response.Body.String())
						require.Contains(t, response.Body.String(), "context_length_exceeded")
						require.Empty(t, wide)
						return
					}
					require.Equal(t, http.StatusOK, response.Code, response.Body.String())
					require.Len(t, wide, 1)
					require.Equal(t, strings.ReplaceAll(prompt, "x", "[MASKED_VALUE_0]"), dispatchedUserText(wide[0]))
					require.Equal(t, llmprotocol.Int64(16), wide[0].Sampling.MaxOutputTokens)
				})
			}
		})
	}
}

func TestFinalAdmissionAutomaticCompressionParity(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			var direct llmprotocol.Request
			for _, tc := range []struct {
				name, refs              string
				smallWindow, wideWindow int
				fallback, wideCompress  bool
			}{
				{"direct", "{model: wide}", 1100, 2400, false, true},
				{"reselection", finalAdmissionModelPair, 1100, 2400, false, true},
				{"fallback_compresses", finalAdmissionModelPair, 6000, 2400, true, true},
				{"wide_direct", "{model: wide}", 2400, 6000, false, false},
				{"fallback_restores_prompt", finalAdmissionModelPair, 2400, 6000, true, false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					smallCalls := make(chan llmprotocol.Request, 3)
					wideCalls := make(chan llmprotocol.Request, 3)
					backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
						body, err := io.ReadAll(req.Body)
						if err != nil {
							t.Error(err)
							w.WriteHeader(http.StatusInternalServerError)
							return
						}
						request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body)
						if err != nil {
							t.Error(err)
							w.WriteHeader(http.StatusBadRequest)
							return
						}
						if strings.HasSuffix(req.URL.Path, "/render") {
							window := tc.smallWindow
							if request.Model == "wide" {
								window = tc.wideWindow
							}
							tokens := 17 + len(dispatchedUserText(request))
							if tokens >= window {
								w.WriteHeader(http.StatusBadRequest)
								_, _ = fmt.Fprint(w, `{"error":{"param":"input_tokens"}}`)
								return
							}
							_ = json.NewEncoder(w).Encode(map[string]any{
								"model": request.Model, "token_ids": make([]int, tokens),
								"sampling_params": map[string]int{"max_tokens": window - tokens},
							})
							return
						}
						if request.Model == "small" {
							smallCalls <- request
							w.WriteHeader(http.StatusServiceUnavailable)
							_, _ = w.Write(extProcTransportErrorFixture(llmprotocol.OpenAIChatV1))
							return
						}
						wideCalls <- request
						_, _ = fmt.Fprint(w, `{"id":"resp","object":"chat.completion","created":1700000000,"model":"wide","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`)
					}))
					t.Cleanup(backend.Close)

					g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, tc.refs, func(document string) string {
						document = strings.Replace(document, "context_window_size: 96", fmt.Sprintf("context_window_size: %d\n      max_output_tokens: 512", tc.smallWindow), 1)
						document = strings.Replace(document, "context_window_size: 4096", fmt.Sprintf("context_window_size: %d\n      max_output_tokens: 512", tc.wideWindow), 1)
						document = regexp.MustCompile(`endpoint: "[^"]+"`).ReplaceAllString(document, fmt.Sprintf("endpoint: %q", strings.TrimPrefix(backend.URL, "http://")))
						if tc.fallback {
							document = enableParityFallback(document)
						}
						return strings.Replace(document, "entrypoints:", `          - type: request_params
            configuration: {default_max_tokens: auto}
          - type: context
            configuration:
              enabled: true
              compress:
                match: {recall: {method: bm25, top_k: 3}}
                targets: {current_user: {mode: truncate}}
entrypoints:`, 1)
					})
					prompt := "HEAD\n" + strings.Repeat("x ", 210) + "\nTAIL"
					response := g.post("/v1/chat/completions", fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":%q}]}`, prompt), "")
					require.Equal(t, http.StatusOK, response.Code, response.Body.String())
					if tc.fallback {
						require.Len(t, smallCalls, 1)
						if !tc.wideCompress {
							require.Contains(t, dispatchedUserText(<-smallCalls), "context omitted by route compression")
						}
					} else {
						require.Empty(t, smallCalls)
					}
					require.Len(t, wideCalls, 1)
					request := <-wideCalls
					sentPrompt := dispatchedUserText(request)
					if tc.wideCompress {
						require.Contains(t, sentPrompt, "context omitted by route compression")
						require.Contains(t, sentPrompt, "[MASKED_VALUE_0]")
					} else {
						require.Equal(t, strings.ReplaceAll(prompt, "x", "[MASKED_VALUE_0]"), sentPrompt)
					}
					wantLimit := min(512, tc.wideWindow-17-len(sentPrompt))
					require.Positive(t, wantLimit)
					require.Equal(t, llmprotocol.Int64(int64(wantLimit)), request.Sampling.MaxOutputTokens)
					if tc.refs == "{model: wide}" {
						direct = request
					}
					require.Equal(t, dispatchedUserText(direct), dispatchedUserText(request))
					require.Equal(t, direct.Sampling.MaxOutputTokens, request.Sampling.MaxOutputTokens)
				})
			}
		})
	}
}
