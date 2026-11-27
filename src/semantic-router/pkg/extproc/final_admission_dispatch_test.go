package extproc

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	rls "github.com/envoyproxy/go-control-plane/envoy/service/ratelimit/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const finalAdmissionModelPair = "{model: small}, {model: wide}"

type finalAdmissionGateway struct {
	router                 *OpenAIRouter
	config                 *config.RouterConfig
	handler                http.Handler
	small, wide, preferred *umbrellaBackend
}

func newFinalAdmissionGateway(t *testing.T, adapter func(*OpenAIRouter) routing.Engine, format llmprotocol.WireFormat, refs string, customize func(string) string) *finalAdmissionGateway {
	t.Helper()
	small, first := newUmbrellaBackend(t, format)
	wide, second := newUmbrellaBackend(t, format)
	preferred, third := newUmbrellaBackend(t, format)
	document := fmt.Sprintf(finalAdmissionGatewayConfig, strings.TrimPrefix(first.URL, "http://"), strings.TrimPrefix(second.URL, "http://"), strings.TrimPrefix(third.URL, "http://"), refs, "{type: static}")
	if format == llmprotocol.AnthropicMessagesV1 {
		document = strings.ReplaceAll(document, "      context_window_size:", "      api_format: anthropic\n      context_window_size:")
		document = strings.ReplaceAll(document, "provider: vllm", "provider: anthropic")
	}
	if customize != nil {
		document = customize(document)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	router := newParityRouter(t, cfg)
	pools, err := upstream.Build(cfg, upstream.Options{})
	require.NoError(t, err)
	t.Cleanup(func() { _ = pools.Close(context.Background()) })
	handler, err := gateway.NewHandler(gateway.Options{Listener: "public", Serving: gateway.Static(gateway.Serving{
		Engine: adapter(router), Upstream: pools, TrustIdentity: gateway.IdentityTrust{Headers: true},
	})})
	require.NoError(t, err)
	useMaskingDetector(t, &stubPIIDetector{found: map[string][]string{"redact": {"x"}}})
	return &finalAdmissionGateway{router, cfg, handler, small, wide, preferred}
}

func (g *finalAdmissionGateway) post(path, body, user string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if user != "" {
		req.Header.Set(g.config.Authz.Identity.GetUserIDHeader(), user)
		req.Header.Set(g.config.Authz.Identity.GetUserGroupsHeader(), "shared")
	}
	recorder := httptest.NewRecorder()
	g.handler.ServeHTTP(recorder, req)
	return recorder
}

func finalAdmissionChatBody(prompt string) string {
	return fmt.Sprintf(`{"model":"auto","max_tokens":16,"messages":[{"role":"user","content":%q}]}`, prompt)
}

func TestFinalAdmissionModelDenialRefundsRequestQuota(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			for _, rule := range []ratelimit.Rule{
				{Name: "request RPM", RequestsPerUnit: 1, Unit: time.Minute},
				{Name: "request TPM", TokensPerUnit: 48, Unit: time.Minute},
			} {
				t.Run(rule.Name, func(t *testing.T) {
					g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair, nil)
					g.router.RateLimiter = ratelimit.NewRateLimitResolver(ratelimit.NewLocalLimiter([]ratelimit.Rule{
						rule, {Name: "model", Match: ratelimit.RuleMatch{Model: "small"}, TokensPerUnit: 16, Unit: time.Minute},
					}))
					first := g.post("/v1/chat/completions", finalAdmissionChatBody(strings.Repeat("hello ", 27)), "")
					require.Equal(t, 429, first.Code)
					require.Equal(t, "16", first.Header().Get("x-ratelimit-limit"))
					require.Empty(t, g.small.ownCalls())
					followUp := g.post("/v1/chat/completions", finalAdmissionChatBody("hi"), "")
					require.Equal(t, 200, followUp.Code, followUp.Body.String())
					require.Len(t, g.small.ownCalls(), 1)
				})
			}
		})
	}
}

func TestFinalAdmissionModelQuotaPrecedesRetrieval(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			var lookups atomic.Int32
			retrieval := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				lookups.Add(1)
				w.WriteHeader(503)
			}))
			t.Cleanup(retrieval.Close)
			g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair, func(document string) string {
				return strings.Replace(document, "entrypoints:", fmt.Sprintf(`          - type: context
            configuration:
              enabled: true
              sources:
                - type: rag
                  backend: external_api
                  on_failure: block
                  match: {recall: {method: embedding, top_k: 3}}
                  backend_config:
                    endpoint: "%s"
                    request_format: custom
                    request_template: '{"query":"{{.Query}}"}'
entrypoints:`, retrieval.URL), 1)
			})
			g.router.Embeddings = ragBlockEmbeddings()
			g.router.RateLimiter = ratelimit.NewRateLimitResolver(ratelimit.NewLocalLimiter([]ratelimit.Rule{{
				Name: "model", Match: ratelimit.RuleMatch{Model: "small"}, RequestsPerUnit: 1, Unit: time.Minute,
			}}))
			used, err := g.router.RateLimiter.Check(ratelimit.Context{Model: "small"})
			require.NoError(t, err)
			require.True(t, used.Allowed)
			response := g.post("/v1/chat/completions", finalAdmissionChatBody("What are the refund terms?"), "")
			require.Equal(t, 429, response.Code)
			require.Zero(t, lookups.Load())
			require.Empty(t, g.small.ownCalls())
		})
	}
}

type combinedAdmissionRLS struct {
	rls.UnimplementedRateLimitServiceServer
	mu        sync.Mutex
	requests  []*rls.RateLimitRequest
	groupHits int
}

func (s *combinedAdmissionRLS) ShouldRateLimit(_ context.Context, request *rls.RateLimitRequest) (*rls.RateLimitResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, request)
	group, blockedModel := false, false
	for _, descriptor := range request.GetDescriptors() {
		for _, entry := range descriptor.GetEntries() {
			group = group || entry.GetKey() == "group"
			blockedModel = blockedModel || entry.GetKey() == "user_id" && entry.GetValue() == "at-model-limit"
		}
	}
	if blockedModel || group && s.groupHits > 0 {
		return &rls.RateLimitResponse{OverallCode: rls.RateLimitResponse_OVER_LIMIT}, nil
	}
	if group {
		s.groupHits++
	}
	return &rls.RateLimitResponse{OverallCode: rls.RateLimitResponse_OK}, nil
}

func TestFinalAdmissionEnvoyChecksOneDescriptorSetForWinner(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			server := grpc.NewServer()
			service := &combinedAdmissionRLS{}
			rls.RegisterRateLimitServiceServer(server, service)
			go func() { _ = server.Serve(listener) }()
			t.Cleanup(server.Stop)
			provider, err := ratelimit.NewEnvoyRLSProvider(listener.Addr().String(), "test")
			require.NoError(t, err)
			t.Cleanup(func() { _ = provider.Close() })
			g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair, nil)
			g.router.RateLimiter = ratelimit.NewRateLimitResolver(provider)
			body := finalAdmissionChatBody(strings.Repeat("x ", 30))
			require.Equal(t, 429, g.post("/v1/chat/completions", body, "at-model-limit").Code)
			require.Empty(t, g.wide.ownCalls())
			require.Equal(t, 200, g.post("/v1/chat/completions", body, "with-quota").Code)
			require.Len(t, g.wide.ownCalls(), 1)
			service.mu.Lock()
			defer service.mu.Unlock()
			require.Len(t, service.requests, 2)
			for _, request := range service.requests {
				require.Len(t, request.GetDescriptors(), 2)
				require.Equal(t, "wide", request.GetDescriptors()[0].GetEntries()[1].GetValue())
				require.Equal(t, "shared", request.GetDescriptors()[1].GetEntries()[0].GetValue())
			}
		})
	}
}

func TestFinalAdmissionReselectionPreservesRequestParams(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			t.Run("blocked effort", func(t *testing.T) {
				g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair, func(document string) string {
					return strings.Replace(document, "entrypoints:", `          - type: request_params
            configuration: {blocked_params: [reasoning_effort]}
entrypoints:`, 1)
				})
				body := fmt.Sprintf(`{"model":"auto","max_tokens":16,"reasoning_effort":"low","messages":[{"role":"user","content":%q}]}`, strings.Repeat("x ", 30))
				response := g.post("/v1/chat/completions", body, "")
				require.Equal(t, 200, response.Code, response.Body.String())
				require.Empty(t, g.small.ownCalls())
				calls := g.wide.ownCalls()
				require.Len(t, calls, 1)
				require.Empty(t, calls[0].ReasoningEffort)
			})
			t.Run("capped thinking", func(t *testing.T) {
				g := newFinalAdmissionGateway(t, adapter, llmprotocol.AnthropicMessagesV1, finalAdmissionModelPair, func(document string) string {
					document = strings.Replace(document, "context_window_size: 96", "context_window_size: 4175", 1)
					document = strings.Replace(document, "context_window_size: 4096", "context_window_size: 16000", 1)
					return strings.Replace(document, "entrypoints:", `          - type: request_params
            configuration: {max_tokens_limit: 4096}
entrypoints:`, 1)
				})
				body := fmt.Sprintf(`{"model":"auto","max_tokens":6000,"thinking":{"type":"enabled","budget_tokens":5000},"messages":[{"role":"user","content":%q}]}`, strings.Repeat("x ", 30))
				response := g.post("/v1/messages", body, "")
				require.Equal(t, 200, response.Code, response.Body.String())
				require.Empty(t, g.small.ownCalls())
				calls := g.wide.ownCalls()
				require.Len(t, calls, 1)
				require.EqualValues(t, 4096, *calls[0].Sampling.MaxOutputTokens)
				require.EqualValues(t, 4095, *calls[0].ReasoningBudgetTokens)
			})
		})
	}
}

func TestFinalAdmissionReselectionStartsFallbackFromWinner(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair+", {model: preferred}", func(document string) string {
				return strings.Replace(document, "        plugins:", "        fallback: {enabled: true, max_attempts: 2}\n        plugins:", 1)
			})
			g.wide.status = 503
			response := g.post("/v1/chat/completions", finalAdmissionChatBody(strings.Repeat("x ", 30)), "")
			require.Equal(t, 200, response.Code, response.Body.String())
			require.Empty(t, g.small.ownCalls())
			require.Len(t, g.wide.ownCalls(), 1)
			require.Len(t, g.preferred.ownCalls(), 1)
		})
	}
}

func TestFinalAdmissionReselectionReusesExactCache(t *testing.T) {
	for mode, adapter := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			g := newFinalAdmissionGateway(t, adapter, llmprotocol.OpenAIChatV1, finalAdmissionModelPair, func(document string) string {
				return strings.Replace(document, "entrypoints:", `          - type: response_cache
            configuration: {enabled: true, mode: exact, scope: global}
entrypoints:`, 1)
			})
			backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, MaxEntries: 20, TTLSeconds: 60})
			g.router.Cache = backend
			g.router.ResponseCache = cache.NewResponseCacheService(cache.NewLegacyBackendAdapter(backend, cache.InMemoryCacheType), cache.DefaultResponseCacheServiceOptions())
			body := finalAdmissionChatBody(strings.Repeat("x ", 30))
			for range 2 {
				response := g.post("/v1/chat/completions", body, "")
				require.Equal(t, 200, response.Code, response.Body.String())
				require.Contains(t, response.Body.String(), `"model":"wide"`)
			}
			require.Empty(t, g.small.ownCalls())
			require.Len(t, g.wide.ownCalls(), 1)
		})
	}
}
