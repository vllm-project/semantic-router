package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// supportedStickyToolsConfig is the tools policy a supported sticky decision
// declares: authoritative operator policy for candidate selection and final
// emission.
func supportedStickyToolsConfig() *config.ToolsPluginConfig {
	return &config.ToolsPluginConfig{
		Enabled: true,
		Mode:    config.ToolsPluginModePassthrough,
		TrustedFacts: &config.TrustedFactsConfig{
			Enabled:      true,
			Enforcement:  config.TrustedEnforcementAuthoritative,
			TrustSources: []string{config.TrustedSourceOperatorPolicy},
			StageRoles:   []string{config.TrustedStageCandidate, config.TrustedStageFinal},
		},
	}
}

func supportedStickyDecision(t *testing.T, name string, mode string) config.Decision {
	t.Helper()
	return stickyDecision(t, name, &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: mode, TopK: 1,
		Sticky: &config.StickyToolSelectionConfig{Enabled: true},
	}, supportedStickyToolsConfig())
}

func stickyDecision(t *testing.T, name string, selection *config.ToolSelectionPluginConfig, toolsCfg *config.ToolsPluginConfig) config.Decision {
	t.Helper()
	plugins := []config.DecisionPlugin{mustToolSelectionDecisionPlugin(t, selection)}
	if toolsCfg != nil {
		plugins = append(plugins, mustToolsDecisionPlugin(t, toolsCfg))
	}
	return config.Decision{Name: name, Plugins: plugins}
}

// stickyEmbeddingProvider embeds known texts to fixed unit vectors, so tests
// control relevance exactly. Unknown texts get a weak shared direction, and
// a text listed in fail returns an error.
type stickyEmbeddingProvider struct {
	vectors map[string][]float32
	fail    map[string]bool
}

func (p *stickyEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if p.fail[text] {
		return nil, errors.New("embedding unavailable")
	}
	if vector, ok := p.vectors[text]; ok {
		return vector, nil
	}
	return []float32{0.01, 0.01, 0.01, 0.01}, nil
}

func (p *stickyEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func (p *stickyEmbeddingProvider) Dimension() int  { return 4 }
func (p *stickyEmbeddingProvider) Backend() string { return config.EmbeddingBackendOpenAICompatible }

// Each query is relevant to exactly one tool.
var stickyTestQueries = map[string]string{
	"weather":  "what is the weather",
	"calendar": "schedule a meeting",
	"email":    "send an email",
	"files":    "list my files",
}

func newStickyEmbeddingProvider() *stickyEmbeddingProvider {
	p := &stickyEmbeddingProvider{vectors: map[string][]float32{}, fail: map[string]bool{}}
	for i, name := range []string{"weather", "calendar", "email", "files"} {
		vector := make([]float32, 4)
		vector[i] = 1
		p.vectors[name] = vector
		p.vectors[stickyTestQueries[name]] = vector
	}
	return p
}

// countingStickyStore records store traffic and can inject a Load failure.
type countingStickyStore struct {
	sessiontools.Store
	loads    atomic.Int64
	swaps    atomic.Int64
	mu       sync.Mutex
	loadFail error
}

func (s *countingStickyStore) Load(ctx context.Context, key string) (sessiontools.VersionedState, error) {
	s.loads.Add(1)
	s.mu.Lock()
	fail := s.loadFail
	s.mu.Unlock()
	if fail != nil {
		return sessiontools.VersionedState{}, fail
	}
	return s.Store.Load(ctx, key)
}

func (s *countingStickyStore) CompareAndSwap(ctx context.Context, key string, expected uint64, next sessiontools.State, ttl time.Duration, quota sessiontools.QuotaKey) (bool, error) {
	s.swaps.Add(1)
	return s.Store.CompareAndSwap(ctx, key, expected, next, ttl, quota)
}

func (s *countingStickyStore) operations() int64 { return s.loads.Load() + s.swaps.Load() }

// stickyClock is a synthetic clock shared by the store and planner.
type stickyClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *stickyClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *stickyClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// installStickyTestRuntime routes store construction through a counting
// wrapper and the sticky clock through a synthetic one for this test.
func installStickyTestRuntime(t *testing.T) (*stickyClock, *[]*countingStickyStore) {
	t.Helper()
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "sticky-test-secret")
	clock := &stickyClock{now: time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)}
	stores := &[]*countingStickyStore{}
	previousClock, previousStore := stickyToolClock, newStickyToolStore
	stickyToolClock = clock.Now
	newStickyToolStore = func(cfg config.ToolSessionStoreConfig) sessiontools.Store {
		store := &countingStickyStore{Store: previousStore(cfg)}
		*stores = append(*stores, store)
		return store
	}
	t.Cleanup(func() { stickyToolClock, newStickyToolStore = previousClock, previousStore })
	return clock, stores
}

// stickyHarness drives sticky requests through the real entrypoint routing
// path and returns the provider-bound body.
type stickyHarness struct {
	t        *testing.T
	router   *OpenAIRouter
	model    string
	format   llmprotocol.WireFormat
	decision config.Decision
	recipe   string
	provider *stickyEmbeddingProvider
	store    *countingStickyStore
	clock    *stickyClock
}

// stickyCatalogSchema carries an integer above 2^53 so tests can check that
// the exact numeral reaches the provider.
const stickyCatalogSchema = `{"type":"object","properties":{"n":{"type":"integer","minimum":9007199254740993}}}`

func stickyCatalogTool(t *testing.T, name, schema string) openai.ChatCompletionToolParam {
	t.Helper()
	var parameters openai.FunctionParameters
	decoder := json.NewDecoder(strings.NewReader(schema))
	decoder.UseNumber()
	require.NoError(t, decoder.Decode(&parameters))
	return openai.ChatCompletionToolParam{Function: openai.FunctionDefinitionParam{Name: name, Parameters: parameters}}
}

func newStickyToolsDatabase(t *testing.T, provider *stickyEmbeddingProvider, schema string, names ...string) *tools.ToolsDatabase {
	t.Helper()
	db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: true, Provider: provider})
	for _, name := range names {
		require.NoError(t, db.AddTool(stickyCatalogTool(t, name, schema), name, "", nil))
	}
	return db
}

func newStickyHarness(t *testing.T, format llmprotocol.WireFormat, decision config.Decision, storeCfg *config.ToolSessionStoreConfig) *stickyHarness {
	t.Helper()
	clock, stores := installStickyTestRuntime(t)
	router, model := routingTestRouterForFormat(format)
	router.Config.Tools = config.ToolsConfig{Enabled: true, TopK: 1, ToolsDBPath: "tools_db.json"}
	router.Config.ToolSessions = storeCfg
	router.Config.Decisions = []config.Decision{decision}
	provider := newStickyEmbeddingProvider()
	router.ToolsDatabase = newStickyToolsDatabase(t, provider, stickyCatalogSchema, "weather", "calendar", "email", "files")
	router.toolEmbedder = newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "")
	router.ResponseAPIFilter = NewResponseAPIFilter(NewMockResponseStore())
	router.ReplayRecorder = nil
	resources := newResourceScope()
	t.Cleanup(func() { _ = resources.close() })
	runtime, err := buildStickyToolRuntime(router.Config, resources)
	require.NoError(t, err)
	require.NotNil(t, runtime, "a sticky decision must build the generation runtime")
	router.stickyTools = runtime
	router.resources = resources
	return &stickyHarness{
		t: t, router: router, model: model, format: format, decision: decision, recipe: "assistant",
		provider: provider, store: (*stores)[len(*stores)-1], clock: clock,
	}
}

// stickyTurn describes one client request.
type stickyTurn struct {
	query      string
	offered    []string // client tools; filter mode selects among them
	called     []string // prior assistant tool calls in history
	toolChoice string   // "" means auto
	principal  string
	session    string
	provenance SessionProvenance
	recipe     string
}

type stickyTurnResult struct {
	tools    []string
	body     []byte
	receipt  *routerreplay.StickyToolSelectionReceipt
	ctx      *RequestContext
	response bool
	status   int
}

func (h *stickyHarness) run(turn stickyTurn) stickyTurnResult {
	h.t.Helper()
	result, err := h.dispatch(turn)
	require.NoError(h.t, err)
	return result
}

// dispatch sends one turn without failing the test, so concurrent callers
// can report errors from their own goroutines.
func (h *stickyHarness) dispatch(turn stickyTurn) (stickyTurnResult, error) {
	if turn.principal == "" {
		turn.principal = "user-a"
	}
	if turn.session == "" {
		turn.session = "session-1"
	}
	if turn.provenance == "" {
		turn.provenance = SessionProvenanceHeader
	}
	if turn.recipe == "" {
		turn.recipe = h.recipe
	}
	ctx := &RequestContext{SourceFormat: h.format, TraceContext: h.t.Context(), Headers: map[string]string{}}
	request, immediate := h.router.prepareProtocolRequest(stickyRequestBody(h.format, turn), ctx)
	if immediate != nil {
		return stickyTurnResult{}, fmt.Errorf("fixture request did not decode: %v", immediate)
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.RecipeName(turn.recipe)})
	decision := h.decision
	ctx.VSRSelectedDecision = &decision
	ctx.AuthenticatedPrincipal = turn.principal
	ctx.SessionID = turn.session
	ctx.SessionProvenance = turn.provenance
	response, err := h.router.handleModelRouting(request, h.model, decision.Name, entropy.ReasoningDecision{}, h.model, ctx)
	if err != nil {
		return stickyTurnResult{}, err
	}
	result := stickyTurnResult{receipt: ctx.StickyToolReceipt, ctx: ctx}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		result.status = int(immediate.GetStatus().GetCode())
		return result, nil
	}
	result.response = true
	result.body = response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(h.format, result.body)
	if err != nil {
		return result, fmt.Errorf("provider-bound body %s: %w", result.body, err)
	}
	for _, tool := range decoded.Tools {
		result.tools = append(result.tools, tool.Name)
	}
	return result, nil
}

func stickyRequestBody(format llmprotocol.WireFormat, turn stickyTurn) []byte {
	query := turn.query
	choice := turn.toolChoice
	if choice == "" {
		choice = "auto"
	}
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		input := []any{}
		if len(turn.called) > 0 {
			input = append(input, map[string]any{"role": "user", "content": "earlier request"})
			for i, name := range turn.called {
				id := fmt.Sprintf("call_%d", i)
				input = append(input,
					map[string]any{"type": "function_call", "call_id": id, "name": name, "arguments": "{}"},
					map[string]any{"type": "function_call_output", "call_id": id, "output": "done"},
				)
			}
		}
		input = append(input, map[string]any{"role": "user", "content": query})
		body := map[string]any{"model": "public-model", "store": false, "input": input, "tool_choice": responsesToolChoice(choice)}
		if len(turn.offered) > 0 {
			offered := []any{}
			for _, name := range turn.offered {
				offered = append(offered, map[string]any{"type": "function", "name": name, "parameters": json.RawMessage(stickyCatalogSchema)})
			}
			body["tools"] = offered
		}
		return mustJSON(body)
	case llmprotocol.AnthropicMessagesV1:
		messages := []any{}
		userContent := []any{map[string]any{"type": "text", "text": query}}
		if len(turn.called) > 0 {
			uses, results := []any{}, []any{}
			for i, name := range turn.called {
				id := fmt.Sprintf("call_%d", i)
				uses = append(uses, map[string]any{"type": "tool_use", "id": id, "name": name, "input": map[string]any{}})
				results = append(results, map[string]any{"type": "tool_result", "tool_use_id": id, "content": "done"})
			}
			messages = append(messages,
				map[string]any{"role": "user", "content": "earlier request"},
				map[string]any{"role": "assistant", "content": uses},
			)
			userContent = append(results, userContent...)
		}
		messages = append(messages, map[string]any{"role": "user", "content": userContent})
		body := map[string]any{"model": "public-model", "max_tokens": 64, "messages": messages, "tool_choice": anthropicToolChoice(choice)}
		if len(turn.offered) > 0 {
			offered := []any{}
			for _, name := range turn.offered {
				offered = append(offered, map[string]any{"name": name, "input_schema": json.RawMessage(stickyCatalogSchema)})
			}
			body["tools"] = offered
		}
		return mustJSON(body)
	default:
		messages := []any{}
		if len(turn.called) > 0 {
			calls, results := []any{}, []any{}
			for i, name := range turn.called {
				id := fmt.Sprintf("call_%d", i)
				calls = append(calls, map[string]any{
					"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": "{}"},
				})
				results = append(results, map[string]any{"role": "tool", "tool_call_id": id, "content": "done"})
			}
			messages = append(messages,
				map[string]any{"role": "user", "content": "earlier request"},
				map[string]any{"role": "assistant", "content": nil, "tool_calls": calls},
			)
			messages = append(messages, results...)
		}
		messages = append(messages, map[string]any{"role": "user", "content": query})
		body := map[string]any{"model": "public-model", "messages": messages, "tool_choice": chatToolChoice(choice)}
		if len(turn.offered) > 0 {
			offered := []any{}
			for _, name := range turn.offered {
				offered = append(offered, map[string]any{"type": "function", "function": map[string]any{
					"name": name, "parameters": json.RawMessage(stickyCatalogSchema),
				}})
			}
			body["tools"] = offered
		}
		return mustJSON(body)
	}
}

func chatToolChoice(choice string) any {
	if name, ok := strings.CutPrefix(choice, "named:"); ok {
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}
	}
	return choice
}

func responsesToolChoice(choice string) any {
	if name, ok := strings.CutPrefix(choice, "named:"); ok {
		return map[string]any{"type": "function", "name": name}
	}
	return choice
}

func anthropicToolChoice(choice string) any {
	switch {
	case choice == "required":
		return map[string]any{"type": "any"}
	case strings.HasPrefix(choice, "named:"):
		return map[string]any{"type": "tool", "name": strings.TrimPrefix(choice, "named:")}
	default:
		return map[string]any{"type": choice}
	}
}

func mustJSON(value any) []byte {
	data, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal fixture: %v", err))
	}
	return data
}

var stickyTestFormats = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
}
