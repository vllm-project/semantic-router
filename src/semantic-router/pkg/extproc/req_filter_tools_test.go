package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// toolsJudge is a fake decision model: it answers tool_need with need and
// scores each tool_rank option by choose, and records every call.
type toolsJudge struct {
	mu     sync.Mutex
	need   float64
	choose map[string]float64
	err    error
	calls  []modelservice.Request
}

func (j *toolsJudge) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	j.mu.Lock()
	j.calls = append(j.calls, request)
	j.mu.Unlock()
	if j.err != nil {
		return modelservice.Response{}, j.err
	}
	answers := make(map[string]modelservice.Answer, len(request.Questions))
	for _, question := range request.Questions {
		answer := modelservice.Answer{Type: question.Type, Noul: j.need, Probabilities: map[string]float64{}}
		total := 0.0
		for _, choice := range question.Choices {
			total += j.choose[choice.Key]
		}
		for _, choice := range question.Choices {
			answer.Probabilities[choice.Key] = 1 / float64(len(question.Choices))
			if total > 0 {
				answer.Probabilities[choice.Key] = j.choose[choice.Key] / total
			}
			if answer.Choice == "" || answer.Probabilities[choice.Key] > answer.Probabilities[answer.Choice] {
				answer.Choice = choice.Key
			}
		}
		answers[question.ID] = answer
	}
	return modelservice.Response{Answers: answers}, nil
}

func (j *toolsJudge) asked() []modelservice.Request {
	j.mu.Lock()
	defer j.mu.Unlock()
	return append([]modelservice.Request(nil), j.calls...)
}

func toolsRouter(judge modelservice.Decider) *OpenAIRouter {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		config.DefaultDecisionDeployment: {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
	}
	return &OpenAIRouter{
		Config:          cfg,
		decisionDecider: judge,
		decisionCards:   map[string]modelservice.ModelCard{config.DefaultDecisionDeployment: {ID: "judge", Surfaces: []string{"decisions"}}},
	}
}

func toolsContext(plugin config.ToolsPluginConfig) *RequestContext {
	decision := &config.Decision{Name: "agent", Plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginTools, Configuration: config.MustStructuredPayload(plugin)},
	}}
	return &RequestContext{RequestID: "tools-1", Headers: map[string]string{}, VSRSelectedDecision: decision, VSRSelectedDecisionName: decision.Name}
}

func toolsRequest(names ...string) *llmprotocol.Request {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: textBlocks("what did we decide last time?")},
		{Role: llmprotocol.RoleAssistant, Content: textBlocks("the release moved")},
		{Role: llmprotocol.RoleUser, Content: textBlocks("find the onboarding guide")},
	}}
	for _, name := range names {
		request.Tools = append(request.Tools, llmprotocol.Tool{Name: name, Description: "Tool " + name, InputSchema: json.RawMessage(`{"type":"object"}`)})
	}
	return request
}

func toolNames(tools []llmprotocol.Tool) []string {
	names := make([]string, len(tools))
	for index, tool := range tools {
		names[index] = tool.Name
	}
	return names
}

func toolsSelect(mode string, edit func(*config.ToolsSelectConfig)) config.ToolsPluginConfig {
	spec := &config.ToolsSelectConfig{Mode: mode}
	if edit != nil {
		edit(spec)
	}
	return config.ToolsPluginConfig{Enabled: true, Select: spec}
}

func applyTools(t *testing.T, router *OpenAIRouter, plugin config.ToolsPluginConfig, request *llmprotocol.Request) (*RequestContext, *ext_proc.ProcessingResponse) {
	t.Helper()
	ctx := toolsContext(plugin)
	ctx.SemanticRequest = request
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	router.applyToolsPlugin(request, response, ctx)
	return ctx, response
}

// Filter mode keeps the tools the judge keeps, in the request's own order:
// tool_rank's default keeps the 3 best at 0.1 of the best or more.
func TestToolsSelectFilterKeepsTheRequestOrder(t *testing.T) {
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"c": 0.5, "a": 0.3, "d": 0.15, "b": 0.05, "e": 0.0}}
	request := toolsRequest("a", "b", "c", "d", "e")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
	applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, nil), request)
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"a", "c", "d"}) {
		t.Fatalf("kept %v, want the three best in the request's order", got)
	}
	if request.ToolChoice.Mode != llmprotocol.ToolChoiceAuto {
		t.Fatal("a request that keeps tools keeps its tool_choice")
	}
}

// Add mode appends the catalog's chosen tools after the request's own, best
// first, never one the request carries or the policy refuses.
func TestToolsSelectAddAppendsCatalogTools(t *testing.T) {
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"w": 0.6, "y": 0.3, "z": 0.02}}
	router := toolsRouter(judge)
	catalog := []config.CatalogTool{}
	for _, name := range []string{"x", "y", "z", "w", "admin.delete"} {
		catalog = append(catalog, config.CatalogTool{Tool: llmprotocol.Tool{Name: name, InputSchema: json.RawMessage(`{"type":"object"}`)}, Description: "Use " + name})
	}
	router.toolCatalogs = map[string]*toolCatalog{"tools.json": {tools: catalog, loadedAt: time.Now()}}
	plugin := toolsSelect(config.ToolsSelectModeAdd, func(s *config.ToolsSelectConfig) { s.ToolsDBPath = "tools.json" })
	plugin.Policy = &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeFiltered, BlockTools: []string{"admin.delete"}}
	request := toolsRequest("x")
	applyTools(t, router, plugin, request)
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"x", "w", "y"}) {
		t.Fatalf("tools %v, want the request's x, then w and y best first", got)
	}
	rank := judge.asked()[0]
	if len(judge.asked()) != 2 || rank.Questions[0].ID == toolNeedQuestionID {
		rank = judge.asked()[1]
	}
	var options []string
	for _, choice := range rank.Questions[0].Choices {
		options = append(options, choice.Key+"="+choice.Description)
	}
	if !reflect.DeepEqual(options, []string{"y=Use y", "z=Use z", "w=Use w"}) {
		t.Fatalf("tool_rank options %v: the catalog's tools the request lacks and the policy allows", options)
	}
}

// Below need_threshold the judge's tools are neither kept nor added: filter
// mode keeps only the request's first preserve_count tools.
func TestToolsSelectNeedGate(t *testing.T) {
	choose := map[string]float64{"c": 0.9, "a": 0.1}
	for name, test := range map[string]struct {
		preserve int
		want     []string
	}{
		"preserve none": {0, []string{}},
		"preserve one":  {1, []string{"a"}},
	} {
		request := toolsRequest("a", "b", "c")
		request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
		applyTools(t, toolsRouter(&toolsJudge{need: 0.2, choose: choose}), toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.PreserveCount = test.preserve }), request)
		if got := toolNames(request.Tools); !reflect.DeepEqual(got, test.want) {
			t.Errorf("%s: tools %v, want %v", name, got, test.want)
		}
		if len(request.Tools) == 0 && request.ToolChoice.Mode != "" {
			t.Errorf("%s: a request left without tools drops its tool_choice", name)
		}
	}
	threshold := 0.1
	request := toolsRequest("a", "b", "c")
	applyTools(t, toolsRouter(&toolsJudge{need: 0.2, choose: choose}), toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.NeedThreshold = &threshold }), request)
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"a", "c"}) {
		t.Fatalf("need_threshold 0.1 passes P(need) 0.2: tools %v", got)
	}
}

// preserve_count keeps the request's first tools beside the judge's.
func TestToolsSelectPreserveCount(t *testing.T) {
	request := toolsRequest("a", "b", "c", "d")
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"d": 0.9}}
	applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.PreserveCount = 2 }), request)
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"a", "b", "d"}) {
		t.Fatalf("tools %v", got)
	}
}

// on_error decides what a failed call does, per mode; fail refuses the
// request with tool_selection_failed.
func TestToolsSelectOnError(t *testing.T) {
	for name, test := range map[string]struct {
		mode, onError string
		want          []string
		refused       bool
	}{
		"filter keeps by default": {config.ToolsSelectModeFilter, "", []string{"a", "b"}, false},
		"filter strips":           {config.ToolsSelectModeFilter, config.ToolsSelectOnErrorStrip, []string{}, false},
		"filter fails":            {config.ToolsSelectModeFilter, config.ToolsSelectOnErrorFail, []string{"a", "b"}, true},
		"add skips by default":    {config.ToolsSelectModeAdd, "", []string{"a", "b"}, false},
		"add fails":               {config.ToolsSelectModeAdd, config.ToolsSelectOnErrorFail, []string{"a", "b"}, true},
	} {
		router := toolsRouter(&toolsJudge{err: modelservice.ErrOverloaded})
		router.toolCatalogs = map[string]*toolCatalog{"tools.json": {tools: []config.CatalogTool{{Tool: llmprotocol.Tool{Name: "c"}}, {Tool: llmprotocol.Tool{Name: "d"}}}}}
		request := toolsRequest("a", "b")
		ctx, response := applyTools(t, router, toolsSelect(test.mode, func(s *config.ToolsSelectConfig) {
			s.OnError = test.onError
			if test.mode == config.ToolsSelectModeAdd {
				s.ToolsDBPath = "tools.json"
			}
		}), request)
		if got := toolNames(request.Tools); !reflect.DeepEqual(got, test.want) {
			t.Errorf("%s: tools %v, want %v", name, got, test.want)
		}
		immediate := response.GetImmediateResponse()
		if test.refused != (immediate != nil) {
			t.Fatalf("%s: refused = %v", name, immediate != nil)
		}
		if test.refused && (immediate.GetStatus().GetCode() != 503 || ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "tool_selection_failed") {
			t.Errorf("%s: response %v, error %+v", name, immediate.GetStatus(), ctx.ImmediateProtocolError)
		}
	}
}

// More request tools than one question takes options: keep forwards them
// unselected without asking, reject refuses the request.
func TestToolsSelectOnOverflow(t *testing.T) {
	names := make([]string, 256)
	for index := range names {
		names[index] = fmt.Sprintf("tool_%03d", index)
	}
	judge := &toolsJudge{need: 0.9}
	request := toolsRequest(names...)
	_, response := applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, nil), request)
	if len(request.Tools) != 256 || response.GetImmediateResponse() != nil || len(judge.asked()) != 0 {
		t.Fatalf("keep forwards every tool unasked: %d tools, %d calls", len(request.Tools), len(judge.asked()))
	}
	ctx, response := applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.OnOverflow = config.ToolsSelectOnOverflowReject }), toolsRequest(names...))
	if response.GetImmediateResponse().GetStatus().GetCode() != 400 || ctx.ImmediateProtocolError.Code != "too_many_tools" {
		t.Fatalf("reject refuses the request: %v %+v", response.GetImmediateResponse().GetStatus(), ctx.ImmediateProtocolError)
	}
}

// A request that decides its tool use itself is not selected for, and a
// policy runs before any selection.
func TestToolsSelectRunsWhenTheModelChoosesTools(t *testing.T) {
	for _, mode := range []llmprotocol.ToolChoiceMode{llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired} {
		judge := &toolsJudge{need: 0.9}
		request := toolsRequest("a", "b")
		request.ToolChoice = llmprotocol.ToolChoice{Mode: mode}
		applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, nil), request)
		if len(judge.asked()) != 0 || len(request.Tools) != 2 {
			t.Errorf("tool_choice %s: %d calls, tools %v", mode, len(judge.asked()), toolNames(request.Tools))
		}
	}
	plugin := config.ToolsPluginConfig{Enabled: true, Policy: &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeFiltered, AllowTools: []string{"a", "c"}}}
	request := toolsRequest("a", "b", "c")
	applyTools(t, toolsRouter(&toolsJudge{}), plugin, request)
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"a", "c"}) {
		t.Fatalf("the filtered policy keeps the allowed tools: %v", got)
	}
}

// With input.history the questions read the earlier user turns as their own
// part, beside the current turn.
func TestToolsSelectReadsHistoryWhenAsked(t *testing.T) {
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"a": 0.9}}
	applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.Input.History = true }), toolsRequest("a", "b"))
	for _, state := range judge.asked() {
		if state.Parts["history"] != "what did we decide last time?" || state.Parts["request"] != "find the onboarding guide" {
			t.Fatalf("state %+v", state)
		}
	}
	judge = &toolsJudge{need: 0.9, choose: map[string]float64{"a": 0.9}}
	applyTools(t, toolsRouter(judge), toolsSelect(config.ToolsSelectModeFilter, nil), toolsRequest("a", "b"))
	if state := judge.asked()[0]; state.State != "find the onboarding guide" || state.Parts != nil {
		t.Fatalf("without history the state is the current turn: %+v", state)
	}
}

// The selection is recorded for the replay record: what it kept and P(need).
func TestToolsSelectIsRecorded(t *testing.T) {
	ctx, _ := applyTools(t, toolsRouter(&toolsJudge{need: 0.75, choose: map[string]float64{"b": 0.9}}), toolsSelect(config.ToolsSelectModeFilter, nil), toolsRequest("a", "b"))
	if len(ctx.pendingToolsOutcomes) != 1 {
		t.Fatalf("outcomes %+v", ctx.pendingToolsOutcomes)
	}
	outcome := ctx.pendingToolsOutcomes[0]
	if outcome.Target != "select" || outcome.Verdict != toolsSelected || outcome.Metadata["kept"] != "b" || outcome.Metadata["tool_need"] != "0.7500" || outcome.Metadata["deployment"] != config.DefaultDecisionDeployment {
		t.Fatalf("outcome %+v", outcome)
	}
}

// tool_rank and tool_need go to the decision model in one decisions call
// about one state.
func TestToolsSelectAsksInOneCall(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "vela"})
	server := httptest.NewServer(runtime.Handler())
	t.Cleanup(server.Close)
	deployments := map[string]config.ModelDeployment{"vela": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(deployments)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	card, err := lease.Card(ctx, "vela")
	if err != nil {
		t.Fatal(err)
	}
	router := toolsRouter(lease)
	router.Config.DecisionModel, router.Config.ModelDeployments = "vela", deployments
	router.decisionCards = map[string]modelservice.ModelCard{"vela": card}
	request := toolsRequest("calendar", "search", "billing")
	applyTools(t, router, toolsSelect(config.ToolsSelectModeFilter, func(s *config.ToolsSelectConfig) { s.TimeoutMs = 20000 }), request)
	calls := runtime.Decisions()
	if len(calls) != 1 || calls[0].State != "find the onboarding guide" || len(calls[0].Questions) != 2 {
		t.Fatalf("decisions calls %+v, want one about the request with both questions", calls)
	}
	if choices := calls[0].Questions["match:tool_rank"].Choices; choices == nil || len(*choices) != 3 {
		t.Fatalf("tool_rank options %+v", calls[0].Questions["match:tool_rank"])
	}
	if _, ok := calls[0].Questions[toolNeedQuestionID]; !ok {
		t.Fatalf("questions %v", calls[0].Questions)
	}
	if got := toolNames(request.Tools); !reflect.DeepEqual(got, []string{"calendar", "search", "billing"}) {
		t.Fatalf("tools %v: the fake needs a tool at 0.8 and chooses the first option at 0.7", got)
	}
}

// The tools plugin answers the client's debug surface and the replay record
// only: it sets no header on the provider-bound request, debug or not.
func TestToolsPluginSetsNoProviderBoundHeader(t *testing.T) {
	router := toolsRouter(&toolsJudge{need: 0.9, choose: map[string]float64{"a": 0.9}})
	router.Config.Headers.Debug = config.DebugHeadersAny
	ctx := toolsContext(toolsSelect(config.ToolsSelectModeFilter, nil))
	if _, err := router.handleRequestHeaders(headerContractRequest(map[string]string{headers.VSRDebug: "true"}), ctx); err != nil {
		t.Fatal(err)
	}
	if !debugHeadersRequested(ctx) {
		t.Fatal("the caller is allowed the debug headers")
	}
	request := toolsRequest("a", "b")
	ctx.SemanticRequest = request
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	router.applyToolsPlugin(request, response, ctx)
	for _, option := range response.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if name := strings.ToLower(option.GetHeader().GetKey()); strings.HasPrefix(name, "x-vsr-") {
			t.Errorf("the tools plugin set %s on the provider-bound request", name)
		}
	}
	if len(ctx.pendingToolsOutcomes) != 1 {
		t.Fatal("the selection is recorded for the replay record")
	}
}

// A judge error is reported by its reason in the replay record.
func TestToolsSelectFailureIsRecorded(t *testing.T) {
	ctx, _ := applyTools(t, toolsRouter(&toolsJudge{err: modelservice.ErrUnavailable}), toolsSelect(config.ToolsSelectModeFilter, nil), toolsRequest("a", "b"))
	outcome := ctx.pendingToolsOutcomes[0]
	if outcome.Verdict != toolsSelectFailed || outcome.Reason != "unavailable" || outcome.Metadata["action"] != config.ToolsSelectOnErrorKeep {
		t.Fatalf("outcome %+v", outcome)
	}
}

func mustToolsDecisionPlugin(t *testing.T, cfg *config.ToolsPluginConfig) config.DecisionPlugin {
	t.Helper()
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return config.DecisionPlugin{Type: config.DecisionPluginTools, Configuration: payload}
}
