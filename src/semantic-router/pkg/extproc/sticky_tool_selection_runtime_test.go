package extproc

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func TestStickyToolModelCapabilitiesUsesFinalSelectedModel(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"request-model":  {Capabilities: []string{"legacy"}},
			"selected-model": {Capabilities: []string{"tools", "vision"}},
		}},
	}}
	ctx := &RequestContext{
		RequestModel:     "request-model",
		VSRSelectedModel: "selected-model",
	}
	got := router.stickyToolModelCapabilities(ctx)
	if len(got) != 2 || got[0] != "tools" || got[1] != "vision" {
		t.Fatalf("capabilities = %#v, want selected-model capabilities", got)
	}
}

func TestStickyToolModelCapabilitiesDoesNotFallbackToRequestModel(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"request-model": {Capabilities: []string{"legacy"}},
		}},
	}}
	ctx := &RequestContext{RequestModel: "request-model"}
	if got := router.stickyToolModelCapabilities(ctx); got != nil {
		t.Fatalf("capabilities = %#v, want nil when final model is unavailable", got)
	}
}

func TestShouldApplyStickyToolSelectionRequiresFinalModel(t *testing.T) {
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config:                     &config.RouterConfig{},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.RequestModel = "request-model"
	ctx.VSRSelectedModel = ""
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	if router.shouldApplyStickyToolSelection(request, selection, ctx) {
		t.Fatal("sticky selection should bypass when final routed model is unavailable")
	}
}

func TestShouldApplyStickyToolSelectionBypassesExplicitAndUnresolvedRequests(t *testing.T) {
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config:                     &config.RouterConfig{},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	base := stickyRuntimeTestContext()
	base.VSRSelectedModel = "selected-model"

	tests := []struct {
		name  string
		setup func(*RequestContext, *llmprotocol.Request)
	}{
		{name: "explicit none", setup: func(_ *RequestContext, req *llmprotocol.Request) {
			req.ToolChoice.Mode = llmprotocol.ToolChoiceNone
		}},
		{name: "explicit required", setup: func(_ *RequestContext, req *llmprotocol.Request) {
			req.ToolChoice.Mode = llmprotocol.ToolChoiceRequired
		}},
		{name: "explicit named", setup: func(_ *RequestContext, req *llmprotocol.Request) {
			req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "search"}
		}},
		{name: "unresolved routing", setup: func(ctx *RequestContext, _ *llmprotocol.Request) {
			ctx.Routing = RequestRoutingContext{}
		}},
		{name: "passthrough routing", setup: func(ctx *RequestContext, _ *llmprotocol.Request) {
			ctx.Routing.SelectPassthrough()
		}},
		{name: "looper request", setup: func(ctx *RequestContext, _ *llmprotocol.Request) {
			ctx.LooperRequest = true
		}},
		{name: "skip processing", setup: func(ctx *RequestContext, _ *llmprotocol.Request) {
			ctx.SkipProcessing = true
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := *base
			req := *request
			tt.setup(&ctx, &req)
			if router.shouldApplyStickyToolSelection(&req, selection, &ctx) {
				t.Fatal("sticky selection should bypass this request")
			}
		})
	}
}

func TestHandleToolSelectionDecisionPluginDefaultsChoiceForStickyModes(t *testing.T) {
	tests := []struct {
		name       string
		mode       string
		sticky     *config.StickyToolSelectionConfig
		wantChoice llmprotocol.ToolChoiceMode
	}{
		{
			name: "sticky absent",
			mode: config.ToolSelectionModeAdd,
		},
		{
			name:   "sticky disabled",
			mode:   config.ToolSelectionModeAdd,
			sticky: &config.StickyToolSelectionConfig{},
		},
		{
			name:       "sticky enabled default add mode",
			sticky:     &config.StickyToolSelectionConfig{Enabled: true},
			wantChoice: llmprotocol.ToolChoiceAuto,
		},
		{
			name:       "sticky enabled filter mode",
			mode:       config.ToolSelectionModeFilter,
			sticky:     &config.StickyToolSelectionConfig{Enabled: true},
			wantChoice: llmprotocol.ToolChoiceAuto,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			request := &llmprotocol.Request{}
			selection := &config.ToolSelectionPluginConfig{
				Enabled: true,
				Mode:    tt.mode,
				Sticky:  tt.sticky,
			}

			handled, err := router.handleToolSelectionDecisionPlugin(
				request,
				"",
				nil,
				nil,
				stickyRuntimeTestContext(),
				selection,
				nil,
			)
			if err != nil {
				t.Fatalf("handleToolSelectionDecisionPlugin: %v", err)
			}
			if !handled {
				t.Fatal("tool_selection plugin should handle the request")
			}
			if request.ToolChoice.Mode != tt.wantChoice {
				t.Fatalf("tool choice = %q, want %q", request.ToolChoice.Mode, tt.wantChoice)
			}
		})
	}
}

func TestStickyCalledToolNamesKeepsLatestDistinctCallsInOrder(t *testing.T) {
	toolCall := func(name string) llmprotocol.Content {
		return llmprotocol.Content{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{Name: name},
		}
	}

	tests := []struct {
		name    string
		request *llmprotocol.Request
		limit   int
		want    []string
	}{
		{
			name: "duplicate calls retain the latest distinct window",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall("A")}},
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall("B"), toolCall("B")}},
			}},
			limit: 2,
			want:  []string{"A", "B"},
		},
		{
			name: "empty and non assistant calls are ignored",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{toolCall("user-call")}},
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
					toolCall(""),
					toolCall("  "),
					toolCall("search"),
				}},
			}},
			limit: 2,
			want:  []string{"search"},
		},
		{
			name: "limit keeps newest distinct names",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
					toolCall("A"),
					toolCall("B"),
					toolCall("C"),
					toolCall("D"),
				}},
			}},
			limit: 2,
			want:  []string{"C", "D"},
		},
		{
			name: "repeated consecutive calls are returned once",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
					toolCall("lookup"),
					toolCall("lookup"),
					toolCall("lookup"),
				}},
			}},
			limit: 3,
			want:  []string{"lookup"},
		},
		{
			name: "nonconsecutive repeats keep their latest position",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
					toolCall("A"),
					toolCall("B"),
					toolCall("A"),
					toolCall("C"),
				}},
			}},
			limit: 3,
			want:  []string{"B", "A", "C"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stickyCalledToolNames(tt.request, tt.limit); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("called tool names = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestApplyStickyToolSelectionDoesNotAdmitUnauthorizedCalledTool(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config:                     &config.RouterConfig{},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{Name: "not-authorized"},
			}},
		}},
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
	}
	search := llmprotocol.Tool{Name: "search", InputSchema: []byte(`{"type":"object"}`)}
	got, committed := router.applyStickyToolSelectionWithStatus(
		request,
		[]llmprotocol.Tool{search},
		[]llmprotocol.Tool{search},
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("selection = %#v, committed=%v", got, committed)
	}
	identity := ResolveStickyToolIdentity(ctx, string(ctx.Routing.RecipeName()), tools.EffectiveToolPolicyFingerprint(selection, nil))
	loaded, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if len(loaded.State.Tools) != 1 || loaded.State.Tools[0].Name != "search" {
		t.Fatalf("sticky state = %#v, want only authorized search tool", loaded.State.Tools)
	}
}

func TestMergeToolSelectionAdvancedIgnoresDisabledLegacyPolicy(t *testing.T) {
	base := &config.AdvancedToolFilteringConfig{Enabled: true, AllowTools: []string{"search"}}
	legacy := &config.ToolsPluginConfig{
		Enabled:    false,
		Mode:       config.ToolsPluginModeFiltered,
		AllowTools: []string{"calculate"},
		BlockTools: []string{"search"},
	}

	got := mergeToolSelectionAdvanced(nil, base, legacy)
	if got != base {
		t.Fatalf("advanced filtering = %#v, want unchanged base policy", got)
	}
}

func TestApplyStickyToolSelectionReusesOnEmptyTurn(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"selected-model": {Capabilities: []string{"tools"}},
			},
		}},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    config.ToolSelectionModeAdd,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	authorized := []llmprotocol.Tool{
		{Name: "search", InputSchema: []byte(`{"type":"object"}`)},
		{Name: "calculate", InputSchema: []byte(`{"type":"object"}`)},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	firstRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	first, committed := router.applyStickyToolSelectionWithStatus(
		firstRequest,
		authorized,
		[]llmprotocol.Tool{authorized[0]},
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(first) != 1 || first[0].Name != "search" {
		t.Fatalf("first selection = %#v, committed=%v", first, committed)
	}

	ctx.TurnIndex = 1
	emptyRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	reused, committed := router.applyStickyToolSelectionWithStatus(
		emptyRequest,
		authorized,
		nil,
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(reused) != 1 || reused[0].Name != "search" {
		t.Fatalf("empty-turn selection = %#v, committed=%v", reused, committed)
	}
}

func TestApplyStickyToolSelectionEmptyTurnPreservesOrdinaryNoOp(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"selected-model": {Capabilities: []string{"tools"}},
			},
		}},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    config.ToolSelectionModeAdd,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	authorized := []llmprotocol.Tool{{
		Name:        "search",
		InputSchema: []byte(`{"type":"object"}`),
	}}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	existing := append([]llmprotocol.Tool(nil), authorized...)
	request := &llmprotocol.Request{
		Tools:      existing,
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
	}
	selected, committed := router.applyStickyToolSelectionWithStatus(
		request,
		authorized,
		nil,
		selection,
		nil,
		"default",
		ctx,
	)
	if committed {
		t.Fatal("empty sticky selection should preserve the ordinary no-op path")
	}
	if len(selected) != 0 {
		t.Fatalf("empty-turn selection = %#v, want no replacement selection", selected)
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != "search" {
		t.Fatalf("request tools = %#v, want the ordinary no-op tools preserved", request.Tools)
	}
}

func TestApplyStickyToolSelectionReauthorizesCatalogAndSchema(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"selected-model": {Capabilities: []string{"tools"}},
			},
		}},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	searchV1 := llmprotocol.Tool{Name: "search", InputSchema: []byte(`{"type":"object"}`)}
	lookup := llmprotocol.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}
	firstRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	first, committed := router.applyStickyToolSelectionWithStatus(
		firstRequest,
		[]llmprotocol.Tool{searchV1, lookup},
		[]llmprotocol.Tool{searchV1},
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(first) != 1 || first[0].Name != "search" {
		t.Fatalf("first selection = %#v, committed=%v", first, committed)
	}

	ctx.TurnIndex++
	searchV2 := llmprotocol.Tool{Name: "search", Description: "updated", InputSchema: []byte(`{"type":"object"}`)}
	secondRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	second, committed := router.applyStickyToolSelectionWithStatus(
		secondRequest,
		[]llmprotocol.Tool{searchV2, lookup},
		[]llmprotocol.Tool{lookup},
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(second) != 1 || second[0].Name != "lookup" {
		t.Fatalf("schema/catalog change reused stale tools: %#v, committed=%v", second, committed)
	}

	ctx.TurnIndex++
	thirdRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	third, committed := router.applyStickyToolSelectionWithStatus(
		thirdRequest,
		[]llmprotocol.Tool{lookup},
		[]llmprotocol.Tool{lookup},
		selection,
		nil,
		"default",
		ctx,
	)
	if !committed || len(third) != 1 || third[0].Name != "lookup" {
		t.Fatalf("catalog removal reused unauthorized tool: %#v, committed=%v", third, committed)
	}
}

func TestApplyStickyToolSelectionInvalidatesPolicyAndCapability(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"selected-model": {Capabilities: []string{"tools"}},
			},
		}},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	search := llmprotocol.Tool{Name: "search", InputSchema: []byte(`{"type":"object"}`)}
	lookup := llmprotocol.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}
	apply := func(selected []llmprotocol.Tool) []llmprotocol.Tool {
		request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
		result, committed := router.applyStickyToolSelectionWithStatus(
			request,
			[]llmprotocol.Tool{search, lookup},
			selected,
			selection,
			nil,
			"default",
			ctx,
		)
		if !committed {
			t.Fatalf("sticky selection did not commit: %#v", result)
		}
		return result
	}

	if got := apply([]llmprotocol.Tool{search}); len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("initial selection = %#v", got)
	}
	ctx.TurnIndex++
	selection.Sticky.MaxTools = intPtr(2)
	if got := apply([]llmprotocol.Tool{lookup}); len(got) != 1 || got[0].Name != "lookup" {
		t.Fatalf("policy change reused stale tools: %#v", got)
	}
	ctx.TurnIndex++
	router.Config.BackendModels.ModelConfig["selected-model"] = config.ModelParams{Capabilities: []string{"vision"}}
	if got := apply([]llmprotocol.Tool{search}); len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("capability change produced unexpected selection: %#v", got)
	}
}

func TestApplyStickyToolSelectionStoreUnavailableFallsBack(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	manager, err := sessiontools.NewManager(&stickyUnavailableStore{}, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	router := &OpenAIRouter{
		Config:                     &config.RouterConfig{},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	selected := []llmprotocol.Tool{{Name: "search", InputSchema: []byte(`{"type":"object"}`)}}
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	got, committed := router.applyStickyToolSelectionWithStatus(request, selected, selected, selection, nil, "default", ctx)
	if committed {
		t.Fatal("store-unavailable selection must not commit sticky state")
	}
	if len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("fallback selection = %#v, want ordinary selection", got)
	}
}

func TestApplyStickyToolSelectionRejectsDuplicateAuthorizedDefinitionsBeforeCommit(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "runtime-test-secret")
	store := sessiontools.NewMemoryStore(config.ToolSessionStoreConfig{}, nil)
	manager, err := sessiontools.NewManager(store, sessiontools.DefaultManagerOptions())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config:                     &config.RouterConfig{},
		stickyToolSelectionManager: manager,
	}
	selection := &config.ToolSelectionPluginConfig{
		Enabled: true,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
	ctx := stickyRuntimeTestContext()
	ctx.VSRSelectedModel = "selected-model"
	duplicate := llmprotocol.Tool{Name: "search", InputSchema: []byte(`{"type":"object"}`)}
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	got, committed := router.applyStickyToolSelectionWithStatus(
		request,
		[]llmprotocol.Tool{duplicate, duplicate},
		[]llmprotocol.Tool{duplicate},
		selection,
		nil,
		"default",
		ctx,
	)
	if committed || len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("duplicate definitions must fail open without projection: %#v, committed=%v", got, committed)
	}
	identity := ResolveStickyToolIdentity(ctx, string(ctx.Routing.RecipeName()), tools.EffectiveToolPolicyFingerprint(selection, nil))
	if !identity.Trusted {
		t.Fatal("test identity should be trusted")
	}
	loaded, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil {
		t.Fatalf("load duplicate-definition state: %v", err)
	}
	if loaded.Found {
		t.Fatal("duplicate authorized definitions must not leave committed state")
	}
}

type stickyUnavailableStore struct{}

func (*stickyUnavailableStore) Load(context.Context, string) (sessiontools.VersionedState, error) {
	return sessiontools.VersionedState{}, errors.New("sticky test store unavailable")
}

func (*stickyUnavailableStore) CompareAndSwap(context.Context, string, uint64, sessiontools.State, time.Duration, sessiontools.QuotaKey) (bool, error) {
	return false, errors.New("sticky test store unavailable")
}

func (*stickyUnavailableStore) Delete(context.Context, string) error {
	return errors.New("sticky test store unavailable")
}

func (*stickyUnavailableStore) Close() error { return nil }

func stickyRuntimeTestContext() *RequestContext {
	decision := &config.Decision{Name: "sticky-runtime"}
	recipe := &config.RoutingRecipe{Name: "recipe-a"}
	ctx := &RequestContext{
		SessionID:              "session-a",
		SessionProvenance:      SessionProvenanceResponseAPI,
		AuthenticatedPrincipal: "principal-a",
		VSRSelectedDecision:    decision,
		TargetFormat:           llmprotocol.OpenAIChatV1,
	}
	ctx.Routing.SelectRecipe(recipe)
	return ctx
}
