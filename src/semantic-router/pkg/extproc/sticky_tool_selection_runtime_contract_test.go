package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func TestStickyRuntimeNoneModeDoesNotRestoreState(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection(config.ToolSelectionModeFilter)
	request := &llmprotocol.Request{
		Tools:      stickyRuntimeContractTools("search", "calculate"),
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
	}
	ctx.SemanticRequest = request
	toolsCfg := &config.ToolsPluginConfig{Enabled: true, Mode: config.ToolsPluginModeNone}

	handled, err := router.handleToolSelectionDecisionPlugin(
		request, "", nil, nil, ctx, selection, toolsCfg,
	)
	if err != nil {
		t.Fatalf("handleToolSelectionDecisionPlugin: %v", err)
	}
	if !handled {
		t.Fatal("none mode should be handled before sticky selection")
	}
	if len(request.Tools) != 0 || request.ToolChoice.Mode != "" {
		t.Fatalf("none mode restored or retained tools: tools=%#v choice=%q", request.Tools, request.ToolChoice.Mode)
	}
	if loaded, err := store.Load(context.Background(), stickyRuntimeContractStorageKey(ctx, selection, nil)); err != nil {
		t.Fatalf("load sticky state: %v", err)
	} else if loaded.Found {
		t.Fatal("none mode should not create sticky state")
	}
}

func TestStickyRuntimeFilteredAutoUsesCurrentAuthorizedCatalog(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection(config.ToolSelectionModeFilter)
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	request.Tools = stickyRuntimeContractTools("search", "calculate")
	ctx.SemanticRequest = request
	toolsCfg := &config.ToolsPluginConfig{
		Enabled:    true,
		Mode:       config.ToolsPluginModeFiltered,
		AllowTools: []string{"search"},
	}

	handled, err := router.handleToolSelectionDecisionPlugin(
		request, "", nil, nil, ctx, selection, toolsCfg,
	)
	if err != nil {
		t.Fatalf("handleToolSelectionDecisionPlugin: %v", err)
	}
	if !handled {
		t.Fatal("filtered auto mode should be handled")
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != "search" {
		t.Fatalf("filtered authorized tools = %#v, want search only", request.Tools)
	}
	loaded, err := store.Load(context.Background(), stickyRuntimeContractStorageKey(ctx, selection, nil))
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if !loaded.Found || len(loaded.State.Tools) != 1 || loaded.State.Tools[0].Name != "search" {
		t.Fatalf("sticky state = %#v, want current authorized search tool", loaded.State)
	}
}

func TestStickyRuntimePassthroughAndExplicitChoiceBypass(t *testing.T) {
	tests := []struct {
		name       string
		selection  string
		mode       string
		toolChoice llmprotocol.ToolChoice
		wantTools  int
	}{
		{
			name:       "passthrough explicit choice",
			selection:  config.ToolSelectionModeFilter,
			mode:       config.ToolsPluginModePassthrough,
			toolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "search"},
			wantTools:  2,
		},
		{
			name:       "filtered explicit choice",
			selection:  config.ToolSelectionModeFilter,
			mode:       config.ToolsPluginModeFiltered,
			toolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "search"},
			wantTools:  1,
		},
		{
			name:       "add explicit choice",
			selection:  config.ToolSelectionModeAdd,
			mode:       config.ToolsPluginModePassthrough,
			toolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "search"},
			wantTools:  2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router, store, ctx := newStickyRuntimeContractRouter(t)
			selection := stickyRuntimeContractSelection(tt.selection)
			request := &llmprotocol.Request{
				Tools:      stickyRuntimeContractTools("search", "calculate"),
				ToolChoice: tt.toolChoice,
			}
			ctx.SemanticRequest = request
			toolsCfg := &config.ToolsPluginConfig{
				Enabled:    true,
				Mode:       tt.mode,
				AllowTools: []string{"search"},
			}

			handled, err := router.handleToolSelectionDecisionPlugin(
				request, "query", nil, nil, ctx, selection, toolsCfg,
			)
			if err != nil {
				t.Fatalf("handleToolSelectionDecisionPlugin: %v", err)
			}
			if !handled {
				t.Fatal("explicit choice should be handled by the tool selection plugin")
			}
			if request.ToolChoice != tt.toolChoice {
				t.Fatalf("tool choice changed: got=%+v want=%+v", request.ToolChoice, tt.toolChoice)
			}
			if len(request.Tools) != tt.wantTools || request.Tools[0].Name != "search" {
				t.Fatalf("explicit choice tools = %#v, want %d tools starting with search", request.Tools, tt.wantTools)
			}
			if loaded, err := store.Load(context.Background(), stickyRuntimeContractStorageKey(ctx, selection, nil)); err != nil {
				t.Fatalf("load sticky state: %v", err)
			} else if loaded.Found {
				t.Fatal("explicit tool choice must bypass sticky state")
			}
		})
	}
}

func TestStickyRuntimeCatalogFingerprintInvalidatesHistoricalSelection(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection(config.ToolSelectionModeAdd)
	oldCatalog := stickyRuntimeContractTools("search", "calculate")
	firstRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	first, committed := router.applyStickyToolSelectionWithStatus(
		firstRequest, oldCatalog, []llmprotocol.Tool{oldCatalog[0]}, selection, nil, "default", ctx,
	)
	if !committed || len(first) != 1 || first[0].Name != "search" {
		t.Fatalf("first selection = %#v, committed=%v", first, committed)
	}

	ctx.TurnIndex = 1
	changedCatalog := stickyRuntimeContractTools("search-v2", "calculate")
	secondRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	second, committed := router.applyStickyToolSelectionWithStatus(
		secondRequest, changedCatalog, []llmprotocol.Tool{changedCatalog[1]}, selection, nil, "default", ctx,
	)
	if !committed || len(second) != 1 || second[0].Name != "calculate" {
		t.Fatalf("invalidated selection = %#v, committed=%v", second, committed)
	}
	loaded, err := store.Load(context.Background(), stickyRuntimeContractStorageKey(ctx, selection, nil))
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if !loaded.Found || len(loaded.State.Tools) != 1 || loaded.State.Tools[0].Name != "calculate" {
		t.Fatalf("state after catalog invalidation = %#v", loaded.State)
	}
}

func TestStickyRuntimePolicyFingerprintInvalidatesHistoricalSelection(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection(config.ToolSelectionModeAdd)
	catalog := stickyRuntimeContractTools("search", "calculate")
	oldPolicy := &config.ToolsPluginConfig{
		Enabled:    true,
		Mode:       config.ToolsPluginModeFiltered,
		AllowTools: []string{"search"},
	}
	firstRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	first, committed := router.applyStickyToolSelectionWithStatus(
		firstRequest, catalog, []llmprotocol.Tool{catalog[0]}, selection, oldPolicy, "default", ctx,
	)
	if !committed || len(first) != 1 || first[0].Name != "search" {
		t.Fatalf("first selection = %#v, committed=%v", first, committed)
	}

	ctx.TurnIndex = 1
	newPolicy := &config.ToolsPluginConfig{
		Enabled:    true,
		Mode:       config.ToolsPluginModeFiltered,
		AllowTools: []string{"calculate"},
	}
	secondRequest := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	second, committed := router.applyStickyToolSelectionWithStatus(
		secondRequest, catalog, []llmprotocol.Tool{catalog[1]}, selection, newPolicy, "default", ctx,
	)
	if !committed || len(second) != 1 || second[0].Name != "calculate" {
		t.Fatalf("invalidated selection = %#v, committed=%v", second, committed)
	}
	oldIdentity := stickyRuntimeContractIdentity(ctx, selection, oldPolicy)
	loaded, err := store.Load(context.Background(), oldIdentity.StorageKey)
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if !loaded.Found || len(loaded.State.Tools) != 1 || loaded.State.Tools[0].Name != "calculate" {
		t.Fatalf("state after policy invalidation = %#v, want calculate", loaded.State)
	}
}

func TestStickyRuntimeProjectionFailureDoesNotCommitStaleState(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection(config.ToolSelectionModeAdd)
	authorized := stickyRuntimeContractTools("search")
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
	selected, committed := router.applyStickyToolSelectionWithStatus(
		request, authorized, authorized, selection, nil, "default", ctx,
	)
	if !committed || len(selected) != 1 {
		t.Fatalf("seed selection = %#v, committed=%v", selected, committed)
	}
	identity := stickyRuntimeContractIdentity(ctx, selection, nil)
	before, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil || !before.Found {
		t.Fatalf("load seed state: found=%v err=%v", before.Found, err)
	}

	ctx.TurnIndex = 1
	duplicateCatalog := append(append([]llmprotocol.Tool(nil), authorized...), authorized[0])
	fallbackSelection := []llmprotocol.Tool{{Name: "calculate", InputSchema: json.RawMessage(`{"type":"object"}`)}}
	result, committed := router.applyStickyToolSelectionWithStatus(
		request, duplicateCatalog, fallbackSelection, selection, nil, "default", ctx,
	)
	if committed {
		t.Fatal("projection failure should not commit a sticky update")
	}
	if len(result) != 1 || result[0].Name != "calculate" {
		t.Fatalf("projection fallback selection = %#v, want ordinary selection", result)
	}
	after, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil || !after.Found {
		t.Fatalf("load state after projection failure: found=%v err=%v", after.Found, err)
	}
	if after.State.Revision != before.State.Revision || len(after.State.Tools) != len(before.State.Tools) {
		t.Fatalf("projection failure changed state: before=%#v after=%#v", before.State, after.State)
	}
}

func newStickyRuntimeContractRouter(t *testing.T) (*OpenAIRouter, sessiontools.Store, *RequestContext) {
	t.Helper()
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "sticky-runtime-contract-secret")
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
	ctx := &RequestContext{
		SessionID:              "sticky-session",
		SessionProvenance:      SessionProvenanceResponseAPI,
		AuthenticatedPrincipal: "sticky-principal",
		VSRSelectedModel:       "selected-model",
		VSRSelectedDecision:    &config.Decision{Name: "sticky-decision"},
		TargetFormat:           llmprotocol.OpenAIChatV1,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "sticky-recipe"})
	return router, store, ctx
}

func stickyRuntimeContractSelection(mode string) *config.ToolSelectionPluginConfig {
	return &config.ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    mode,
		Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
	}
}

func stickyRuntimeContractTools(names ...string) []llmprotocol.Tool {
	result := make([]llmprotocol.Tool, 0, len(names))
	for _, name := range names {
		result = append(result, llmprotocol.Tool{
			Name:        name,
			InputSchema: json.RawMessage(`{"type":"object"}`),
		})
	}
	return result
}

func stickyRuntimeContractIdentity(
	ctx *RequestContext,
	selection *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) ResolvedStickyIdentity {
	policy := tools.EffectiveToolPolicyFingerprint(selection, toolsCfg)
	return ResolveStickyToolIdentity(ctx, string(ctx.Routing.RecipeName()), policy)
}

func stickyRuntimeContractStorageKey(
	ctx *RequestContext,
	selection *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) string {
	return stickyRuntimeContractIdentity(ctx, selection, toolsCfg).StorageKey
}
