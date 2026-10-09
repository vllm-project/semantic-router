package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

// Add mode retains the selected prefix, grows within max_tools, and never lets
// relevance replace a retained tool. Each turn's provider-bound array is
// decoded from the dispatched body.
func TestStickyAddModeReusesAndGrowsAcrossTurns(t *testing.T) {
	for _, format := range stickyTestFormats {
		t.Run(string(format), func(t *testing.T) {
			decision := stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
				Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
				Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(2)},
			}, supportedStickyToolsConfig())
			h := newStickyHarness(t, format, decision, nil)

			first := h.run(stickyTurn{query: stickyTestQueries["weather"]})
			require.Equal(t, []string{"weather"}, first.tools)
			require.Equal(t, "seeded", first.receipt.Outcome)
			require.Contains(t, string(first.body), "9007199254740993", "exact schema numeral must reach the provider")

			second := h.run(stickyTurn{query: stickyTestQueries["calendar"]})
			require.Equal(t, []string{"weather", "calendar"}, second.tools)
			require.Equal(t, "updated", second.receipt.Outcome)
			require.Equal(t, 1, second.receipt.Reused)
			require.Equal(t, 1, second.receipt.Added)

			third := h.run(stickyTurn{query: stickyTestQueries["email"]})
			require.Equal(t, []string{"weather", "calendar"}, third.tools, "relevance must not replace a retained tool at capacity")
			require.Equal(t, "reused", third.receipt.Outcome)
		})
	}
}

func stickyAddDecision(t *testing.T, sticky *config.StickyToolSelectionConfig) config.Decision {
	t.Helper()
	sticky.Enabled = true
	return stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1, Sticky: sticky,
	}, supportedStickyToolsConfig())
}

// Filter mode emits only tools the current request offers. A retained tool
// the client stops offering is gone, and the catalog change resets state.
func TestStickyFilterModeKeepsOfferedCatalog(t *testing.T) {
	all := []string{"weather", "calendar", "email", "files"}
	for _, format := range stickyTestFormats {
		t.Run(string(format), func(t *testing.T) {
			decision := stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
				Enabled: true, Mode: config.ToolSelectionModeFilter,
				Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(3)},
			}, supportedStickyToolsConfig())
			h := newStickyHarness(t, format, decision, nil)

			require.Equal(t, []string{"weather"}, h.run(stickyTurn{query: stickyTestQueries["weather"], offered: all}).tools)
			second := h.run(stickyTurn{query: stickyTestQueries["email"], offered: all})
			require.Equal(t, []string{"weather", "email"}, second.tools)
			require.Equal(t, "updated", second.receipt.Outcome)

			narrowed := h.run(stickyTurn{query: stickyTestQueries["files"], offered: []string{"email", "files"}})
			require.Equal(t, []string{"files"}, narrowed.tools, "an unoffered retained tool must not be restored")
			require.Equal(t, "reset", narrowed.receipt.Outcome)
			require.Equal(t, "catalog_changed", narrowed.receipt.Reason)
		})
	}
}

// A called tool is pinned and replaces the newest unpinned tool; more pins
// than max_tools reset continuity to a bounded fresh selection.
func TestStickyPinsCalledToolsAndBoundsOverflow(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(2)}), nil)
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	require.Equal(t, []string{"weather", "calendar"}, h.run(stickyTurn{query: stickyTestQueries["calendar"]}).tools)

	pinned := h.run(stickyTurn{query: stickyTestQueries["files"], called: []string{"email"}})
	require.Equal(t, []string{"weather", "email"}, pinned.tools, "a pin replaces the newest unpinned tool")
	require.Equal(t, 1, pinned.receipt.Pinned)

	overflow := h.run(stickyTurn{query: stickyTestQueries["files"], called: []string{"email", "calendar", "weather"}})
	require.Equal(t, "reset", overflow.receipt.Outcome)
	require.Equal(t, "pin_capacity_exceeded", overflow.receipt.Reason)
	require.LessOrEqual(t, len(overflow.tools), 2)
}

// Revocation takes effect on the next request, including for a pinned tool:
// a block rule removes it from eligibility, and history that called it
// cannot restore it.
func TestStickyRevokedPinnedToolDisappears(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	pinned := h.run(stickyTurn{query: stickyTestQueries["weather"], called: []string{"email"}})
	require.Equal(t, []string{"weather", "email"}, pinned.tools)

	revoked := supportedStickyToolsConfig()
	revoked.Mode = config.ToolsPluginModeFiltered
	revoked.BlockTools = []string{"email"}
	h.decision = stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
		Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(3)},
	}, revoked)
	after := h.run(stickyTurn{query: stickyTestQueries["email"], called: []string{"email"}})
	require.NotContains(t, after.tools, "email", "a revoked pinned tool must disappear")
	require.NotEqual(t, "reused", after.receipt.Outcome)
}

// State is partitioned by authenticated principal and recipe: the same
// session value under another principal or recipe starts fresh.
func TestStickyIsolatesPrincipalsAndRecipes(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	require.Equal(t, []string{"weather", "calendar"}, h.run(stickyTurn{query: stickyTestQueries["calendar"]}).tools)

	other := h.run(stickyTurn{query: stickyTestQueries["email"], principal: "user-b"})
	require.Equal(t, []string{"email"}, other.tools)
	require.Equal(t, "seeded", other.receipt.Outcome)

	recipe := h.run(stickyTurn{query: stickyTestQueries["email"], recipe: "other-recipe"})
	require.Equal(t, []string{"email"}, recipe.tools)
	require.Equal(t, "seeded", recipe.receipt.Outcome)
}

// Without a trusted identity, sticky selection never touches the store and
// emits the ordinary current selection.
func TestStickyUntrustedIdentityPerformsNoStoreOperations(t *testing.T) {
	cases := map[string]struct {
		turn   stickyTurn
		reason string
	}{
		"missing principal":   {turn: stickyTurn{principal: " "}, reason: stickyToolIdentityReasonMissingPrincipal},
		"heuristic session":   {turn: stickyTurn{provenance: SessionProvenanceMessageHash}, reason: stickyToolIdentityReasonUntrustedProvenance},
		"request-id session":  {turn: stickyTurn{provenance: SessionProvenanceRequestID}, reason: stickyToolIdentityReasonUntrustedProvenance},
		"prompt-cache header": {turn: stickyTurn{provenance: SessionProvenanceAnthropicPromptCache}, reason: stickyToolIdentityReasonUntrustedProvenance},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{}), nil)
			tc.turn.query = stickyTestQueries["weather"]
			result := h.run(tc.turn)
			require.Equal(t, []string{"weather"}, result.tools)
			require.Equal(t, stickyToolOutcomeStateless, result.receipt.Outcome)
			require.Equal(t, tc.reason, result.receipt.Reason)
			require.Zero(t, h.store.operations(), "untrusted identity must not reach the store")
		})
	}
}

// Explicit tool choices and empty queries keep their ordinary handling and
// never read or write session state.
func TestStickyExplicitChoicesAndEmptyQueriesStayStateless(t *testing.T) {
	cases := map[string]struct {
		turn   stickyTurn
		tools  []string
		reason string
	}{
		"tool_choice none":     {turn: stickyTurn{query: stickyTestQueries["weather"], toolChoice: "none", offered: []string{"email"}}, tools: []string{"email"}, reason: stickyToolReasonToolChoiceNotAuto},
		"tool_choice required": {turn: stickyTurn{query: stickyTestQueries["weather"], toolChoice: "required", offered: []string{"email"}}, tools: []string{"email"}, reason: stickyToolReasonToolChoiceNotAuto},
		"named tool_choice":    {turn: stickyTurn{query: stickyTestQueries["weather"], toolChoice: "named:email", offered: []string{"email"}}, tools: []string{"email"}, reason: stickyToolReasonToolChoiceNotAuto},
		"empty query":          {turn: stickyTurn{query: " "}, reason: stickyToolReasonEmptyQuery},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{}), nil)
			result := h.run(tc.turn)
			require.True(t, result.response)
			require.Equal(t, tc.tools, result.tools)
			require.Equal(t, stickyToolOutcomeStateless, result.receipt.Outcome)
			require.Equal(t, tc.reason, result.receipt.Reason)
			require.Zero(t, h.store.operations())
		})
	}
}

// A store that fails or times out never fails the request: the current
// eligible stateless selection is emitted with a bounded reason.
func TestStickyStoreFailureFallsBackToStatelessSelection(t *testing.T) {
	cases := map[string]struct {
		fail   error
		reason string
	}{
		"unavailable": {fail: errors.New("store unavailable"), reason: stickyToolReasonStoreError},
		"timeout":     {fail: context.DeadlineExceeded, reason: stickyToolReasonStoreTimeout},
		"corrupted":   {fail: sessiontools.ErrStateCorrupted, reason: stickyToolReasonStateCorrupted},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
			h.run(stickyTurn{query: stickyTestQueries["weather"]})
			h.store.mu.Lock()
			h.store.loadFail = tc.fail
			h.store.mu.Unlock()
			result := h.run(stickyTurn{query: stickyTestQueries["email"]})
			require.Equal(t, []string{"email"}, result.tools, "fallback is the current stateless selection, not the stored set")
			require.Equal(t, stickyToolOutcomeStateless, result.receipt.Outcome)
			require.Equal(t, tc.reason, result.receipt.Reason)
		})
	}
}

// Expired state is never reused; the next trusted request starts fresh.
func TestStickyExpiredStateStartsFresh(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}),
		&config.ToolSessionStoreConfig{TTLSeconds: intPtr(60)})
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	h.clock.Advance(59 * time.Second)
	require.Equal(t, []string{"weather", "calendar"}, h.run(stickyTurn{query: stickyTestQueries["calendar"]}).tools, "sliding TTL keeps active state")
	h.clock.Advance(61 * time.Second)
	expired := h.run(stickyTurn{query: stickyTestQueries["email"]})
	require.Equal(t, []string{"email"}, expired.tools)
	require.Equal(t, "seeded", expired.receipt.Outcome)
}

// Adjacent integers above 2^53 are different schemas: replacing one with the
// other changes the catalog, invalidates reuse, and reaches the provider
// exactly.
func TestStickySchemaChangeAboveTwoToTheFiftyThirdInvalidatesReuse(t *testing.T) {
	for _, format := range stickyTestFormats {
		t.Run(string(format), func(t *testing.T) {
			h := newStickyHarness(t, format, stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
			h.run(stickyTurn{query: stickyTestQueries["weather"]})
			h.router.ToolsDatabase = newStickyToolsDatabase(t, h.provider,
				`{"type":"object","properties":{"n":{"type":"integer","minimum":9007199254740992}}}`,
				"weather", "calendar", "email", "files")
			changed := h.run(stickyTurn{query: stickyTestQueries["calendar"]})
			require.Equal(t, []string{"calendar"}, changed.tools)
			require.Equal(t, "reset", changed.receipt.Outcome)
			require.Equal(t, "catalog_changed", changed.receipt.Reason)
			require.Contains(t, string(changed.body), "9007199254740992")
			require.NotContains(t, string(changed.body), "9007199254740993")
		})
	}
}

// A retrieval failure fails the request instead of forwarding the tools it
// arrived with, unless the decision explicitly falls back to no tools.
func TestStickyRetrievalFailureDoesNotForwardOriginalTools(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{}), nil)
	h.provider.fail[stickyTestQueries["weather"]] = true
	failed := h.run(stickyTurn{query: stickyTestQueries["weather"], offered: []string{"email"}})
	require.False(t, failed.response)
	require.Equal(t, 503, failed.status)
	require.Equal(t, "tool_selection_unavailable", failed.ctx.ImmediateProtocolError.Code)
	require.Equal(t, stickyToolReasonRetrievalFailed, failed.receipt.Reason)

	fallback := true
	h.decision = stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1, FallbackToEmpty: &fallback,
		Sticky: &config.StickyToolSelectionConfig{Enabled: true},
	}, supportedStickyToolsConfig())
	empty := h.run(stickyTurn{query: stickyTestQueries["weather"], offered: []string{"email"}})
	require.True(t, empty.response)
	require.Empty(t, empty.tools)
	require.Equal(t, stickyToolReasonRetrievalFailed, empty.receipt.Reason)
}

// The bounded receipt reaches Replay once its record exists, carrying only
// an outcome, a reason, and counts.
func TestStickyReceiptReachesReplayWithoutContent(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{}), nil)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	h.router.ReplayRecorder = recorder
	result := h.run(stickyTurn{query: stickyTestQueries["weather"]})
	require.Len(t, result.ctx.pendingStickyToolOutcomes, 1, "selection runs before Replay starts")

	replayConfig := config.DefaultRouterReplayPluginConfig()
	result.ctx.RouterReplayPluginConfig = &replayConfig
	result.ctx.RequestID = "sticky-replay"
	h.router.startRouterReplay(result.ctx, h.model, h.model, h.decision.Name)
	require.NotEmpty(t, result.ctx.RouterReplayID)
	require.Empty(t, result.ctx.pendingStickyToolOutcomes)
	record, ok := recorder.GetRecord(result.ctx.RouterReplayID)
	require.True(t, ok)
	var outcomes []routerreplay.Outcome
	for _, outcome := range record.Outcomes {
		if outcome.Target == routerreplay.StickyToolSelectionTarget {
			outcomes = append(outcomes, outcome)
		}
	}
	require.Len(t, outcomes, 1)
	require.Equal(t, "seeded", outcomes[0].Verdict)
	require.Equal(t, map[string]string{"selected": "1", "reused": "0", "added": "1", "pinned": "0", "removed": "0"}, outcomes[0].Metadata)
	encoded, err := json.Marshal(outcomes[0])
	require.NoError(t, err)
	for _, secret := range []string{"weather", "user-a", "session-1", "sticky-test-secret", "vsr:st"} {
		require.NotContains(t, string(encoded), secret, "receipts must not carry tool, identity, or key material")
	}
}

// Missing or stale availability narrows tools through trusted facts before
// selection, so sticky state is never read and a retained tool is not
// restored.
func TestStickyStaleTrustedFactsSkipSessionState(t *testing.T) {
	toolsCfg := supportedStickyToolsConfig()
	toolsCfg.TrustedFacts.TrustSources = append(toolsCfg.TrustedFacts.TrustSources, config.TrustedSourceRuntimeFresh)
	toolsCfg.TrustedFacts.FreshnessSeconds = 60
	h := newStickyHarness(t, "openai.chat.v1", stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
		Sticky: &config.StickyToolSelectionConfig{Enabled: true},
	}, toolsCfg), nil)
	require.Equal(t, []string{"weather"}, h.run(stickyTurn{query: stickyTestQueries["weather"]}).tools)
	operations := h.store.operations()

	trustedFactsNow = func() time.Time { return time.Now().Add(2 * time.Minute) }
	t.Cleanup(func() { trustedFactsNow = time.Now })
	stale := h.run(stickyTurn{query: stickyTestQueries["calendar"]})
	require.Empty(t, stale.tools, "narrowing keeps no unrequested retained tool")
	require.Equal(t, stickyToolReasonTrustedFactsRestrict, stale.receipt.Reason)
	require.Equal(t, operations, h.store.operations(), "stale facts must not reach the store")
}

// Concurrent turns of one session stay within the bounded planner contract.
func TestStickyConcurrentTurnsStayBounded(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(2)}), nil)
	queries := []string{stickyTestQueries["weather"], stickyTestQueries["calendar"], stickyTestQueries["email"], stickyTestQueries["files"]}
	var wg sync.WaitGroup
	results := make([]stickyTurnResult, 16)
	errs := make([]error, len(results))
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = h.dispatch(stickyTurn{query: queries[i%len(queries)]})
		}(i)
	}
	wg.Wait()
	for i, result := range results {
		require.NoError(t, errs[i])
		require.True(t, result.response)
		require.LessOrEqual(t, len(result.tools), 2)
		require.NotEmpty(t, result.receipt.Outcome)
	}
}

// Identical emitted tools leave the request untouched; any representation
// change, including nil versus an explicit empty list, is a new generation.
func TestApplyStickyToolsChangesGenerationOnlyForNewRepresentation(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	request := &llmprotocol.Request{Tools: []llmprotocol.Tool{{Name: "weather", InputSchema: schema}}}
	applyStickyTools(request, []llmprotocol.Tool{{Name: "weather", InputSchema: schema}}, true)
	require.Zero(t, request.Generation)

	applyStickyTools(request, []llmprotocol.Tool{{Name: "weather", InputSchema: json.RawMessage(`{"type": "object"}`)}}, true)
	require.Equal(t, uint64(1), request.Generation, "raw schema bytes are part of the emitted representation")

	applyStickyTools(request, nil, false)
	require.NotNil(t, request.Tools)
	require.Empty(t, request.Tools)
	require.Equal(t, uint64(2), request.Generation)
	applyStickyTools(request, nil, true)
	require.Nil(t, request.Tools)
	require.Equal(t, uint64(3), request.Generation)
}
