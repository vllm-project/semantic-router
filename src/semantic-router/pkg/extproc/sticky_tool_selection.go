package extproc

import (
	"bytes"
	"context"
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// Closed vocabulary for sticky selection that used no session state. Each is
// bounded and safe as a log or Replay label; the identity reasons in
// sticky_tool_identity.go complete it.
const (
	stickyToolOutcomeStateless = "stateless"

	stickyToolReasonRuntimeUnavailable    = "runtime_unavailable"
	stickyToolReasonTrustedFactsRestrict  = "trusted_facts_restricted"
	stickyToolReasonTrustedFactsNotAllow  = "trusted_facts_not_allowed"
	stickyToolReasonCapabilityUnsupported = "capability_unsupported"
	stickyToolReasonInvalidCatalog        = "invalid_catalog"
	stickyToolReasonRetrievalFailed       = "retrieval_failed"
	stickyToolReasonToolChoiceNotAuto     = "tool_choice_not_auto"
	stickyToolReasonEmptyQuery            = "empty_query"
	stickyToolReasonStoreClosed           = "store_closed"
	stickyToolReasonStoreTimeout          = "store_timeout"
	stickyToolReasonRetryExhausted        = "retry_exhausted"
	stickyToolReasonStateCorrupted        = "state_corrupted"
	stickyToolReasonStateTooLarge         = "state_too_large"
	stickyToolReasonTurnExhausted         = "turn_exhausted"
	stickyToolReasonRevisionExhausted     = "revision_exhausted"
	stickyToolReasonInvalidSelection      = "invalid_selection"
	stickyToolReasonStoreError            = "store_error"
)

// stickyStrategyIDLimit matches the planner's strategy identifier bound.
const stickyStrategyIDLimit = 128

type stickyCatalogEntry struct {
	tool        llmprotocol.Tool
	fingerprint string
}

// stickyToolScope is a sticky decision's request-local evidence, resolved
// before relevance ranking: the current eligible catalog and its definitions,
// the full policy, catalog, and capability fingerprints, and the first hard
// constraint that forbids session state, if any. Nothing in it comes from
// stored state.
type stickyToolScope struct {
	plugin          *config.ToolSelectionPluginConfig
	eligible        []llmprotocol.Tool
	byName          map[string]stickyCatalogEntry
	fingerprints    sessiontools.Fingerprints
	recipe          string
	fallbackToEmpty bool
	stateless       string
}

// newStickyToolScope returns nil unless ts enables sticky selection. catalog
// is the current definition source: the configured database in add mode, or
// the request's offered tools in filter mode. Only its policy-eligible part
// is ever ranked, reused, or emitted.
func (r *OpenAIRouter) newStickyToolScope(
	request *llmprotocol.Request,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
	catalog []llmprotocol.Tool,
) *stickyToolScope {
	if !ts.StickyEnabled() {
		return nil
	}
	scope := &stickyToolScope{
		plugin:          ts,
		fallbackToEmpty: r.effectiveToolSelectionFallback(ts),
		recipe:          string(ctx.Routing.RecipeName()),
		stateless:       r.stickyToolHardConstraint(request, ctx, toolsCfg),
	}
	advanced := mergeToolSelectionAdvanced(ts, r.Config.Tools.AdvancedFiltering, toolsCfg)
	if reason := scope.resolveCatalog(catalog, advanced); reason != "" && scope.stateless == "" {
		scope.stateless = reason
	}
	scope.fingerprints = sessiontools.Fingerprints{
		Policy:     r.stickyToolPolicyFingerprint(ts, toolsCfg),
		Catalog:    tools.ToolCatalogFingerprint(scope.eligible),
		Capability: r.stickyToolCapabilityFingerprint(ctx),
	}
	return scope
}

// stickyToolHardConstraint returns the first current fact that forbids
// session state. Trusted facts must allow tool use both for candidate
// selection and for the final model that receives the request, and the
// dispatch target must be able to carry tools. Missing, stale, or unknown
// facts never allow.
func (r *OpenAIRouter) stickyToolHardConstraint(request *llmprotocol.Request, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig) string {
	if r.stickyTools == nil {
		return stickyToolReasonRuntimeUnavailable
	}
	if !toolsCfg.TrustedFactsEnabled() {
		return stickyToolReasonTrustedFactsNotAllow
	}
	available := r.trustedFactsToolsAvailable(toolsCfg.TrustedFacts.FreshnessSeconds)
	for _, stage := range []llmprotocol.TrustedStage{trustedFactsRequestStage, llmprotocol.TrustedStageFinal} {
		facts, ok := resolveTrustedFactsGate(request, toolsCfg, available, stage)
		if !ok || llmprotocol.EvaluateTrustedFacts(facts) != llmprotocol.TrustedAllow {
			return stickyToolReasonTrustedFactsNotAllow
		}
	}
	if ctx.TargetFormat == "" {
		return stickyToolReasonCapabilityUnsupported
	}
	if capabilities, ok := r.codecCapabilitiesForFormat(ctx.TargetFormat); !ok || !capabilities.Supports(llmprotocol.CapabilityTools) {
		return stickyToolReasonCapabilityUnsupported
	}
	return ""
}

// resolveCatalog keeps the catalog tools the decision's allow and block lists
// permit, matched like retrieval matches them. A duplicate name makes the
// catalog ambiguous, so only its first definition stays eligible and session
// state is not used.
func (s *stickyToolScope) resolveCatalog(catalog []llmprotocol.Tool, advanced *config.AdvancedToolFilteringConfig) string {
	var allow, block map[string]struct{}
	if advanced != nil && advanced.Enabled {
		allow, block = stickyToolNameSet(advanced.AllowTools), stickyToolNameSet(advanced.BlockTools)
	}
	reason := ""
	s.byName = make(map[string]stickyCatalogEntry, len(catalog))
	s.eligible = make([]llmprotocol.Tool, 0, len(catalog))
	for _, tool := range catalog {
		name := strings.TrimSpace(tool.Name)
		key := strings.ToLower(name)
		if _, blocked := block[key]; blocked {
			continue
		}
		if _, allowed := allow[key]; len(allow) > 0 && !allowed {
			continue
		}
		if _, duplicate := s.byName[name]; duplicate {
			reason = stickyToolReasonInvalidCatalog
			continue
		}
		s.byName[name] = stickyCatalogEntry{tool: tool, fingerprint: tools.ToolDefinitionFingerprint(tool)}
		s.eligible = append(s.eligible, tool)
	}
	return reason
}

func stickyToolNameSet(names []string) map[string]struct{} {
	set := make(map[string]struct{}, len(names))
	for _, name := range names {
		if key := strings.ToLower(strings.TrimSpace(name)); key != "" {
			set[key] = struct{}{}
		}
	}
	return set
}

// rank keeps the strategy's candidates that match a current eligible
// definition exactly, in the strategy's order, and returns that definition.
func (s *stickyToolScope) rank(candidates []llmprotocol.Tool) []llmprotocol.Tool {
	ranked := make([]llmprotocol.Tool, 0, len(candidates))
	seen := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		name := strings.TrimSpace(candidate.Name)
		entry, ok := s.byName[name]
		if !ok || seen[name] || entry.fingerprint != tools.ToolDefinitionFingerprint(candidate) {
			continue
		}
		seen[name] = true
		ranked = append(ranked, entry.tool)
	}
	return ranked
}

// selectionInput freezes the planner evidence. Ranked candidates keep the
// strategy's order through strictly decreasing scores, so the strategy, not
// the planner, decides relevance.
func (s *stickyToolScope) selectionInput(ranked []llmprotocol.Tool, called []string, strategyID string, maxStateBytes int) sessiontools.SelectionInput {
	eligible := make([]sessiontools.ToolIdentity, 0, len(s.eligible))
	for _, tool := range s.eligible {
		name := strings.TrimSpace(tool.Name)
		eligible = append(eligible, sessiontools.ToolIdentity{Name: name, DefinitionFingerprint: s.byName[name].fingerprint})
	}
	candidates := make([]sessiontools.RankedTool, 0, len(ranked))
	for i, tool := range ranked {
		name := strings.TrimSpace(tool.Name)
		candidates = append(candidates, sessiontools.RankedTool{
			ToolIdentity: sessiontools.ToolIdentity{Name: name, DefinitionFingerprint: s.byName[name].fingerprint},
			Score:        float64(len(ranked) - i),
		})
	}
	if len(strategyID) > stickyStrategyIDLimit {
		strategyID = strategyID[:stickyStrategyIDLimit]
	}
	sticky := s.plugin.Sticky
	return sessiontools.SelectionInput{
		Eligible: eligible,
		Ranked:   candidates,
		Called:   called,
		Bounds: sessiontools.SelectionBounds{
			MaxTools:           sticky.EffectiveMaxTools(),
			MaxNewToolsPerTurn: sticky.EffectiveMaxNewToolsPerTurn(),
			PinCalledTools:     sticky.EffectivePinCalledTools(),
			MaxStateBytes:      maxStateBytes,
		},
		Fingerprints: s.fingerprints,
		StrategyID:   strategyID,
	}
}

// rehydrate returns the current definitions for the planner's identities,
// in its order, rechecking each against the current eligible catalog. A
// stored name or fingerprint can never restore a definition that is absent
// or changed now.
func (s *stickyToolScope) rehydrate(selected []sessiontools.ToolIdentity) []llmprotocol.Tool {
	emitted := make([]llmprotocol.Tool, 0, len(selected))
	for _, identity := range selected {
		entry, ok := s.byName[identity.Name]
		if !ok || entry.fingerprint != identity.DefinitionFingerprint {
			continue
		}
		emitted = append(emitted, entry.tool)
	}
	return emitted
}

// stickyCalledTools returns the eligible tool names that the normalized
// assistant history called: tool-call blocks in the neutral request,
// including history materialized from a Responses previous_response_id. They
// are pinning preferences only; an observation cannot make a tool eligible.
func stickyCalledTools(request *llmprotocol.Request, scope *stickyToolScope) []string {
	called := make([]string, 0)
	seen := make(map[string]bool)
	for _, name := range extractSemanticRequestSignals(request).AssistantToolNames {
		name = strings.TrimSpace(name)
		if _, eligible := scope.byName[name]; !eligible || seen[name] {
			continue
		}
		seen[name] = true
		called = append(called, name)
	}
	return called
}

// finalizeStickyToolSelection applies a sticky decision's selection. Session
// state is read and written only for a trusted identity whose current facts
// allow it; everything else, including any store failure, emits the current
// eligible stateless ranking instead. A retrieval failure without an
// explicit empty fallback fails the request rather than forwarding the
// original tools.
func (r *OpenAIRouter) finalizeStickyToolSelection(
	request *llmprotocol.Request,
	ctx *RequestContext,
	result toolSelectionResult,
	scope *stickyToolScope,
) error {
	if result.err != nil {
		if !scope.fallbackToEmpty {
			r.recordStickyToolReceipt(ctx, routerreplay.StickyToolSelectionReceipt{
				Outcome: stickyToolOutcomeStateless, Reason: stickyToolReasonRetrievalFailed,
			})
			return errStickyToolSelectionFailed{cause: result.err}
		}
		return r.emitStatelessStickyTools(request, ctx, scope, nil, stickyToolReasonRetrievalFailed)
	}
	ranked := scope.rank(result.tools)
	if scope.stateless != "" {
		return r.emitStatelessStickyTools(request, ctx, scope, ranked, scope.stateless)
	}
	identity := ResolveStickyToolIdentity(ctx, scope.recipe, scope.fingerprints.Policy)
	if !identity.Trusted {
		return r.emitStatelessStickyTools(request, ctx, scope, ranked, identity.Reason)
	}
	update, err := r.stickyTools.manager.Update(stickyToolContext(ctx), sessiontools.UpdateRequest{
		Key:   identity.StorageKey,
		Quota: identity.QuotaKey,
		Selection: scope.selectionInput(
			ranked, stickyCalledTools(request, scope), result.strategyID, r.stickyTools.maxStateBytes,
		),
	})
	if err != nil {
		return r.emitStatelessStickyTools(request, ctx, scope, ranked, stickyToolStoreReason(err))
	}
	emitted := scope.rehydrate(update.Tools)
	applyStickyTools(request, emitted, scope.fallbackToEmpty)
	receipt := update.Receipt
	r.recordStickyToolReceipt(ctx, routerreplay.StickyToolSelectionReceipt{
		Outcome: string(receipt.Outcome), Reason: string(receipt.Reason), Selected: len(emitted),
		Reused: receipt.Reused, Added: receipt.Added, Pinned: receipt.Pinned, Removed: receipt.Removed,
	})
	return commitToolSelection(request, ctx)
}

func (r *OpenAIRouter) emitStatelessStickyTools(
	request *llmprotocol.Request,
	ctx *RequestContext,
	scope *stickyToolScope,
	ranked []llmprotocol.Tool,
	reason string,
) error {
	applyStickyTools(request, ranked, scope.fallbackToEmpty)
	r.recordStickyToolReceipt(ctx, routerreplay.StickyToolSelectionReceipt{
		Outcome: stickyToolOutcomeStateless, Reason: reason, Selected: len(ranked),
	})
	return commitToolSelection(request, ctx)
}

// applyStickyTools publishes the final tool list and changes the request only
// when its emitted representation changes. An empty selection follows the
// decision's fallback: no tools field, or an explicit empty list.
func applyStickyTools(request *llmprotocol.Request, selected []llmprotocol.Tool, fallbackToEmpty bool) {
	var next []llmprotocol.Tool
	switch {
	case len(selected) > 0:
		next = selected
	case !fallbackToEmpty:
		next = []llmprotocol.Tool{}
	}
	if sameToolRepresentation(request.Tools, next) {
		return
	}
	request.Tools = next
	request.Generation++
}

func sameToolRepresentation(current, next []llmprotocol.Tool) bool {
	if (current == nil) != (next == nil) || len(current) != len(next) {
		return false
	}
	for i := range current {
		if tools.ToolDefinitionFingerprint(current[i]) != tools.ToolDefinitionFingerprint(next[i]) ||
			!bytes.Equal(current[i].InputSchema, next[i].InputSchema) {
			return false
		}
	}
	return true
}

func stickyToolContext(ctx *RequestContext) context.Context {
	if ctx != nil && ctx.TraceContext != nil {
		return ctx.TraceContext
	}
	return context.Background()
}

// stickyToolStoreReason maps a planner or store failure onto the closed
// reason vocabulary without carrying the error text.
func stickyToolStoreReason(err error) string {
	switch {
	case errors.Is(err, sessiontools.ErrStoreClosed):
		return stickyToolReasonStoreClosed
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		return stickyToolReasonStoreTimeout
	case errors.Is(err, sessiontools.ErrRetryExhausted):
		return stickyToolReasonRetryExhausted
	case errors.Is(err, sessiontools.ErrStateCorrupted):
		return stickyToolReasonStateCorrupted
	case errors.Is(err, sessiontools.ErrStateTooLarge):
		return stickyToolReasonStateTooLarge
	case errors.Is(err, sessiontools.ErrTurnExhausted):
		return stickyToolReasonTurnExhausted
	case errors.Is(err, sessiontools.ErrRevisionExhausted):
		return stickyToolReasonRevisionExhausted
	case errors.Is(err, sessiontools.ErrInvalidSelection):
		return stickyToolReasonInvalidSelection
	default:
		return stickyToolReasonStoreError
	}
}

// stickyToolPolicyFingerprint resolves the decision's effective policy,
// including inherited global defaults and the merged allow and block rules,
// before fingerprinting it.
func (r *OpenAIRouter) stickyToolPolicyFingerprint(ts *config.ToolSelectionPluginConfig, toolsCfg *config.ToolsPluginConfig) string {
	effective := *ts
	effective.Mode = stickyToolSelectionMode(ts)
	effective.AdvancedFiltering = mergeToolSelectionAdvanced(ts, r.Config.Tools.AdvancedFiltering, toolsCfg)
	if effective.Mode == config.ToolSelectionModeAdd {
		effective.TopK = effectivePluginToolTopK(ts, r.Config.Tools.TopK)
		if effective.SimilarityThreshold == nil {
			effective.SimilarityThreshold = r.Config.Tools.SimilarityThreshold
		}
	} else {
		threshold := toolSelectionFilterThreshold(ts)
		effective.RelevanceThreshold = &threshold
	}
	return tools.EffectiveToolPolicyFingerprint(&effective, r.effectiveToolSelectionFallback(ts), toolsCfg)
}

// stickyToolCapabilityFingerprint covers what the target can carry: the
// target wire format, its codec capabilities, and the selected model's
// declared capabilities.
func (r *OpenAIRouter) stickyToolCapabilityFingerprint(ctx *RequestContext) string {
	codec, _ := r.codecCapabilitiesForFormat(ctx.TargetFormat)
	names := make([]string, 0)
	for _, name := range codec.Names() {
		names = append(names, "codec:"+name)
	}
	if declared, ok := r.declaredModelCapabilities(ctx.RequestModel); ok {
		for _, name := range declared.Names() {
			names = append(names, "model:"+name)
		}
	}
	return tools.ToolCapabilityFingerprint(names, string(ctx.TargetFormat))
}

func stickyToolSelectionMode(ts *config.ToolSelectionPluginConfig) string {
	if mode := strings.TrimSpace(ts.Mode); mode != "" {
		return mode
	}
	return config.ToolSelectionModeAdd
}

// recordStickyToolReceipt keeps the bounded receipt on the request and in
// Replay. Selection runs before Replay creates its record, so the outcome
// waits on the request context until startRouterReplay appends it.
func (r *OpenAIRouter) recordStickyToolReceipt(ctx *RequestContext, receipt routerreplay.StickyToolSelectionReceipt) {
	logStickyToolReceipt(ctx, receipt)
	if ctx == nil {
		return
	}
	ctx.StickyToolReceipt = &receipt
	outcome := receipt.ReplayOutcome(stickyToolClock())
	if ctx.RouterReplayID == "" {
		ctx.pendingStickyToolOutcomes = append(ctx.pendingStickyToolOutcomes, outcome)
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil && r != nil {
		recorder = r.ReplayRecorder
	}
	appendStickyToolReplayOutcome(recorder, ctx.RouterReplayID, outcome)
}

// errStickyToolSelectionFailed reports that a sticky decision could not
// select tools and has no permitted fallback. The request must fail instead
// of continuing with the tools it arrived with.
type errStickyToolSelectionFailed struct{ cause error }

func (e errStickyToolSelectionFailed) Error() string {
	return "sticky tool selection failed: " + e.cause.Error()
}

func (e errStickyToolSelectionFailed) Unwrap() error { return e.cause }

// recordStickyToolBypass records a sticky decision whose request never
// reached selection, so no session state was read or written.
func (r *OpenAIRouter) recordStickyToolBypass(ctx *RequestContext, ts *config.ToolSelectionPluginConfig, reason string) {
	if !ts.StickyEnabled() {
		return
	}
	r.recordStickyToolReceipt(ctx, routerreplay.StickyToolSelectionReceipt{Outcome: stickyToolOutcomeStateless, Reason: reason})
}

// stickyToolSelectionDecision reports whether the request's decision enables
// sticky selection.
func stickyToolSelectionDecision(ctx *RequestContext) bool {
	return ctx != nil && ctx.VSRSelectedDecision != nil &&
		ctx.VSRSelectedDecision.GetToolSelectionConfig().StickyEnabled()
}
