package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// A preview owns these copies. Reads never hydrate/expire live session memory,
// increment turns, record outcomes, or consume the production sampling source.
// Stores are read under their own locks; this is a captured-input receipt, not
// a transaction spanning independent learning, session, and telemetry stores.
type routerLearningPreviewSnapshot struct {
	CapturedAt    time.Time
	Seed          int64
	Experience    map[string]routerLearningModelExperience
	Sessions      map[string]previewSession
	LastModels    map[string]previewLastModel
	Warmth        map[string]previewWarmth
	LookupEntries map[string]lookuptable.Entry
	lookup        lookuptable.LookupTable
}
type (
	previewSession struct {
		Snapshot sessiontelemetry.RouterSessionSnapshot
		Found    bool
	}
	previewLastModel struct {
		Model string
		Idle  time.Duration
		Found bool
	}
	previewWarmth struct {
		Value float64
		Found bool
	}
)

func (r *OpenAIRouter) newLearningPreview(seed int64) (*routerLearningPreviewSnapshot, error) {
	snapshot := &routerLearningPreviewSnapshot{
		CapturedAt: time.Now().UTC(), Seed: seed,
		Experience: map[string]routerLearningModelExperience{}, Sessions: map[string]previewSession{},
		LastModels: map[string]previewLastModel{}, Warmth: map[string]previewWarmth{},
	}
	r.routerLearningMu.Lock()
	runtime := r.routerLearningRuntime
	r.routerLearningMu.Unlock()
	if runtime != nil && runtime.shared != nil {
		runtime.shared.mu.Lock()
		for key, experience := range runtime.shared.experience {
			if experience != nil {
				snapshot.Experience[key] = *experience
			}
		}
		runtime.shared.mu.Unlock()
	}
	if r.LookupTable != nil {
		source, ok := r.LookupTable.(interface {
			All() map[string]lookuptable.Entry
		})
		if !ok {
			return nil, fmt.Errorf("learning lookup table does not support a read-only snapshot")
		}
		snapshot.LookupEntries = source.All()
		table := lookuptable.NewMemoryStorage()
		for key, entry := range snapshot.LookupEntries {
			parsed, err := lookuptable.ParseKey(key)
			if err != nil {
				return nil, fmt.Errorf("learning lookup snapshot contains an invalid key")
			}
			if err := table.Set(parsed, entry); err != nil {
				return nil, err
			}
		}
		snapshot.lookup = table
	}
	return snapshot, nil
}

func (p *routerLearningPreviewSnapshot) session(key string) (sessiontelemetry.RouterSessionSnapshot, bool) {
	value, ok := p.Sessions[key]
	if !ok {
		value.Snapshot, value.Found = sessiontelemetry.PeekRouterSessionSnapshot(key, p.CapturedAt)
		p.Sessions[key] = value
	}
	return value.Snapshot, value.Found
}

func (p *routerLearningPreviewSnapshot) lastModel(key string) previewLastModel {
	value, ok := p.LastModels[key]
	if !ok {
		value.Model, value.Idle, value.Found = sessiontelemetry.PeekLastModelInfo(key, p.CapturedAt)
		p.LastModels[key] = value
	}
	return value
}

func (p *routerLearningPreviewSnapshot) warmth(model string) (float64, bool) {
	value, ok := p.Warmth[model]
	if !ok {
		value.Value, value.Found = estimateGateCacheWarmth(model, p.CapturedAt)
		p.Warmth[model] = value
	}
	return value.Value, value.Found
}

func (r *OpenAIRouter) learningExperience(ctx *RequestContext, decision string, tier int, model string) routerLearningModelExperience {
	if ctx == nil || ctx.learningPreview == nil {
		return r.routerLearningRuntimeState().experienceSnapshot(decision, tier, model)
	}
	for _, key := range []string{modelExperienceKey(decision, tier, model), modelExperienceKey("", tier, model), modelExperienceKey("", 0, model)} {
		if experience, ok := ctx.learningPreview.Experience[key]; ok {
			return experience
		}
	}
	return defaultRouterLearningModelExperience()
}

func (r *OpenAIRouter) prepareEvalRequest(input services.EvalModelSelectionInput, decision *config.Decision) (*RequestContext, error) {
	ctx := &RequestContext{
		Headers: map[string]string{}, VSRSelectedDecision: decision,
		SemanticRequest: input.SemanticRequest, VSRConversationFacts: input.ConversationFacts,
		VSRContextTokenCount: input.ContextTokenCount,
	}
	if recipe, ok := r.Config.RecipeByName(input.Recipe); ok {
		ctx.Routing.SelectRecipe(recipe)
	}
	seed := int64(0)
	if input.PreviewContext != nil {
		cfg := r.Config.RouterLearning.Protection
		ctx.Headers[cfg.HeaderName("session")] = strings.TrimSpace(input.PreviewContext.SessionID)
		ctx.Headers[cfg.HeaderName("conversation")] = strings.TrimSpace(input.PreviewContext.ConversationID)
		ctx.SessionID = strings.TrimSpace(input.PreviewContext.SessionID)
		if input.PreviewContext.SamplingSeed != nil {
			seed = *input.PreviewContext.SamplingSeed
		}
	}
	if r.Config.RouterLearning.Enabled {
		snapshot, err := r.newLearningPreview(seed)
		if err != nil {
			return nil, err
		}
		ctx.learningPreview = snapshot
	}
	if ctx.SemanticRequest != nil && len(ctx.SemanticRequest.Messages) != 0 {
		populateSemanticSessionIDIfNeeded(ctx)
		ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionMessages(ctx.SemanticRequest.Messages)) - 1
		ctx.HistoryTokenCount = historyTokensFromSemanticMessages(ctx.SemanticRequest.Messages)
	}
	if ctx.learningPreview != nil {
		key := routingSessionStateKey(ctx)
		last := ctx.learningPreview.lastModel(key)
		if last.Found {
			ctx.PreviousModel = last.Model
		}
		snapshot, found := ctx.learningPreview.session(key)
		if found {
			if ctx.PreviousModel == "" {
				ctx.PreviousModel = snapshot.CurrentModel
			}
			ctx.SessionIdleSeconds, ctx.SessionIdleKnown = snapshot.IdleFor.Seconds(), true
		} else if last.Found {
			ctx.SessionIdleSeconds, ctx.SessionIdleKnown = last.Idle.Seconds(), true
		}
	}
	return ctx, nil
}

func (r *OpenAIRouter) finishEvalLearning(ctx *RequestContext, selCtx *selection.SelectionContext, base *selection.SelectionResult, ref *config.ModelRef, method string) services.EvalModelSelection {
	selected, result := ref, base
	if ctx.learningPreview != nil {
		_, learned, candidate, _, err := r.applyRouterLearning(selCtx, base, ref, ctx)
		if err != nil {
			return evalSelectionUnavailable(err.Error())
		}
		selected, result = candidate, learned
	}
	reason := strings.TrimSpace(result.Reasoning)
	if reason == "" {
		reason = "selected by the live runtime selector"
	}
	output := selectedEvalModel(selected, method, boundedSelectionReasoning(reason))
	output.Provenance = &services.SelectionProvenance{Mode: "stateless", ConfigHash: r.Config.DocumentHash}
	if ctx.learningPreview == nil {
		if method != "static" && method != "single" {
			raw, err := json.Marshal(base)
			if err != nil {
				return evalSelectionUnavailable("selector snapshot receipt could not be encoded")
			}
			digest := sha256.Sum256(raw)
			output.Provenance = &services.SelectionProvenance{
				Mode: "read_only_snapshot", ConfigHash: r.Config.DocumentHash,
				StateDependent: true, StateHash: hex.EncodeToString(digest[:]), CapturedAt: time.Now().UTC().Format(time.RFC3339Nano),
				Caveat: "This selector may consult live telemetry. The receipt captures its resulting scores, not an atomic runtime snapshot or a guarantee of a later selection.",
			}
		}
		return output
	}
	p := ctx.learningPreview
	receipt := struct {
		Snapshot *routerLearningPreviewSnapshot
		Base     *selection.SelectionResult
	}{p, base}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return evalSelectionUnavailable("learning snapshot receipt could not be encoded")
	}
	digest := sha256.Sum256(raw)
	provenance := &services.SelectionProvenance{
		Mode: "read_only_snapshot", ConfigHash: r.Config.DocumentHash,
		StateDependent: true, StateHash: hex.EncodeToString(digest[:]), CapturedAt: p.CapturedAt.Format(time.RFC3339Nano),
		Caveat: "Selection uses read-only captured learning/session inputs; later execution may differ if routing state, time, or request context changes. Independent stores are not one atomic transaction.",
	}
	if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation); ok && policy.Details.Adaptation != nil && policy.Details.Adaptation.sampling.used {
		provenance.Sampled = true
		seed := p.Seed
		provenance.SamplingSeed = &seed
		provenance.Caveat += " Exploration uses this preview's local seed; it does not predict a later production random draw."
	}
	output.Provenance = provenance
	output.Reason += "; read-only learning snapshot"
	return output
}
