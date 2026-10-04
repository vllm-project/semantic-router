package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Grounding-aware fusion scores each panel response for faithfulness before the
// judge synthesizes, then ranks/filters the panel so the judge works from the
// most-grounded responses. It makes NO extra LLM calls — it uses the router's
// hallucination (groundedness) detector.
//
// Reference selection (config.FusionGroundingReference*):
//   - context: score answers against provided RAG/tool context.
//   - panel:   score answers against each other: every peer answer is the
//     context the detector reads a response against (the panel acts as its
//     own mutual reference).
//   - hybrid:  use context when the request carries it, otherwise the panel.
//
// Honest framing: grounding measures faithfulness/consistency, not truth. With no
// authoritative source we can only down-weight the least-supported responses, not
// certify correctness.

// HallucinationDetectFunc returns the unsupported spans of answer relative to
// context, plus the detector's score: its highest hallucinated-token
// probability, or 0 when it reports none.
type HallucinationDetectFunc func(ctx context.Context, contextText, question, answer string) (unsupportedSpans []string, score float32, err error)

// GroundingBackends belongs to the request's prepared recipe and generation.
// Its model handles remain protected by the router's existing generation lease.
type GroundingBackends struct {
	Detect HallucinationDetectFunc
}

// groundingScore captures the per-response groundedness outcome (parallel to the
// panel slice it was computed from, before ranking).
type groundingScore struct {
	Model        string   `json:"model"`
	Score        float64  `json:"score"`
	FlaggedSpans []string `json:"flagged_spans,omitempty"`
	Dropped      bool     `json:"dropped,omitempty"`
}

// FusionGroundingTrace is attached to FusionTrace for observability.
type FusionGroundingTrace struct {
	ReferenceMode string           `json:"reference_mode,omitempty"`
	Policy        string           `json:"policy,omitempty"`
	Scores        []groundingScore `json:"scores,omitempty"`
}

// applyGrounding scores, ranks and filters the panel. It returns the (possibly
// reordered/filtered) panel that the judge should use, the per-response scores,
// the reference mode actually used, and an error only when on_error=fail.
// When grounding is disabled or unavailable (and on_error=skip), it returns the
// panel unchanged so Fusion behaves exactly as before.
func (l *FusionLooper) applyGrounding(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panel []*ModelResponse,
) (kept []*ModelResponse, scores []groundingScore, referenceMode string, err error) {
	if !cfg.GroundingEnabled || len(panel) == 0 {
		return panel, nil, "", nil
	}

	backends := req.Grounding
	if backends == nil {
		backends = l.grounding
	}
	if backends == nil {
		backends = &GroundingBackends{}
	}
	question := extractOriginalContent(req.OriginalRequest)
	contextText := extractGroundingContext(req.OriginalRequest)
	useContext := resolveGroundingReference(cfg.GroundingReference, contextText)

	if useContext {
		scores, err = scoreByContext(ctx, contextText, question, panel, backends.Detect)
		referenceMode = config.FusionGroundingReferenceContext
	} else {
		scores, err = scoreByPanel(ctx, question, panel, cfg, backends.Detect)
		referenceMode = config.FusionGroundingReferencePanel
	}
	if err != nil {
		if cfg.GroundingOnError == config.FusionOnErrorFail {
			return nil, nil, "", fmt.Errorf("fusion grounding failed: %w", err)
		}
		logging.ComponentWarnEvent("looper", "fusion_grounding_skipped", map[string]interface{}{
			"reference_mode": referenceMode,
			"error":          err.Error(),
		})
		return panel, nil, "", nil
	}

	// Policy decides what we do with the scores. Only `filter` drops responses;
	// `weight`/`annotate` keep the full panel and let the judge soft-weight from
	// the groundedness notes (the synthesis prompt is annotated in runFusionFinal).
	// Hard-dropping the least mutually-consistent response regresses quality on
	// contested factual items (see bench/grounded_fusion/FINDINGS.md), so it is no
	// longer the default.
	if cfg.GroundingPolicy == config.FusionGroundingPolicyFilter {
		kept = filterPanelByScore(panel, scores, cfg)
	} else {
		kept = panel
	}
	logging.ComponentEvent("looper", "fusion_grounding_applied", map[string]interface{}{
		"reference_mode": referenceMode,
		"policy":         cfg.GroundingPolicy,
		"panel_in":       len(panel),
		"panel_kept":     len(kept),
	})
	return kept, scores, referenceMode, nil
}

func resolveGroundingReference(mode, contextText string) (useContext bool) {
	switch strings.TrimSpace(mode) {
	case config.FusionGroundingReferenceContext:
		return true
	case config.FusionGroundingReferencePanel:
		return false
	default: // hybrid (and empty)
		return strings.TrimSpace(contextText) != ""
	}
}

// scoreByPanel scores each response by how well its peers support it — the
// panel as its own mutual reference. It routes through the shared
// peer-consistency verifier contract (issue #2857).
func scoreByPanel(ctx context.Context, question string, panel []*ModelResponse, cfg fusionExecutionConfig, detect HallucinationDetectFunc) ([]groundingScore, error) {
	if detect == nil {
		return nil, fmt.Errorf("hallucination detector backend not configured")
	}
	candidates, idx := groundingVerifierCandidates(panel)
	res, err := NewPeerConsistencyVerifier(detect, cfg.GroundingContradictionPenalty).
		Verify(ctx, &VerifierRequest{Task: question, Candidates: candidates})
	if err != nil {
		return nil, err
	}
	return groundingScoresFromVerifier(res, panel, idx), nil
}

// scoreByContext scores each response by its faithfulness to the provided context
// (fewer unsupported spans => higher score). It routes through the shared
// faithfulness verifier contract (issue #2857); scoring is unchanged.
func scoreByContext(ctx context.Context, contextText, question string, panel []*ModelResponse, detect HallucinationDetectFunc) ([]groundingScore, error) {
	if detect == nil {
		return nil, fmt.Errorf("hallucination detector backend not configured")
	}
	candidates, idx := groundingVerifierCandidates(panel)
	res, err := NewFaithfulnessVerifier(detect).
		Verify(ctx, &VerifierRequest{Task: question, TrustedContext: contextText, Candidates: candidates})
	if err != nil {
		return nil, err
	}
	return groundingScoresFromVerifier(res, panel, idx), nil
}

// groundingVerifierCandidates converts the panel to contract candidates with
// index-tagged opaque IDs, preserving order and skipping nil slots so results
// map back by index exactly like the legacy path.
func groundingVerifierCandidates(panel []*ModelResponse) ([]VerifierCandidate, []int) {
	candidates := make([]VerifierCandidate, 0, len(panel))
	idx := make([]int, 0, len(panel))
	for i, r := range panel {
		if r == nil {
			continue
		}
		candidates = append(candidates, VerifierCandidate{ID: fmt.Sprintf("panel-%d", i), Content: r.Content})
		idx = append(idx, i)
	}
	return candidates, idx
}

// groundingScoresFromVerifier rebuilds per-index grounding scores from a
// verifier result using the candidate order established by
// groundingVerifierCandidates. Peer-consistency flags reference peers by their
// internal candidate ID and are remapped to caller-visible model labels so
// traces, grounding spans, and the synthesis notes stay associated with real
// responses. Only peer-consistency flags are ID-based; faithfulness flags are
// arbitrary unsupported text spans and pass through verbatim (a literal span
// like "panel-0" must never be rewritten) (regression #2857).
func groundingScoresFromVerifier(res *VerifierResult, panel []*ModelResponse, idx []int) []groundingScore {
	scores := make([]groundingScore, len(panel))
	byID := make(map[string]groundingScore, len(res.Scores))
	for _, s := range res.Scores {
		byID[s.CandidateID] = groundingScore{Score: s.Confidence, FlaggedSpans: s.Flags}
	}
	var idToModel map[string]string
	if res.Kind == VerifierKindPeerConsistency {
		idToModel = make(map[string]string, len(idx))
		for _, i := range idx {
			idToModel[fmt.Sprintf("panel-%d", i)] = modelName(panel[i])
		}
	}
	for _, i := range idx {
		id := fmt.Sprintf("panel-%d", i)
		scores[i] = groundingScore{Model: modelName(panel[i])}
		if gs, ok := byID[id]; ok {
			scores[i].Score = gs.Score
			scores[i].FlaggedSpans = remapFlaggedSpans(gs.FlaggedSpans, idToModel)
		}
	}
	return scores
}

// remapFlaggedSpans resolves internal candidate IDs in flag spans to their
// caller-visible model labels; textual spans (e.g. hallucination-detector
// output) pass through unchanged.
func remapFlaggedSpans(spans []string, idToModel map[string]string) []string {
	if len(spans) == 0 {
		return nil
	}
	out := make([]string, 0, len(spans))
	for _, s := range spans {
		if model, ok := idToModel[s]; ok {
			out = append(out, model)
		} else {
			out = append(out, s)
		}
	}
	return out
}

// filterPanelByScore returns the panel ranked by score (desc), dropping responses
// below min_score while always keeping at least min_keep of the highest-scoring
// responses. It mutates scores[i].Dropped to record what was filtered out.
func filterPanelByScore(panel []*ModelResponse, scores []groundingScore, cfg fusionExecutionConfig) []*ModelResponse {
	minKeep := cfg.GroundingMinKeep
	if minKeep <= 0 {
		minKeep = 1
	}
	if minKeep > len(panel) {
		minKeep = len(panel)
	}

	order := make([]int, len(panel))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return scores[order[a]].Score > scores[order[b]].Score
	})

	kept := make([]*ModelResponse, 0, len(panel))
	for rank, idx := range order {
		if rank < minKeep || scores[idx].Score >= cfg.GroundingMinScore {
			kept = append(kept, panel[idx])
		} else {
			scores[idx].Dropped = true
		}
	}
	return kept
}

// extractGroundingContext returns the authoritative context for the request:
// the concatenation of system and tool message contents (where RAG/tool output
// is conventionally injected). Empty when the request carries no such context.
func extractGroundingContext(req *openai.ChatCompletionNewParams) string {
	if req == nil {
		return ""
	}
	data, err := json.Marshal(req)
	if err != nil {
		return ""
	}
	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return ""
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role != "system" && role != "tool" {
			continue
		}
		if content, ok := msg["content"].(string); ok && strings.TrimSpace(content) != "" {
			b.WriteString(content)
			b.WriteString("\n")
		}
	}
	return strings.TrimSpace(b.String())
}

// formatGroundingNotes renders a concise groundedness summary for the judge
// prompt so synthesis is told which responses were corroborated vs flagged.
func formatGroundingNotes(scores []groundingScore) string {
	if len(scores) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Groundedness notes (higher score = better supported; flagged = contradicted/unsupported):\n")
	for _, s := range scores {
		fmt.Fprintf(&b, "- %s: score %.2f", s.Model, s.Score)
		if s.Dropped {
			b.WriteString(" [dropped]")
		}
		if len(s.FlaggedSpans) > 0 {
			fmt.Fprintf(&b, " [flagged: %s]", strings.Join(s.FlaggedSpans, "; "))
		}
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

// groundingSynthesisNotes renders the groundedness notes for the final synthesis
// prompt. For the `weight` policy it prepends an explicit instruction to weight
// each panel answer by its score (while protecting a correct lone dissenter);
// `annotate` emits the notes without that instruction. `filter` returns empty —
// the panel was already pruned, so the judge needs no per-response weighting.
func groundingSynthesisNotes(scores []groundingScore, policy string) string {
	if policy == config.FusionGroundingPolicyFilter {
		return ""
	}
	notes := formatGroundingNotes(scores)
	if notes == "" {
		return ""
	}
	if policy == config.FusionGroundingPolicyWeight {
		return "Weight each panel answer by its groundedness score below: prefer better-supported answers and treat flagged/contradicted claims with extra skepticism. Do not discard a lower-scoring answer if it is the only one that is correct — consistency is not the same as correctness.\n\n" + notes
	}
	return notes
}

func modelName(r *ModelResponse) string {
	if r == nil {
		return ""
	}
	return r.Model
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
