package looper

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Grounding-aware fusion scores each panel response for faithfulness before the
// judge synthesizes, then ranks/filters the panel so the judge works from the
// most-grounded responses. It makes NO extra LLM calls — it uses local encoder
// models (a hallucination/groundedness detector and an NLI entailment model).
//
// Reference selection (config.FusionGroundingReference*):
//   - context: score answers against provided RAG/tool context via the detector.
//   - panel:   score answers against each other via cross-model NLI (the panel
//     acts as its own mutual reference).
//   - hybrid:  use context when the request carries it, otherwise the panel.
//
// Honest framing: grounding measures faithfulness/consistency, not truth. With no
// authoritative source we can only down-weight the least-supported responses, not
// certify correctness.

// NLIClassifyFunc returns the entailment and contradiction probabilities for the
// hypothesis given the premise.
type NLIClassifyFunc func(premise, hypothesis string) (entailment, contradiction float32, err error)

// HallucinationDetectFunc returns the unsupported spans of answer relative to
// context, plus the detector's confidence.
type HallucinationDetectFunc func(context, question, answer string) (unsupportedSpans []string, confidence float32, err error)

// Backends are injected once at startup (see classification lifecycle) so the
// candle/CGO dependency stays out of this package's import graph and the scoring
// logic stays hermetically unit-testable.
var (
	groundingNLIClassify NLIClassifyFunc
	groundingDetect      HallucinationDetectFunc
)

// SetGroundingBackends wires the NLI + hallucination detection backends used by
// grounding-aware fusion. Safe to call again to replace them.
func SetGroundingBackends(nli NLIClassifyFunc, detect HallucinationDetectFunc) {
	groundingNLIClassify = nli
	groundingDetect = detect
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
	req *Request,
	cfg fusionExecutionConfig,
	panel []*ModelResponse,
) (kept []*ModelResponse, scores []groundingScore, referenceMode string, err error) {
	if !cfg.GroundingEnabled || len(panel) == 0 {
		return panel, nil, "", nil
	}

	question := extractOriginalContent(req.OriginalRequest)
	contextText := extractGroundingContext(req.OriginalRequest)
	useContext := resolveGroundingReference(cfg.GroundingReference, contextText)

	if useContext {
		scores, err = scoreByContext(contextText, question, panel, cfg)
		referenceMode = config.FusionGroundingReferenceContext
	} else {
		scores, err = scoreByPanel(panel, cfg)
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

// NLI input budgets. The underlying NLI encoder truncates its (premise [SEP]
// hypothesis) input to ~512 tokens. Panel answers routinely run to thousands of
// tokens, so a single document-vs-document call truncates the hypothesis away
// entirely and the model — seeing only half of one answer — predicts "neutral"
// for everything. To keep the hypothesis intact we score it sentence-by-sentence
// against bounded windows of the premise (SummaC/AlignScore-style), taking each
// hypothesis sentence's best-supporting premise window. Budgets are in runes
// (~4 chars/token, kept well under the 512-token cap with room for both sides).
const (
	nliSingleCallBudget    = 1600 // premise+hypothesis below this -> one fast call
	nliPremiseWindowChars  = 1200
	nliHypSentenceMaxChars = 600
	nliMaxHypSentences     = 16 // caps per-pair NLI calls (= sentences x windows)
	nliMaxPremiseWindows   = 2
)

// scoreByPanel scores each response by how well its peers entail (vs contradict)
// it — the panel as its own mutual reference.
func scoreByPanel(panel []*ModelResponse, cfg fusionExecutionConfig) ([]groundingScore, error) {
	if groundingNLIClassify == nil {
		return nil, fmt.Errorf("nli backend not configured")
	}
	penalty := cfg.GroundingNLIContradictionPenalty
	if penalty <= 0 {
		penalty = 1.0
	}
	scores := make([]groundingScore, len(panel))
	for i, r := range panel {
		scores[i].Model = modelName(r)
		if r == nil {
			continue
		}
		var sum float64
		var n int
		var flagged []string
		for j, p := range panel {
			if i == j || p == nil {
				continue
			}
			// Directional consistency: does peer p (premise) entail/contradict
			// response r (hypothesis)?
			entail, contradict, err := nliPairSignal(p.Content, r.Content)
			if err != nil {
				return nil, err
			}
			sum += entail - penalty*contradict
			n++
			if contradict > entail && contradict >= 0.5 {
				flagged = append(flagged, p.Model)
			}
		}
		raw := 0.0
		if n > 0 {
			raw = sum / float64(n)
		}
		// raw is in [-penalty, 1]; map to [0,1].
		scores[i].Score = clamp01((raw + penalty) / (1.0 + penalty))
		scores[i].FlaggedSpans = flagged
	}
	return scores, nil
}

// nliPairSignal returns the mean entailment and contradiction of premise toward
// hypothesis. Short inputs use a single NLI call (preserving the cheap path);
// long inputs are chunked so the hypothesis is never truncated away: each
// hypothesis sentence is scored against every bounded premise window and credited
// with its best-supporting window, then averaged over sentences.
func nliPairSignal(premise, hypothesis string) (entail, contradict float64, err error) {
	if runeLen(premise)+runeLen(hypothesis) <= nliSingleCallBudget {
		e, c, err := groundingNLIClassify(premise, hypothesis)
		return float64(e), float64(c), err
	}

	sentences := splitSentencesCapped(hypothesis, nliHypSentenceMaxChars, nliMaxHypSentences)
	windows := chunkTextCapped(premise, nliPremiseWindowChars, nliMaxPremiseWindows)
	if len(sentences) == 0 || len(windows) == 0 {
		// Nothing usable to chunk; fall back to a single (truncated) call.
		e, c, err := groundingNLIClassify(premise, hypothesis)
		return float64(e), float64(c), err
	}

	var sumE, sumC float64
	var n int
	for _, s := range sentences {
		bestSignal := math.Inf(-1)
		var bestE, bestC float64
		for _, w := range windows {
			e, c, err := groundingNLIClassify(w, s)
			if err != nil {
				return 0, 0, err
			}
			if signal := float64(e) - float64(c); signal > bestSignal {
				bestSignal, bestE, bestC = signal, float64(e), float64(c)
			}
		}
		sumE += bestE
		sumC += bestC
		n++
	}
	if n == 0 {
		return 0, 0, nil
	}
	return sumE / float64(n), sumC / float64(n), nil
}

// splitSentencesCapped splits text on sentence terminators / newlines, drops
// trivial fragments, hard-caps each sentence to maxChars (rune-safe) and limits
// the count to maxCount.
func splitSentencesCapped(text string, maxChars, maxCount int) []string {
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?' || r == '\n'
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if runeLen(f) < 12 { // skip trivial fragments (headings, list bullets)
			continue
		}
		out = append(out, truncateRunes(f, maxChars))
		if len(out) >= maxCount {
			break
		}
	}
	return out
}

// chunkTextCapped splits text into at most maxWindows contiguous windows of
// roughly maxChars runes each.
func chunkTextCapped(text string, maxChars, maxWindows int) []string {
	r := []rune(text)
	out := make([]string, 0, maxWindows)
	for start := 0; start < len(r) && len(out) < maxWindows; start += maxChars {
		end := start + maxChars
		if end > len(r) {
			end = len(r)
		}
		if w := strings.TrimSpace(string(r[start:end])); w != "" {
			out = append(out, w)
		}
	}
	return out
}

func runeLen(s string) int { return utf8.RuneCountInString(s) }

func truncateRunes(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	return string([]rune(s)[:max])
}

// scoreByContext scores each response by its faithfulness to the provided context
// (fewer unsupported spans => higher score).
func scoreByContext(contextText, question string, panel []*ModelResponse, cfg fusionExecutionConfig) ([]groundingScore, error) {
	if groundingDetect == nil {
		return nil, fmt.Errorf("hallucination detector backend not configured")
	}
	scores := make([]groundingScore, len(panel))
	for i, r := range panel {
		scores[i].Model = modelName(r)
		if r == nil {
			continue
		}
		spans, _, err := groundingDetect(contextText, question, r.Content)
		if err != nil {
			return nil, err
		}
		// 0 unsupported spans => 1.0; degrades as unsupported spans accumulate.
		scores[i].Score = 1.0 / (1.0 + float64(len(spans)))
		scores[i].FlaggedSpans = spans
	}
	return scores, nil
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
