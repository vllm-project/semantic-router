package looper

import (
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

	kept = filterPanelByScore(panel, scores, cfg)
	logging.ComponentEvent("looper", "fusion_grounding_applied", map[string]interface{}{
		"reference_mode": referenceMode,
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
			entail, contradict, err := groundingNLIClassify(p.Content, r.Content)
			if err != nil {
				return nil, err
			}
			sum += float64(entail) - penalty*float64(contradict)
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
