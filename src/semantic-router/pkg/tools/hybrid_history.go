package tools

import (
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type hybridHistoryResolved struct {
	historyHorizon             int
	minHistorySteps            int
	historyConfidenceThreshold float32
	wSemantic                  float32
	wHistory                   float32
	wPrior                     float32
	repetitionPenaltyStrength  float32
}

func resolveHybridHistoryParams(h *config.HybridHistoryToolRetrievalConfig) hybridHistoryResolved {
	// Defaults avoid inventing tuned coefficients: equal weights = maximum-entropy prior for the
	// three additive terms until metrics-driven calibration; repetition penalty off until enabled;
	// confidence threshold 0 = do not downgrade to semantic-only based on historySignalStrength
	// (only min_history_steps still triggers fallback).
	const (
		defHorizon     = 8
		defMinSteps    = 1
		defConfThresh  = float32(0)
		defWSem        = float32(1)
		defWHist       = float32(1)
		defWPrior      = float32(1)
		defRepStrength = float32(0)
	)
	out := hybridHistoryResolved{
		historyHorizon:             defHorizon,
		minHistorySteps:            defMinSteps,
		historyConfidenceThreshold: defConfThresh,
		wSemantic:                  defWSem,
		wHistory:                   defWHist,
		wPrior:                     defWPrior,
		repetitionPenaltyStrength:  defRepStrength,
	}
	if h == nil {
		return out
	}
	if h.HistoryHorizon != nil && *h.HistoryHorizon > 0 {
		out.historyHorizon = *h.HistoryHorizon
	}
	if h.MinHistorySteps != nil && *h.MinHistorySteps > 0 {
		out.minHistorySteps = *h.MinHistorySteps
	}
	if h.HistoryConfidenceThreshold != nil {
		out.historyConfidenceThreshold = *h.HistoryConfidenceThreshold
	}
	if h.WeightSemantic != nil {
		out.wSemantic = *h.WeightSemantic
	}
	if h.WeightHistoryTransition != nil {
		out.wHistory = *h.WeightHistoryTransition
	}
	if h.WeightDecisionPrior != nil {
		out.wPrior = *h.WeightDecisionPrior
	}
	if h.RepetitionPenaltyStrength != nil {
		out.repetitionPenaltyStrength = *h.RepetitionPenaltyStrength
	}
	return out
}

func filterAndRankHybridHistory(
	query string,
	filtered []ToolSimilarity,
	topK int,
	advanced *config.AdvancedToolFilteringConfig,
	selectedCategory string,
	toolHistory []string,
	decisionConfidence float64,
) []openai.ChatCompletionToolParam {
	cfg := resolveHybridHistoryParams(advanced.HybridHistory)
	names := trimToolHistory(toolHistory, cfg.historyHorizon)
	a := newHybridRankArgs(query, names, cfg, advanced, selectedCategory, decisionConfidence)

	if shouldFallbackHybridToSemantic(names, cfg) {
		passing := make([]ToolSimilarity, 0, len(filtered))
		for _, candidate := range filtered {
			if _, ok := hybridCandidateScore(candidate, a); ok {
				passing = append(passing, candidate)
			}
		}
		return selectTopKBySimilarity(passing, topK)
	}

	scored := make([]scoredCandidate, 0, len(filtered))
	for _, candidate := range filtered {
		if h, ok := hybridCandidateScore(candidate, a); ok {
			scored = append(scored, scoredCandidate{ToolSimilarity: candidate, CombinedScore: h})
		}
	}

	if len(scored) == 0 {
		return []openai.ChatCompletionToolParam{}
	}
	sortScoredCandidatesByCombinedThenSimilarity(scored)
	return topKToolsFromScored(scored, topK)
}

type hybridRankArgs struct {
	querySet           map[string]struct{}
	minOverlap         int
	minCombined        float32
	lastTool           string
	names              []string
	wSum               float32
	cfg                hybridHistoryResolved
	repStrength        float32
	selectedCategory   string
	decisionConfidence float64
}

func newHybridRankArgs(
	query string,
	names []string,
	cfg hybridHistoryResolved,
	advanced *config.AdvancedToolFilteringConfig,
	selectedCategory string,
	decisionConfidence float64,
) hybridRankArgs {
	lastTool := ""
	if len(names) > 0 {
		lastTool = names[len(names)-1]
	}
	minO, minC := minLexicalOverlapAndMinCombined(advanced)
	w := cfg.wSemantic + cfg.wHistory + cfg.wPrior
	if w <= 0 {
		w = 1
	}
	return hybridRankArgs{
		querySet:           tokenSet(tokenize(query)),
		minOverlap:         minO,
		minCombined:        minC,
		lastTool:           lastTool,
		names:              names,
		wSum:               w,
		cfg:                cfg,
		repStrength:        cfg.repetitionPenaltyStrength,
		selectedCategory:   selectedCategory,
		decisionConfidence: decisionConfidence,
	}
}

func hybridCandidateScore(candidate ToolSimilarity, a hybridRankArgs) (float32, bool) {
	nameT := tokenize(candidate.Entry.Tool.Function.Name)
	descT := tokenize(candidate.Entry.Description)
	catT := tokenize(candidate.Entry.Category)
	lexicalTokens := make([]string, 0, len(nameT)+len(descT)+len(catT))
	lexicalTokens = append(lexicalTokens, nameT...)
	lexicalTokens = append(lexicalTokens, descT...)
	lexicalTokens = append(lexicalTokens, catT...)

	lexicalOverlap := countOverlap(a.querySet, lexicalTokens)
	if a.minOverlap > 0 && lexicalOverlap < a.minOverlap {
		return 0, false
	}

	sim := clamp01(candidate.Similarity)
	candName := candidate.Entry.Tool.Function.Name
	hTrans := historyTransitionScore(a.lastTool, candName, a.names)
	prior := decisionCategoryPriorScore(a.selectedCategory, candidate.Entry.Category, a.decisionConfidence)
	rep := repetitionPenalty(candName, a.names, a.repStrength)

	hyb := (a.cfg.wSemantic*sim + a.cfg.wHistory*hTrans + a.cfg.wPrior*prior) / a.wSum
	hyb -= rep
	if hyb < 0 {
		hyb = 0
	}
	if hyb < a.minCombined {
		return 0, false
	}
	return hyb, true
}

func trimToolHistory(names []string, horizon int) []string {
	if len(names) == 0 || horizon <= 0 {
		return names
	}
	if len(names) <= horizon {
		return append([]string(nil), names...)
	}
	return append([]string(nil), names[len(names)-horizon:]...)
}

func shouldFallbackHybridToSemantic(names []string, cfg hybridHistoryResolved) bool {
	if len(names) < cfg.minHistorySteps {
		return true
	}
	if cfg.historyConfidenceThreshold > 0 &&
		historySignalStrength(names, cfg.historyHorizon) < cfg.historyConfidenceThreshold {
		return true
	}
	return false
}

// historySignalStrength estimates how much we can trust bigram/repetition structure in the window.
func historySignalStrength(names []string, horizon int) float32 {
	if len(names) == 0 {
		return 0
	}
	cov := float32(len(names)) / float32(max(1, horizon))
	uniq := map[string]struct{}{}
	for _, n := range names {
		uniq[n] = struct{}{}
	}
	diversity := float32(1)
	if len(uniq) == 1 && len(names) >= 4 {
		diversity = 0.35
	}
	return minFloat32(1, cov*diversity)
}

func historyTransitionScore(lastTool, candidateName string, history []string) float32 {
	if lastTool == "" {
		return 0.5
	}
	var fromLast, toCand int
	for i := 0; i < len(history)-1; i++ {
		if history[i] == lastTool {
			fromLast++
			if history[i+1] == candidateName {
				toCand++
			}
		}
	}
	if fromLast > 0 {
		return float32(toCand) / float32(fromLast)
	}
	// No observed (last->*) edge: discourage immediate same-tool repeat.
	if candidateName == lastTool {
		return 0.12
	}
	return 0.62
}

func decisionCategoryPriorScore(selectedCategory, toolCategory string, decisionConfidence float64) float32 {
	cat := categoryMatchScore(selectedCategory, toolCategory)
	d := float32(decisionConfidence)
	if decisionConfidence != decisionConfidence { // NaN
		d = 0
	}
	if d < 0 {
		d = 0
	}
	if d > 1 {
		d = 1
	}
	return cat * (0.45 + 0.55*d)
}

func repetitionPenalty(candidateName string, history []string, strength float32) float32 {
	if strength <= 0 || len(history) == 0 {
		return 0
	}
	var count int
	for _, h := range history {
		if h == candidateName {
			count++
		}
	}
	if count == 0 {
		return 0
	}
	// Penalize redundant re-selection; first occurrence is not penalized.
	excess := count - 1
	if excess < 0 {
		excess = 0
	}
	return strength * float32(excess) * 0.35
}

func minFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
