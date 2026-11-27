package turnselection

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

type turnScore struct {
	relevance  float64
	recency    float64
	continuity float64
	total      float64
	boosted    bool
}

type scoringOutcome struct {
	scores              map[int]turnScore // keyed by Turn.Ordinal
	source              ScoringSource
	fallbackReason      string
	scoredTurns         int
	candidatesTruncated bool
	continuityBoosted   int
}

func scoreEligibleTurns(
	ctx context.Context,
	policy Policy,
	scorer contextcompression.RelevanceScorer,
	turns []Turn,
	eligibleIndices []int,
) (scoringOutcome, error) {
	outcome := scoringOutcome{
		scores: make(map[int]turnScore, len(eligibleIndices)),
		source: ScoringNone,
	}
	if len(eligibleIndices) == 0 {
		return outcome, nil
	}
	if ctx.Err() != nil {
		return outcome, ctx.Err()
	}

	// Bound candidate scanning work to the most recent MaxCandidateTurns
	// eligible turns. Older turns outside the cap receive zero relevance and
	// continuity so they are evicted before scored turns when budget is tight.
	window := eligibleIndices
	if policy.Scoring.MaxCandidateTurns > 0 && len(window) > policy.Scoring.MaxCandidateTurns {
		outcome.candidatesTruncated = true
		window = eligibleIndices[len(eligibleIndices)-policy.Scoring.MaxCandidateTurns:]
	}
	outcome.scoredTurns = len(window)

	query := extractLiveQuery(turns, policy.Scoring.MaxQueryBytes)
	docs := make([]string, len(window))
	for i, turnIdx := range window {
		docs[i] = turns[turnIdx].Text
	}

	lexicalRaw := bm25Scores(query, docs)
	lexicalNorm := normalizeNonNegativeScores(lexicalRaw)

	relevanceScores := lexicalNorm
	outcome.source = ScoringLexical

	if scorer != nil {
		extScores, fallbackReason, err := scoreWithExternalScorer(ctx, policy, scorer, query, docs, lexicalNorm)
		if err != nil {
			outcome.source = ScoringExternal
			outcome.fallbackReason = fallbackReason
			return outcome, err
		}
		if fallbackReason == FallbackNone {
			relevanceScores = extScores
			outcome.source = ScoringExternal
		} else {
			outcome.source = ScoringLexicalFallback
			outcome.fallbackReason = fallbackReason
			if extScores != nil {
				relevanceScores = extScores
			}
		}
	}

	var continuityScores []float64
	if policy.Scoring.Continuity == ContinuityEnabled {
		continuityScores = computeContinuityScores(query, turns, window, relevanceScores, policy.Scoring.MaxContinuityAnchors)
	} else {
		continuityScores = make([]float64, len(window))
	}

	totalEligible := len(eligibleIndices)
	// Assign recency decay for every eligible turn (including any outside the
	// scoring window) using ordinal surviving position rather than TurnID deltas.
	for pos, turnIdx := range eligibleIndices {
		distance := (totalEligible - 1) - pos
		rec := 1.0 / float64(1+distance)
		outcome.scores[turns[turnIdx].Ordinal] = turnScore{
			recency: rec,
			total:   policy.Scoring.RecencyWeight * rec,
		}
	}

	for i, turnIdx := range window {
		ordinal := turns[turnIdx].Ordinal
		entry := outcome.scores[ordinal]
		entry.relevance = relevanceScores[i]
		entry.continuity = continuityScores[i]
		if entry.continuity > 0 {
			entry.boosted = true
			outcome.continuityBoosted++
		}
		entry.total = policy.Scoring.RelevanceWeight*entry.relevance +
			policy.Scoring.RecencyWeight*entry.recency +
			policy.Scoring.ContinuityWeight*entry.continuity
		outcome.scores[ordinal] = entry
	}
	return outcome, nil
}

func scoreWithExternalScorer(
	ctx context.Context,
	policy Policy,
	scorer contextcompression.RelevanceScorer,
	query string,
	docs []string,
	lexicalFallback []float64,
) ([]float64, string, error) {
	if ctx.Err() != nil {
		return nil, FallbackScorerTimeout, ctx.Err()
	}

	// Select a bounded subset of candidates (up to MaxScoredTurns and
	// MaxScoringBytes) for the external scorer, preferring stronger lexical
	// candidates and more recent turns deterministically.
	selectedIndices := selectExternalScorerSubset(docs, lexicalFallback, policy.Scoring.MaxScoredTurns, policy.Scoring.MaxScoringBytes)
	capped := len(selectedIndices) < len(docs)
	if len(selectedIndices) == 0 {
		if policy.Scoring.ScorerFallback == ScorerFallbackFail {
			return nil, FallbackCandidateCap, ErrScoringFailed
		}
		return lexicalFallback, FallbackCandidateCap, nil
	}

	texts := make([]string, len(selectedIndices))
	for i, idx := range selectedIndices {
		texts[i] = docs[idx]
	}

	callCtx := ctx
	var cancel context.CancelFunc
	if policy.Scoring.ScoringTimeout > 0 {
		callCtx, cancel = context.WithTimeout(ctx, policy.Scoring.ScoringTimeout)
		defer cancel()
	}

	rawScores, err := scorer.Score(callCtx, query, texts)
	if err != nil || callCtx.Err() != nil {
		reason := FallbackScorerError
		if callCtx.Err() != nil {
			reason = FallbackScorerTimeout
		}
		if ctx.Err() != nil {
			return nil, FallbackScorerTimeout, ctx.Err()
		}
		if policy.Scoring.ScorerFallback == ScorerFallbackFail {
			return nil, reason, ErrScoringFailed
		}
		return lexicalFallback, reason, nil
	}
	if len(rawScores) != len(texts) {
		if policy.Scoring.ScorerFallback == ScorerFallbackFail {
			return nil, FallbackLengthMismatch, ErrScoringFailed
		}
		return lexicalFallback, FallbackLengthMismatch, nil
	}
	for _, s := range rawScores {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			if policy.Scoring.ScorerFallback == ScorerFallbackFail {
				return nil, FallbackInvalidScore, ErrScoringFailed
			}
			return lexicalFallback, FallbackInvalidScore, nil
		}
	}

	normalized := normalizeFiniteScores(rawScores)
	merged := append([]float64(nil), lexicalFallback...)
	for i, idx := range selectedIndices {
		merged[idx] = normalized[i]
	}
	if capped {
		return merged, FallbackCandidateCap, nil
	}
	return merged, FallbackNone, nil
}

func selectExternalScorerSubset(
	docs []string,
	lexicalScores []float64,
	maxTurns int,
	maxBytes int,
) []int {
	if len(docs) == 0 {
		return nil
	}
	order := make([]int, len(docs))
	for i := range order {
		order[i] = i
	}
	// Prioritize higher lexical signal, breaking ties toward more recent turns.
	sort.SliceStable(order, func(left, right int) bool {
		li, ri := order[left], order[right]
		if lexicalScores[li] != lexicalScores[ri] {
			return lexicalScores[li] > lexicalScores[ri]
		}
		return li > ri
	})
	selected := make([]int, 0, len(docs))
	usedBytes := 0
	for _, idx := range order {
		if maxTurns > 0 && len(selected) >= maxTurns {
			break
		}
		docBytes := len(docs[idx])
		if maxBytes > 0 && len(selected) > 0 && usedBytes+docBytes > maxBytes {
			continue
		}
		if maxBytes > 0 && len(selected) == 0 && docBytes > maxBytes {
			continue
		}
		selected = append(selected, idx)
		usedBytes += docBytes
	}
	sort.Ints(selected)
	return selected
}

func bm25Scores(query string, docs []string) []float64 {
	scores := make([]float64, len(docs))
	queryTerms := filterBM25QueryTerms(tokenizeTerms(query))
	if len(queryTerms) == 0 || len(docs) == 0 {
		return scores
	}

	docTerms := make([][]string, len(docs))
	docFreq := make(map[string]int)
	totalLen := 0
	for i, doc := range docs {
		terms := tokenizeTerms(doc)
		docTerms[i] = terms
		totalLen += len(terms)
		seen := make(map[string]struct{}, len(terms))
		for _, term := range terms {
			if _, ok := seen[term]; ok {
				continue
			}
			seen[term] = struct{}{}
			docFreq[term]++
		}
	}
	avgLen := float64(totalLen) / float64(max(1, len(docs)))
	for i := range docs {
		scores[i] = bm25DocumentScore(queryTerms, docTerms[i], docFreq, len(docs), avgLen)
	}
	return scores
}

func bm25DocumentScore(
	queryTerms []string,
	docTerms []string,
	docFreq map[string]int,
	docCount int,
	avgDocLen float64,
) float64 {
	if len(queryTerms) == 0 || len(docTerms) == 0 {
		return 0
	}
	tfMap := make(map[string]int, len(docTerms))
	for _, term := range docTerms {
		tfMap[term]++
	}
	seenQuery := make(map[string]struct{}, len(queryTerms))
	const k1 = 1.2
	const b = 0.75
	score := 0.0
	for _, term := range queryTerms {
		if _, dup := seenQuery[term]; dup {
			continue
		}
		seenQuery[term] = struct{}{}
		tf := float64(tfMap[term])
		if tf == 0 {
			continue
		}
		df := float64(docFreq[term])
		idf := math.Log(1 + (float64(docCount)-df+0.5)/(df+0.5))
		lengthRatio := float64(len(docTerms)) / math.Max(1, avgDocLen)
		score += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+b*lengthRatio))
	}
	return score
}

func normalizeNonNegativeScores(raw []float64) []float64 {
	out := make([]float64, len(raw))
	peak := 0.0
	for _, s := range raw {
		if s > peak {
			peak = s
		}
	}
	if peak <= 0 {
		return out
	}
	for i, s := range raw {
		if s > 0 {
			out[i] = s / peak
		}
	}
	return out
}

func normalizeFiniteScores(raw []float64) []float64 {
	out := make([]float64, len(raw))
	if len(raw) == 0 {
		return out
	}
	minScore, maxScore := raw[0], raw[0]
	for _, s := range raw[1:] {
		if s < minScore {
			minScore = s
		}
		if s > maxScore {
			maxScore = s
		}
	}
	if maxScore == minScore {
		if maxScore > 0 {
			for i := range out {
				out[i] = 1
			}
		}
		return out
	}
	span := maxScore - minScore
	for i, s := range raw {
		out[i] = (s - minScore) / span
	}
	return out
}

// computeContinuityScores computes deterministic, bounded continuity signals:
//  1. Referential option-enumeration anchoring: when the live request refers to
//     an enumerated option/plan without repeating its name (for example,
//     "continue with the second option"), earlier turns that introduce multiple
//     options receive an explicit continuity anchor score.
//  2. Forward revision/correction linking: for up to maxAnchors top-anchored
//     turns, later eligible turns that share distinctive non-stopword lexical
//     overlap with an anchor AND carry explicit correction, rejection, or
//     state-update cues receive a continuity boost so superseding updates
//     survive alongside the older definition they modify.
func computeContinuityScores(
	query string,
	turns []Turn,
	window []int,
	relevanceScores []float64,
	maxAnchors int,
) []float64 {
	continuity := make([]float64, len(window))
	if len(window) == 0 {
		return continuity
	}

	referential := hasReferentialOptionCue(query)
	distinctTerms := make([]map[string]struct{}, len(window))
	docFreq := make(map[string]int)
	for i, turnIdx := range window {
		terms := distinctiveTermSet(turns[turnIdx].Text)
		distinctTerms[i] = terms
		for term := range terms {
			docFreq[term]++
		}
		if referential && introducesMultipleOptions(turns[turnIdx].Text) {
			continuity[i] = 0.75
		}
	}

	if maxAnchors <= 0 {
		return continuity
	}

	type anchorCandidate struct {
		windowIdx int
		strength  float64
	}
	anchors := make([]anchorCandidate, 0, len(window))
	for i := range window {
		strength := math.Max(relevanceScores[i], continuity[i])
		if strength > 0 {
			anchors = append(anchors, anchorCandidate{windowIdx: i, strength: strength})
		}
	}
	sort.SliceStable(anchors, func(left, right int) bool {
		if anchors[left].strength != anchors[right].strength {
			return anchors[left].strength > anchors[right].strength
		}
		return anchors[left].windowIdx > anchors[right].windowIdx
	})
	if len(anchors) > maxAnchors {
		anchors = anchors[:maxAnchors]
	}

	for _, anchor := range anchors {
		anchorTerms := distinctTerms[anchor.windowIdx]
		if len(anchorTerms) == 0 {
			continue
		}
		for later := anchor.windowIdx + 1; later < len(window); later++ {
			if !hasRevisionCue(turns[window[later]].Text) {
				continue
			}
			overlap := distinctiveOverlapScore(anchorTerms, distinctTerms[later], docFreq, len(window))
			if overlap <= 0 {
				continue
			}
			// Preserve superseding corrections with a strong continuity signal
			// bounded in (0, 1] so genuinely linked updates outrank unrelated
			// zero-relevance chatter without overriding higher-relevance anchors.
			boost := math.Min(1.0, math.Max(0.85, anchor.strength)*overlap)
			if boost > continuity[later] {
				continuity[later] = boost
			}
		}
	}
	return continuity
}

func distinctiveOverlapScore(
	anchorTerms map[string]struct{},
	candidateTerms map[string]struct{},
	docFreq map[string]int,
	docCount int,
) float64 {
	if len(anchorTerms) == 0 || len(candidateTerms) == 0 {
		return 0
	}
	shared := 0
	rareShared := 0
	for term := range anchorTerms {
		if _, ok := candidateTerms[term]; !ok {
			continue
		}
		shared++
		// Terms appearing in at most half the candidate window (or at most 2 turns)
		// indicate specific topical entities rather than boilerplate prose.
		if docFreq[term] <= max(2, docCount/2) {
			rareShared++
		}
	}
	if shared >= 2 && rareShared >= 1 {
		return 1.0
	}
	if shared == 1 && rareShared == 1 {
		// Single shared rare identifier (e.g. "postgresql", "ts_999") is enough
		// when it is genuinely specific (at most 2 turns in the window).
		for term := range anchorTerms {
			if _, ok := candidateTerms[term]; ok && docFreq[term] <= 2 && len([]rune(term)) >= 4 {
				return 0.9
			}
		}
	}
	return 0
}

func hasReferentialOptionCue(query string) bool {
	lower := strings.ToLower(query)
	for _, phrase := range []string{
		"first option", "second option", "third option",
		"option 1", "option 2", "option 3",
		"option a", "option b", "option c",
		"first one", "second one", "third one",
		"that option", "this option", "the option",
		"that plan", "this plan", "the former", "the latter",
		"continue with", "proceed with", "go with", "pick the second", "pick the first",
		"第一个方案", "第二个方案", "方案一", "方案二", "继续那个", "按第二个", "按第一个",
	} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func introducesMultipleOptions(text string) bool {
	lower := strings.ToLower(text)
	pairs := [][2]string{
		{"option 1", "option 2"},
		{"option a", "option b"},
		{"first option", "second option"},
		{"approach 1", "approach 2"},
		{"plan a", "plan b"},
		{"方案一", "方案二"},
		{"方案1", "方案2"},
		{"选项一", "选项二"},
	}
	for _, pair := range pairs {
		if strings.Contains(lower, pair[0]) && strings.Contains(lower, pair[1]) {
			return true
		}
	}
	return strings.Contains(lower, "two options") ||
		strings.Contains(lower, "two approaches") ||
		strings.Contains(lower, "两个方案") ||
		(strings.Contains(lower, "1.") && strings.Contains(lower, "2."))
}

func hasRevisionCue(text string) bool {
	lower := strings.ToLower(text)
	for _, phrase := range []string{
		"do not use", "don't use", "cannot use", "can't use", "won't use",
		"no longer", "ruled out", "rule out", "instead of", "rather than",
		"switch to", "switched to", "change to", "changed to",
		"不要用", "改为", "换成", "更正", "纠正", "排除", "取消", "废弃", "实际是", "其实",
	} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	for _, term := range tokenizeTerms(lower) {
		if _, ok := revisionCueTerms[term]; ok {
			return true
		}
	}
	return false
}

var revisionCueTerms = map[string]struct{}{
	"correction": {}, "correct": {}, "corrected": {}, "correcting": {},
	"actually": {}, "instead": {}, "reject": {}, "rejected": {}, "rejecting": {},
	"deprecated": {}, "deprecate": {}, "replace": {}, "replaced": {}, "replacing": {},
	"revise": {}, "revised": {}, "revision": {}, "cancel": {}, "canceled": {}, "cancelled": {},
	"override": {}, "overridden": {}, "amend": {}, "amended": {}, "supersede": {}, "superseded": {},
	"update": {}, "updated": {}, "upgraded": {}, "upgrade": {}, "scratch": {}, "blocked": {},
	"drop": {}, "dropped": {}, "discard": {}, "discarded": {}, "avoid": {},
}

// grammaticalStopWords excludes pure function words that otherwise distort IDF
// in short multi-turn conversations (e.g., matching "and" in a single unrelated
// turn could outrank a domain term appearing in multiple relevant turns).
var grammaticalStopWords = map[string]struct{}{
	"am": {}, "an": {}, "and": {}, "any": {}, "are": {}, "as": {}, "at": {},
	"be": {}, "been": {}, "being": {}, "but": {}, "by": {}, "can": {}, "could": {},
	"did": {}, "do": {}, "does": {}, "each": {}, "for": {}, "from": {}, "get": {}, "got": {},
	"had": {}, "has": {}, "have": {}, "he": {}, "her": {}, "here": {}, "him": {}, "his": {}, "how": {},
	"if": {}, "in": {}, "into": {}, "is": {}, "it": {}, "its": {}, "just": {}, "let": {}, "lets": {},
	"may": {}, "me": {}, "might": {}, "more": {}, "most": {}, "must": {}, "my": {},
	"no": {}, "not": {}, "now": {}, "of": {}, "on": {}, "only": {}, "or": {}, "other": {}, "our": {}, "out": {}, "over": {},
	"please": {}, "shall": {}, "she": {}, "should": {}, "so": {}, "some": {}, "such": {},
	"than": {}, "thank": {}, "thanks": {}, "that": {}, "the": {}, "their": {}, "them": {}, "then": {},
	"there": {}, "these": {}, "they": {}, "this": {}, "those": {}, "to": {},
	"under": {}, "up": {}, "us": {}, "very": {}, "was": {}, "we": {}, "were": {},
	"what": {}, "when": {}, "where": {}, "which": {}, "while": {}, "who": {}, "whom": {}, "why": {},
	"will": {}, "with": {}, "would": {}, "you": {}, "your": {}, "also": {}, "about": {}, "all": {},
}

// dialogueGenericWords are excluded from anchor-to-revision topical overlap
// (while remaining searchable by BM25 for explicit queries like "second option").
var dialogueGenericWords = map[string]struct{}{
	"use": {}, "using": {}, "need": {}, "want": {}, "make": {}, "keep": {},
	"take": {}, "give": {}, "list": {}, "next": {}, "steps": {}, "step": {},
	"continue": {}, "proceed": {}, "first": {}, "second": {}, "third": {},
	"one": {}, "two": {}, "three": {}, "plan": {}, "run": {}, "runs": {},
}

func filterBM25QueryTerms(terms []string) []string {
	if len(terms) == 0 {
		return nil
	}
	filtered := make([]string, 0, len(terms))
	for _, term := range terms {
		if _, stop := grammaticalStopWords[term]; !stop {
			filtered = append(filtered, term)
		}
	}
	// Preserve original terms when the entire query consists solely of stop words.
	if len(filtered) == 0 {
		return terms
	}
	return filtered
}

func distinctiveTermSet(text string) map[string]struct{} {
	terms := tokenizeTerms(text)
	result := make(map[string]struct{}, len(terms))
	for _, term := range terms {
		if _, stop := grammaticalStopWords[term]; stop {
			continue
		}
		if _, generic := dialogueGenericWords[term]; generic {
			continue
		}
		if _, cue := revisionCueTerms[term]; cue {
			continue
		}
		result[term] = struct{}{}
	}
	return result
}

func tokenizeTerms(text string) []string {
	lower := []rune(strings.ToLower(text))
	terms := make([]string, 0, len(lower)/4)
	word := make([]rune, 0, 16)
	flush := func() {
		if len(word) > 1 {
			terms = append(terms, string(word))
		}
		word = word[:0]
	}
	var prevCJK rune
	for _, r := range lower {
		switch {
		case isCJK(r):
			flush()
			terms = append(terms, string(r))
			if prevCJK != 0 {
				terms = append(terms, string([]rune{prevCJK, r}))
			}
			prevCJK = r
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
			prevCJK = 0
			word = append(word, r)
		default:
			flush()
			prevCJK = 0
		}
	}
	flush()
	return terms
}

func isCJK(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}
