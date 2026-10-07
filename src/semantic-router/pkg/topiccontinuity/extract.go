package topiccontinuity

import (
	"context"
	"math"
	"strings"
)

const recencyDecay = 0.85

// extract computes every threshold-independent feature of one prepared group,
// so rules sharing a history policy never repeat text work.
func extract(ctx context.Context, prepared preparation) extraction {
	if prepared.Status != "" {
		return extraction{
			Policy: prepared.Policy, Coverage: prepared.Coverage, Scope: prepared.Scope,
			Status: prepared.Status,
		}
	}
	cancelled := func(out extraction, capped bool) extraction {
		// Cancellation keeps the scope observed so far.
		out.Scope.FeatureCapReached = out.Scope.FeatureCapReached || capped
		out.Features = Features{}
		out.Coverage = CoveragePartial
		out.Status = ReasonCancelled
		return out
	}
	out := extraction{Policy: prepared.Policy, Coverage: prepared.Coverage, Scope: prepared.Scope}
	probe := newCancellationProbe(ctx)
	if probe.cancelled() {
		return cancelled(out, false)
	}

	liveTerms := termSet(prepared.Live.User)
	if probe.cancelled() {
		return cancelled(out, false)
	}
	liveEntities, capped := entitySet(prepared.Live.User, nil)
	if probe.cancelled() {
		return cancelled(out, capped)
	}
	features := Features{
		PriorTurnsExamined: len(prepared.Prior),
		InputBytes:         prepared.InputBytes,
		LiveTerms:          len(liveTerms),
		LiveProseTerms:     proseTermCount(prepared.Live.User),
	}
	bestDecayed := -1.0
	for k, turn := range prepared.Prior {
		if probe.cancelled() {
			return cancelled(out, capped)
		}
		segments := append(append([]textSegment{}, turn.User...), turn.Assistant...)
		turnEntities, turnCapped := entitySet(segments, turn.ToolNames)
		capped = capped || turnCapped
		if probe.cancelled() {
			return cancelled(out, capped)
		}
		lexical := containment(liveTerms, termSet(segments))
		raw := lexical
		entity := 0.0
		if len(liveEntities) > 0 {
			entity = containment(liveEntities, turnEntities)
			raw = 0.5*lexical + 0.5*entity
		}
		decayed := raw * math.Pow(recencyDecay, float64(k))
		if decayed > bestDecayed {
			bestDecayed = decayed
			features.CombinedScore = decayed
			features.LexicalScore = lexical
			features.EntityScore = entity
		}
		features.MaxRawScore = math.Max(features.MaxRawScore, raw)
		features.MaxEntityScore = math.Max(features.MaxEntityScore, entity)
	}
	if probe.cancelled() {
		return cancelled(out, capped)
	}
	if capped {
		out.Scope.FeatureCapReached = true
		out.Coverage = CoveragePartial
	}
	if !applyPhraseFlags(probe, &features, prepared.Live.User) {
		return cancelled(out, capped)
	}
	out.Features = features
	return out
}

func entitySet(segments []textSegment, toolNames []string) (map[string]struct{}, bool) {
	out := make(map[string]struct{})
	capped := false
	for _, segment := range segments {
		entities, segmentCapped := segmentEntities(segment)
		capped = capped || segmentCapped
		for entity := range entities {
			out[entity] = struct{}{}
		}
	}
	for _, name := range toolNames {
		out[strings.ToLower(name)] = struct{}{}
	}
	return out, capped
}

// proseTermCount counts distinct non-stopword terms in the unmasked runs of
// the live segments, so a bare paste of code or quoted text never counts as a
// self-contained new request.
func proseTermCount(segments []textSegment) int {
	var prose []textSegment
	for _, segment := range segments {
		text := string(segment)
		for _, run := range unmaskedRuns(len(text), maskedRanges(text)) {
			prose = append(prose, textSegment(text[run.Start:run.End]))
		}
	}
	return len(termSet(prose))
}

// containment is |live ∩ other| / |live|, or 0 when live is empty.
func containment(live, other map[string]struct{}) float64 {
	if len(live) == 0 {
		return 0
	}
	shared := 0
	for term := range live {
		if _, ok := other[term]; ok {
			shared++
		}
	}
	return float64(shared) / float64(len(live))
}

// applyPhraseFlags sets the reference, acknowledgement, and change-marker
// flags. Continuation flags use the unstripped view; change markers use the
// masked view, so stripping can only ever hide change evidence. It probes
// between passes and returns false on cancellation; the caller then discards
// the flags.
func applyPhraseFlags(probe cancellationProbe, features *Features, live []textSegment) bool {
	if len(live) == 0 {
		return true
	}
	continuation := make([][]phraseToken, len(live))
	for i, segment := range live {
		continuation[i] = continuationView(segment)
		if probe.cancelled() {
			return false
		}
	}
	for _, tokens := range continuation {
		for index := range tokens {
			if matchAny(tokens, index, strongReferencePhrases) > 0 {
				features.StrongReference = true
			}
		}
	}
	if probe.cancelled() {
		return false
	}
	first := continuation[0]
	lead := leadingAcknowledgements(first)
	for index := lead; index < len(first) && index < lead+weakReferenceWindow; index++ {
		if _, ok := weakReferenceTokens[first[index].Text]; ok {
			features.WeakReference = true
		}
	}
	features.Acknowledgement = isAcknowledgement(continuation)
	if probe.cancelled() {
		return false
	}
	clean, ambiguous, ok := changeMarkers(probe, live)
	features.MarkerAmbiguous = ambiguous
	features.ChangeMarker = clean && !ambiguous
	return ok
}

// isAcknowledgement requires every live token to be covered, without gaps, by
// acknowledgement phrases, with at most maxAckTokens tokens in total.
func isAcknowledgement(segments [][]phraseToken) bool {
	total := 0
	for _, tokens := range segments {
		if leadingAcknowledgements(tokens) != len(tokens) {
			return false
		}
		total += len(tokens)
	}
	return total > 0 && total <= maxAckTokens
}

// changeMarkers scans the masked change view of every live segment. A clean
// occurrence is leading (first segment, within the leading window after
// acknowledgements), not negated, and not inside paired single quotes. Any
// other occurrence is ambiguous.
func changeMarkers(probe cancellationProbe, live []textSegment) (clean, ambiguous, ok bool) {
	for segmentIndex, segment := range live {
		tokens := changeView(segment)
		if probe.cancelled() {
			return false, false, false
		}
		quotes := singleQuoteSpans(string(segment))
		if probe.cancelled() {
			return false, false, false
		}
		lead := leadingAcknowledgements(tokens)
		for index := range tokens {
			if matchAny(tokens, index, changeMarkerPhrases) == 0 {
				continue
			}
			leading := segmentIndex == 0 && index >= lead && index-lead < changeLeadingWindow
			if leading && !negated(tokens, index) && !insideAny(quotes, tokens[index].Offset) {
				clean = true
			} else {
				ambiguous = true
			}
		}
		if probe.cancelled() {
			return false, false, false
		}
	}
	return clean, ambiguous, true
}

// negated looks back across run barriers, which can only add ambiguity.
func negated(tokens []phraseToken, index int) bool {
	for back := 1; back <= negationLookback && index-back >= 0; back++ {
		if _, ok := negationTokens[tokens[index-back].Text]; ok {
			return true
		}
	}
	return false
}

func insideAny(spans []byteRange, offset int) bool {
	for _, span := range spans {
		if span.contains(offset) {
			return true
		}
	}
	return false
}
