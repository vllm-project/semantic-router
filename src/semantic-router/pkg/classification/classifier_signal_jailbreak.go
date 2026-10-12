package classification

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// cachedJailbreakResult stores a cached jailbreak classification result.
type cachedJailbreakResult struct {
	result   SequenceClassificationResult
	decision *tasks.LabelDecision
	err      error
}

// JailbreakClassificationErrorType is the sentinel jailbreak type reported
// when on_error: block forces a rule to match because inference itself
// failed (e.g. an unreachable http_chat/http_classify endpoint), rather than
// because the content was actually classified as a jailbreak. It is
// distinguishable from a real detected type in results/logs. Without this
// fail-closed path (on_error: allow, the default), a classify error is
// indistinguishable from a genuinely safe request - see @adaamko's review on
// #2760. LoadJailbreakMapping rejects any configured jailbreak_mapping label
// that resolves to this value, in any of the supported label_to_idx/
// label_to_id/idx_to_label/id_to_label shapes, so a real detection can never
// collide with it.
const JailbreakClassificationErrorType = "classification_error"

// JailbreakUnscannedType is the sentinel jailbreak type reported when a rule
// matches because the model did not read all of the content: it was longer
// than the model's input or than the scan budget of a model that reads it in
// windows, it was truncated, or its scan missed the signal's deadline. A guard
// does not pass content it did not read, so this match holds whatever
// on_error says (on_error governs backend failures) unless on_unscanned is
// allow. Like JailbreakClassificationErrorType it is a policy match, not model
// evidence, and no configured label may resolve to it.
const JailbreakUnscannedType = "unscanned"

const jailbreakEvaluationFailedCode = "jailbreak_evaluation_failed"

func (c *Classifier) collectJailbreakClassifierContentPieces(current, history []string) []string {
	seen := make(map[string]struct{})
	var contents []string
	addUnique := func(s string) {
		if s == "" {
			return
		}
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			contents = append(contents, s)
		}
	}
	for _, rule := range c.Config.RequestJailbreakRules() {
		if rule.Method == "contrastive" {
			continue
		}
		for _, text := range current {
			addUnique(text)
		}
		if !rule.IncludeHistory {
			continue
		}
		for _, msg := range history {
			addUnique(msg)
		}
	}
	return contents
}

func (c *Classifier) evaluateJailbreakSignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, jailbreakText string, nonUserMessages []string) {
	c.evaluateJailbreakSignalPieces(ctx, results, mu, []string{jailbreakText}, nonUserMessages)
}

func (c *Classifier) evaluateJailbreakSignalPieces(ctx context.Context, results *SignalResults, mu *sync.Mutex, current, history []string) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()

	// Step 1: Collect unique content pieces needed by classifier (non-contrastive) rules.
	classifierContents := c.collectJailbreakClassifierContentPieces(current, history)

	// Step 2: Run classifier inference exactly once per unique content piece,
	// every piece concurrently so the pieces share one bundle.
	jailbreakCache := make(map[string][]cachedJailbreakResult, len(classifierContents))
	type piece struct{ content, chunk string }
	var pieces []piece
	for _, content := range classifierContents {
		chunks := c.jailbreakModelInputs(content)
		for _, chunk := range chunks {
			pieces = append(pieces, piece{content, chunk})
		}
	}
	classified := make([][]cachedJailbreakResult, len(pieces))
	modelservice.Fan(ctx, len(pieces), func(i int) {
		classified[i] = c.classifyJailbreakWindows(ctx, pieces[i].chunk)
	})
	for i, p := range pieces {
		jailbreakCache[p.content] = append(jailbreakCache[p.content], classified[i]...)
	}

	// Step 3: Evaluate all rules concurrently.
	rules := c.Config.RequestJailbreakRules()
	modelservice.Fan(ctx, len(rules), func(i int) {
		c.evaluateJailbreakRulePieces(rules[i], current, history, jailbreakCache, start, results, mu)
	})
	c.recordJailbreakObservedRisk(results, mu)

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.Jailbreak.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	available := results.JailbreakScoreAvailable
	results.Metrics.Jailbreak.ConfidenceAvailable = &available
	if available {
		results.Metrics.Jailbreak.Confidence = float64(results.JailbreakConfidence)
	}

	c.recordSignalExtraction(config.SignalTypeJailbreak, "jailbreak_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] Jailbreak signal evaluation completed in %v", elapsed)
}

// Publish valid observed risk independently of whether any rule matched. A
// partial scan may contribute a real score; its SignalErrors remain unresolved.
// Contrastive scores and categorical decisions do not supply probability values.
func (c *Classifier) recordJailbreakObservedRisk(results *SignalResults, mu *sync.Mutex) {
	mu.Lock()
	defer mu.Unlock()
	for _, rule := range c.Config.RequestJailbreakRules() {
		if rule.Method == "contrastive" {
			continue
		}
		risk, available := results.SignalValues[signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)]
		if available && (!results.JailbreakScoreAvailable || float32(risk) > results.JailbreakConfidence) {
			results.JailbreakConfidence = float32(risk)
			results.JailbreakScoreAvailable = true
		}
	}
}

func (c *Classifier) evaluateJailbreakRulePieces(rule config.JailbreakRule, current, history []string, jailbreakCache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	var contentToAnalyze []string
	for _, text := range current {
		if text != "" {
			contentToAnalyze = append(contentToAnalyze, text)
		}
	}
	if rule.IncludeHistory {
		contentToAnalyze = append(contentToAnalyze, history...)
	}
	if len(contentToAnalyze) == 0 {
		return
	}

	switch rule.Method {
	case "contrastive":
		var chunks []string
		for _, content := range contentToAnalyze {
			chunks = append(chunks, c.jailbreakInputs(content)...)
		}
		c.evaluateContrastiveJailbreakRule(rule, chunks, start, results, mu)
	default:
		c.evaluateBERTJailbreakRule(rule, contentToAnalyze, jailbreakCache, start, results, mu)
	}
}

func (c *Classifier) evaluateContrastiveJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, start time.Time, results *SignalResults, mu *sync.Mutex) {
	cjc, ok := c.contrastiveJailbreakClassifiers[rule.Name]
	if !ok {
		logging.Errorf("[Signal Computation] Contrastive jailbreak classifier not found for rule %q", rule.Name)
		return
	}
	analysisResult := cjc.AnalyzeMessages(contentToAnalyze)
	if analysisResult.FailedMessages > 0 {
		c.recordJailbreakRuleError(rule, results, mu, jailbreakEvaluationFailedCode)
	}
	threshold := rule.Threshold
	if threshold <= 0 {
		threshold = 0.10
	}
	if analysisResult.MaxScore < threshold {
		// Nothing scored above the threshold. If some message could not be
		// embedded at all it was never checked, so under on_error: block the
		// content is unverified rather than clean - the same policy the BERT
		// path applies to a classify error.
		if analysisResult.FailedMessages > 0 && c.Config.PromptGuard.IsBlock() {
			logging.Errorf("[Signal Computation] Contrastive jailbreak rule %q: %d/%d messages could not be embedded; failing closed",
				rule.Name, analysisResult.FailedMessages, analysisResult.TotalMessages)
			c.recordJailbreakRuleMatch(rule, JailbreakClassificationErrorType, 0, start, results, mu)
		}
		return
	}

	c.recordJailbreakRuleMatch(rule, "contrastive", analysisResult.MaxScore, start, results, mu)

	logging.Debugf("[Signal Computation] Contrastive jailbreak rule %q matched: score=%.4f threshold=%.4f worst_msg_idx=%d time=%v",
		rule.Name, analysisResult.MaxScore, threshold, analysisResult.WorstMsgIndex, analysisResult.ProcessingTime)
}

// recordJailbreakRuleMatch records one matched jailbreak rule into results.
func (c *Classifier) recordJailbreakRuleMatch(rule config.JailbreakRule, jailbreakType string, confidence float32, start time.Time, results *SignalResults, mu *sync.Mutex) {
	c.recordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	c.recordSignalMatch(config.SignalTypeJailbreak, rule.Name)

	mu.Lock()
	defer mu.Unlock()
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	if jailbreakType == JailbreakClassificationErrorType || jailbreakType == JailbreakUnscannedType {
		if results.SignalErrorMatches == nil {
			results.SignalErrorMatches = make(map[string]bool)
		}
		results.SignalErrorMatches[signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)] = true
		if !results.JailbreakDetected {
			results.JailbreakDetected = true
			results.JailbreakType = jailbreakType
		}
		// This is a policy match on unverified content, not model evidence.
		return
	}
	if !results.JailbreakDetected || !results.JailbreakScoreAvailable || confidence > results.JailbreakConfidence {
		results.JailbreakType = jailbreakType
		results.JailbreakConfidence = confidence
		results.JailbreakScoreAvailable = true
	}
	results.JailbreakDetected = true
	results.SignalConfidences["jailbreak:"+rule.Name] = float64(confidence)
}

func (c *Classifier) recordJailbreakRuleError(rule config.JailbreakRule, results *SignalResults, mu *sync.Mutex, code string) {
	mu.Lock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	results.SignalErrors[signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)] = code
	mu.Unlock()
}

func (c *Classifier) evaluateBERTJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	if jailbreakDecisionBackend(c.jailbreakInference) != nil {
		c.evaluateCategoricalJailbreakRule(rule, contentToAnalyze, jailbreakCache, start, results, mu)
		return
	}
	observed := c.findJailbreakRuleObservation(rule, contentToAnalyze, jailbreakCache)
	if observed.riskAvailable {
		mu.Lock()
		if results.SignalValues == nil {
			results.SignalValues = make(map[string]float64)
		}
		key := signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)
		results.SignalValues[key] = float64(observed.riskScore)
		// A scan reads a long prompt in several windows and reports the riskiest
		// one, so the value above belongs to a span rather than to the whole
		// prompt. Without these a tail window blocking a benign document reads
		// the same as the document itself scoring high.
		if observed.window != nil {
			results.SignalValues[key+":window_start"] = float64(observed.window.Start)
			results.SignalValues[key+":window_end"] = float64(observed.window.End)
			results.SignalValues[key+":windows"] = float64(observed.window.Count)
		}
		mu.Unlock()
	}
	if observed.unresolved {
		c.recordJailbreakRuleError(rule, results, mu, observed.errorCode)
	}
	if observed.matchedType != "" {
		c.recordJailbreakRuleMatch(rule, observed.matchedType, observed.matchedScore, start, results, mu)
	}
}

type jailbreakCandidateOutcome int

const (
	jailbreakCandidateNone jailbreakCandidateOutcome = iota
	jailbreakCandidateMatched
	jailbreakCandidateUnknown
)

// jailbreakCandidate is one cached result's contribution to
// findBestJailbreakMatch's scan.
type jailbreakCandidate struct {
	outcome       jailbreakCandidateOutcome
	errorCode     string
	jailbreakType string
	riskScore     float32
	riskAvailable bool
	window        *tasks.ScanWindow
	// unscanned: the model did not read all of this piece (its length).
	unscanned bool
}

// evaluateCachedJailbreakResult classifies a single cached result into a
// jailbreakCandidate. Split out of findBestJailbreakMatch to keep its
// cognitive complexity within the repo's lint gate (the same reason
// assignScoreToMapping is split out of alignScoresToMapping).
func (c *Classifier) evaluateCachedJailbreakResult(rule config.JailbreakRule, cached cachedJailbreakResult) jailbreakCandidate {
	if cached.err == nil {
		cached.err = validateJailbreakDistribution(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, cached.result)
	}
	if cached.err != nil {
		logging.Errorf("[Signal Computation] Jailbreak rule %q: inference error: %v", rule.Name, cached.err)
		return jailbreakCandidate{outcome: jailbreakCandidateUnknown, errorCode: boundedSignalErrorCode(cached.err, jailbreakEvaluationFailedCode), unscanned: UnscannedInput(cached.err)}
	}
	class, _ := deriveArgmax(cached.result.Probabilities)
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(class)
	if !ok {
		// The call succeeded but its answer is uninterpretable against the
		// configured mapping (e.g. a 3-class checkpoint behind a 2-class
		// mapping - nothing validates the model head against the mapping,
		// since the initializers ignore numClasses and the FFI reports the
		// model's own head size). That is "could not verify safe" just like a
		// transport error, so on_error: block must close here too rather than
		// treat it as a clean result.
		logging.Errorf("[Signal Computation] Jailbreak rule %q: unknown class index %d", rule.Name, class)
		return jailbreakCandidate{outcome: jailbreakCandidateUnknown, errorCode: jailbreakEvaluationFailedCode}
	}
	aboveThreshold, riskScore := isJailbreakRiskAboveThreshold(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, cached.result, rule.Threshold)
	candidate := jailbreakCandidate{jailbreakType: jailbreakType, riskScore: riskScore, riskAvailable: true, window: cached.result.Window}
	if aboveThreshold {
		candidate.outcome = jailbreakCandidateMatched
	}
	return candidate
}

// findBestJailbreakMatch scans cached BERT results and returns the highest
// combined-positive-label-risk match, via the same isJailbreakRiskAboveThreshold
// helper CheckForJailbreakWithRisk uses.
func (c *Classifier) findBestJailbreakMatch(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult) (string, float32) {
	bestType, bestScore, _ := c.findBestJailbreakMatchOutcome(rule, contentToAnalyze, jailbreakCache)
	return bestType, bestScore
}

func (c *Classifier) findBestJailbreakMatchOutcome(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult) (string, float32, bool) {
	observed := c.findJailbreakRuleObservation(rule, contentToAnalyze, jailbreakCache)
	return observed.matchedType, observed.matchedScore, observed.unresolved
}

// Match evidence and value availability are independent: a valid zero or a
// value below this rule's threshold is still an observed positive-label risk.
type jailbreakRuleObservation struct {
	matchedType   string
	matchedScore  float32
	riskScore     float32
	riskAvailable bool
	unresolved    bool
	errorCode     string
	window        *tasks.ScanWindow
}

func (c *Classifier) findJailbreakRuleObservation(rule config.JailbreakRule, contents []string, cache map[string][]cachedJailbreakResult) jailbreakRuleObservation {
	observed := jailbreakRuleObservation{}
	unscanned := false
	for _, content := range contents {
		if content == "" {
			continue
		}
		for _, cached := range cache[content] {
			candidate := c.evaluateCachedJailbreakResult(rule, cached)
			if candidate.riskAvailable && (!observed.riskAvailable || candidate.riskScore > observed.riskScore) {
				observed.riskScore = candidate.riskScore
				observed.riskAvailable = true
				observed.window = candidate.window
			}
			unscanned = unscanned || candidate.unscanned
			switch candidate.outcome {
			case jailbreakCandidateUnknown:
				observed.unresolved = true
				observed.errorCode = mergeSignalErrorCode(observed.errorCode, candidate.errorCode)
			case jailbreakCandidateMatched:
				if candidate.riskScore > observed.matchedScore {
					observed.matchedScore = candidate.riskScore
					observed.matchedType = candidate.jailbreakType
				}
			}
		}
	}
	// A policy match has no fabricated probability; real observed pieces retain
	// their own risk alongside the error. Content the model did not read
	// matches whatever on_error says; a backend failure follows on_error.
	if observed.matchedType == "" && unscanned && c.Config.PromptGuard.UnscannedBlocks() {
		observed.matchedType = JailbreakUnscannedType
	}
	if observed.matchedType == "" && observed.unresolved && c.Config.PromptGuard.IsBlock() {
		observed.matchedType = JailbreakClassificationErrorType
	}
	return observed
}

func (c *Classifier) evaluateCategoricalJailbreakRule(rule config.JailbreakRule, contents []string, cache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	var matched *tasks.LabelDecision
	unresolved, unscanned := false, false
	errorCode := ""
	for _, content := range contents {
		for _, entry := range cache[content] {
			if entry.err != nil || entry.decision == nil {
				unresolved = true
				unscanned = unscanned || UnscannedInput(entry.err)
				errorCode = mergeSignalErrorCode(errorCode, boundedSignalErrorCode(entry.err, jailbreakEvaluationFailedCode))
				continue
			}
			if _, ok := c.JailbreakMapping.GetIndexForJailbreakType(entry.decision.Label); !ok {
				unresolved = true
				errorCode = mergeSignalErrorCode(errorCode, jailbreakEvaluationFailedCode)
				continue
			}
			if isPositiveJailbreakLabel(c.Config.PromptGuard.PositiveLabels, entry.decision.Label) {
				matched = entry.decision
			}
		}
	}
	if unresolved {
		c.recordJailbreakRuleError(rule, results, mu, errorCode)
	}
	if matched == nil {
		switch {
		case unscanned && c.Config.PromptGuard.UnscannedBlocks():
			c.recordJailbreakRuleMatch(rule, JailbreakUnscannedType, 0, start, results, mu)
		case unresolved && c.Config.PromptGuard.IsBlock():
			c.recordJailbreakRuleMatch(rule, JailbreakClassificationErrorType, 0, start, results, mu)
		}
		return
	}
	c.recordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	c.recordSignalMatch(config.SignalTypeJailbreak, rule.Name)
	mu.Lock()
	defer mu.Unlock()
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	results.JailbreakDetected = true
	results.JailbreakType = matched.Label
	results.JailbreakDecision = matched
	// A categorical match has no entry in the probability map.
}
