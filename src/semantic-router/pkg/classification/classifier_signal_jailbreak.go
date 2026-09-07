package classification

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// cachedJailbreakResult stores a cached jailbreak classification result.
type cachedJailbreakResult struct {
	result SequenceClassificationResult
	err    error
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

const jailbreakEvaluationFailedCode = "jailbreak_evaluation_failed"

// collectJailbreakClassifierContents returns the deduplicated set of text pieces
// that need BERT classifier inference (contrastive rules are excluded).
func (c *Classifier) collectJailbreakClassifierContents(jailbreakText string, nonUserMessages []string) []string {
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
		addUnique(jailbreakText)
		if !rule.IncludeHistory {
			continue
		}
		for _, msg := range nonUserMessages {
			addUnique(msg)
		}
	}
	return contents
}

func (c *Classifier) evaluateJailbreakSignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, jailbreakText string, nonUserMessages []string) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()

	// Step 1: Collect unique content pieces needed by classifier (non-contrastive) rules.
	classifierContents := c.collectJailbreakClassifierContents(jailbreakText, nonUserMessages)

	// Step 2: Run classifier inference exactly once per unique content piece.
	jailbreakCache := make(map[string][]cachedJailbreakResult, len(classifierContents))
	for _, content := range classifierContents {
		chunks := jailbreakSignalChunks(content)
		cached := make([]cachedJailbreakResult, 0, len(chunks))
		for _, chunk := range chunks {
			result, err := c.jailbreakInference.Classify(ctx, chunk)
			cached = append(cached, cachedJailbreakResult{result, err})
		}
		jailbreakCache[content] = cached
	}

	// Step 3: Evaluate all rules concurrently.
	var ruleWg sync.WaitGroup
	for _, rule := range c.Config.RequestJailbreakRules() {
		ruleWg.Add(1)
		go func() {
			defer ruleWg.Done()
			c.evaluateJailbreakRule(rule, jailbreakText, nonUserMessages, jailbreakCache, start, results, mu)
		}()
	}
	ruleWg.Wait()

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.Jailbreak.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if results.JailbreakConfidence > 0 {
		results.Metrics.Jailbreak.Confidence = float64(results.JailbreakConfidence)
	}

	c.recordSignalExtraction(config.SignalTypeJailbreak, "jailbreak_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] Jailbreak signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluateJailbreakRule(rule config.JailbreakRule, jailbreakText string, nonUserMessages []string, jailbreakCache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	contentToAnalyze := buildContentList(jailbreakText, nonUserMessages, rule.IncludeHistory)
	if len(contentToAnalyze) == 0 {
		return
	}

	switch rule.Method {
	case "contrastive":
		var chunks []string
		for _, content := range contentToAnalyze {
			chunks = append(chunks, jailbreakSignalChunks(content)...)
		}
		c.evaluateContrastiveJailbreakRule(rule, chunks, start, results, mu)
	default:
		c.evaluateBERTJailbreakRule(rule, contentToAnalyze, jailbreakCache, start, results, mu)
	}
}

// buildContentList assembles the text pieces to analyze for a single rule.
func buildContentList(text string, nonUserMessages []string, includeHistory bool) []string {
	var content []string
	if text != "" {
		content = append(content, text)
	}
	if includeHistory && len(nonUserMessages) > 0 {
		content = append(content, nonUserMessages...)
	}
	return content
}

func (c *Classifier) evaluateContrastiveJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, start time.Time, results *SignalResults, mu *sync.Mutex) {
	cjc, ok := c.contrastiveJailbreakClassifiers[rule.Name]
	if !ok {
		logging.Errorf("[Signal Computation] Contrastive jailbreak classifier not found for rule %q", rule.Name)
		return
	}
	analysisResult := cjc.AnalyzeMessages(contentToAnalyze)
	if analysisResult.FailedMessages > 0 {
		c.recordJailbreakRuleError(rule, results, mu)
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
			c.recordJailbreakRuleMatch(rule, JailbreakClassificationErrorType, 1.0, start, results, mu)
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
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	if jailbreakType == JailbreakClassificationErrorType {
		if results.SignalErrorMatches == nil {
			results.SignalErrorMatches = make(map[string]bool)
		}
		results.SignalErrorMatches[signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)] = true
	}
	if confidence > results.JailbreakConfidence {
		results.JailbreakDetected = true
		results.JailbreakType = jailbreakType
		results.JailbreakConfidence = confidence
	}
	results.SignalConfidences["jailbreak:"+rule.Name] = float64(confidence)
	mu.Unlock()
}

func (c *Classifier) recordJailbreakRuleError(rule config.JailbreakRule, results *SignalResults, mu *sync.Mutex) {
	mu.Lock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	results.SignalErrors[signalConfidenceKey(config.SignalTypeJailbreak, rule.Name)] = jailbreakEvaluationFailedCode
	mu.Unlock()
}

func (c *Classifier) evaluateBERTJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	bestType, bestScore, unresolved := c.findBestJailbreakMatchOutcome(rule, contentToAnalyze, jailbreakCache)
	if unresolved {
		c.recordJailbreakRuleError(rule, results, mu)
	}
	if bestScore <= 0 {
		return
	}
	c.recordJailbreakRuleMatch(rule, bestType, bestScore, start, results, mu)
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
	jailbreakType string
	riskScore     float32
}

// evaluateCachedJailbreakResult classifies a single cached result into a
// jailbreakCandidate. Split out of findBestJailbreakMatch to keep its
// cognitive complexity within the repo's lint gate (the same reason
// assignScoreToMapping is split out of alignScoresToMapping).
func (c *Classifier) evaluateCachedJailbreakResult(rule config.JailbreakRule, cached cachedJailbreakResult) jailbreakCandidate {
	if cached.err != nil {
		logging.Errorf("[Signal Computation] Jailbreak rule %q: inference error: %v", rule.Name, cached.err)
		return jailbreakCandidate{outcome: jailbreakCandidateUnknown}
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
		return jailbreakCandidate{outcome: jailbreakCandidateUnknown}
	}
	aboveThreshold, riskScore := isJailbreakRiskAboveThreshold(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, cached.result, rule.Threshold)
	if !aboveThreshold {
		return jailbreakCandidate{}
	}
	return jailbreakCandidate{outcome: jailbreakCandidateMatched, jailbreakType: jailbreakType, riskScore: riskScore}
}

// findBestJailbreakMatch scans cached BERT results and returns the highest
// combined-positive-label-risk match, via the same isJailbreakRiskAboveThreshold
// helper CheckForJailbreakWithRisk uses.
func (c *Classifier) findBestJailbreakMatch(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult) (string, float32) {
	bestType, bestScore, _ := c.findBestJailbreakMatchOutcome(rule, contentToAnalyze, jailbreakCache)
	return bestType, bestScore
}

func (c *Classifier) findBestJailbreakMatchOutcome(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult) (string, float32, bool) {
	var bestType string
	var bestScore float32
	unresolved := false
	for _, content := range contentToAnalyze {
		if content == "" {
			continue
		}
		cachedResults, ok := jailbreakCache[content]
		if !ok {
			continue
		}
		for _, cached := range cachedResults {
			candidate := c.evaluateCachedJailbreakResult(rule, cached)
			switch candidate.outcome {
			case jailbreakCandidateUnknown:
				unresolved = true
			case jailbreakCandidateMatched:
				if candidate.riskScore > bestScore {
					bestScore = candidate.riskScore
					bestType = candidate.jailbreakType
				}
			}
		}
	}

	// A genuine detection wins over the fail-closed sentinel. Both block the
	// request - the sentinel's 1.0 is >= any real score, so every decision
	// that would fire still fires either way - but returning the sentinel
	// when a real match exists erases the detected label and its true score,
	// and those flow on to results.JailbreakType/JailbreakConfidence, the
	// replay record's jailbreak_type, and the jailbreak.type span attribute.
	// A replayed genuine attack would then be indistinguishable from an
	// unreachable guardrail, and eval confidence would skew to 1.0.
	if bestType != "" {
		return bestType, bestScore, unresolved
	}
	if unresolved && c.Config.PromptGuard.IsBlock() {
		return JailbreakClassificationErrorType, 1.0, true
	}
	return bestType, bestScore, unresolved
}
