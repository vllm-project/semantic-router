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

// jailbreakClassificationErrorType is the sentinel jailbreak type reported
// when config.PromptGuardOnErrorBlock forces a rule to match because
// inference itself failed (e.g. an unreachable http_chat/http_classify
// endpoint), rather than because the content was actually classified as a
// jailbreak. It is distinguishable from a real detected type in
// results/logs. Without this fail-closed path (config.PromptGuardOnErrorAllow,
// the default), a classify error is indistinguishable from a genuinely safe
// request - see @adaamko's review on #2760.
const jailbreakClassificationErrorType = "classification_error"

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
	for _, rule := range c.Config.JailbreakRules {
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
	for _, rule := range c.Config.JailbreakRules {
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
	threshold := rule.Threshold
	if threshold <= 0 {
		threshold = 0.10
	}
	if analysisResult.MaxScore < threshold {
		return
	}

	c.recordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	c.recordSignalMatch(config.SignalTypeJailbreak, rule.Name)

	confidence := analysisResult.MaxScore
	mu.Lock()
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	if confidence > results.JailbreakConfidence {
		results.JailbreakDetected = true
		results.JailbreakType = "contrastive"
		results.JailbreakConfidence = confidence
	}
	results.SignalConfidences["jailbreak:"+rule.Name] = float64(confidence)
	mu.Unlock()

	logging.Debugf("[Signal Computation] Contrastive jailbreak rule %q matched: score=%.4f threshold=%.4f worst_msg_idx=%d time=%v",
		rule.Name, analysisResult.MaxScore, threshold, analysisResult.WorstMsgIndex, analysisResult.ProcessingTime)
}

func (c *Classifier) evaluateBERTJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string][]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	bestType, bestScore := c.findBestJailbreakMatch(rule, contentToAnalyze, jailbreakCache)
	if bestScore <= 0 {
		return
	}

	c.recordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	c.recordSignalMatch(config.SignalTypeJailbreak, rule.Name)

	mu.Lock()
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	if bestScore > results.JailbreakConfidence {
		results.JailbreakDetected = true
		results.JailbreakType = bestType
		results.JailbreakConfidence = bestScore
	}
	results.SignalConfidences["jailbreak:"+rule.Name] = float64(bestScore)
	mu.Unlock()
}

// jailbreakCandidateOutcome is the disposition of one cached result within
// findBestJailbreakMatch's scan. A dedicated type (rather than two
// independent bools) makes the invalid combination - both "matched" and
// "fail-closed" true at once - unrepresentable.
type jailbreakCandidateOutcome int

const (
	// jailbreakCandidateNone means this cached result contributed nothing:
	// a classify error tolerated by on_error: skip, an unknown class index,
	// or a risk score below the rule's threshold. The zero value, so a
	// zero-value jailbreakCandidate is safely "no contribution".
	jailbreakCandidateNone jailbreakCandidateOutcome = iota
	// jailbreakCandidateMatched means jailbreakType/riskScore are a genuine
	// content-based match.
	jailbreakCandidateMatched
	// jailbreakCandidateFailClosed means on_error: fail turned a classify
	// error into an immediate, maximum-confidence match.
	jailbreakCandidateFailClosed
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
		if c.Config.PromptGuard.OnError == config.PromptGuardOnErrorBlock {
			return jailbreakCandidate{outcome: jailbreakCandidateFailClosed}
		}
		return jailbreakCandidate{}
	}
	class, _ := deriveArgmax(cached.result.Probabilities)
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(class)
	if !ok {
		logging.Errorf("[Signal Computation] Jailbreak rule %q: unknown class index %d", rule.Name, class)
		return jailbreakCandidate{}
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
	var bestType string
	var bestScore float32
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
			case jailbreakCandidateNone:
				// No contribution: a tolerated error, an unknown class
				// index, or a risk score below the rule's threshold.
			case jailbreakCandidateFailClosed:
				return jailbreakClassificationErrorType, 1.0
			case jailbreakCandidateMatched:
				if candidate.riskScore > bestScore {
					bestScore = candidate.riskScore
					bestType = candidate.jailbreakType
				}
			}
		}
	}
	return bestType, bestScore
}
