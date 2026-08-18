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
			if cached.err != nil {
				logging.Errorf("[Signal Computation] Jailbreak rule %q: inference error: %v", rule.Name, cached.err)
				continue
			}
			class, _ := deriveArgmax(cached.result.Probabilities)
			jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(class)
			if !ok {
				logging.Errorf("[Signal Computation] Jailbreak rule %q: unknown class index %d", rule.Name, class)
				continue
			}
			aboveThreshold, riskScore := isJailbreakRiskAboveThreshold(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, cached.result, rule.Threshold)
			if !aboveThreshold {
				continue
			}
			if riskScore > bestScore {
				bestScore = riskScore
				bestType = jailbreakType
			}
		}
	}
	return bestType, bestScore
}
