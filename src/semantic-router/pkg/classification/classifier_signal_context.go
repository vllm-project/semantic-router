package classification

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// cachedJailbreakResult stores a cached jailbreak classification result.
type cachedJailbreakResult struct {
	result candle_binding.ClassResult
	err    error
}

// cachedPIIResult stores a cached PII token classification result.
type cachedPIIResult struct {
	result candle_binding.TokenClassificationResult
	err    error
}

// EvaluateAllSignalsWithContext evaluates all signal types with separate text for context counting.
//
// text: (possibly compressed) text for signal evaluation
// contextText: text for context token counting (usually all messages combined)
// nonUserMessages: conversation history for jailbreak/PII with include_history
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
// uncompressedText: original text before prompt compression (empty = no compression happened)
// skipCompressionSignals: signal types that must use uncompressedText instead of text
// imageURL: optional image URL for multimodal signals
// signalReadiness returns a map indicating whether each signal type's infrastructure is ready.
// Separated from EvaluateAllSignalsWithContext to keep cyclomatic complexity under the linter limit.
func (c *Classifier) signalReadiness() map[string]bool {
	return map[string]bool{
		config.SignalTypeKeyword:      c.keywordClassifier != nil,
		config.SignalTypeEmbedding:    c.keywordEmbeddingClassifier != nil,
		config.SignalTypeDomain:       c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil,
		config.SignalTypeFactCheck:    len(c.Config.FactCheckRules) > 0 && c.IsFactCheckEnabled(),
		config.SignalTypeUserFeedback: len(c.Config.UserFeedbackRules) > 0 && c.IsFeedbackDetectorEnabled(),
		config.SignalTypePreference:   len(c.Config.PreferenceRules) > 0 && c.IsPreferenceClassifierEnabled(),
		config.SignalTypeLanguage:     len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled(),
		config.SignalTypeContext:      c.contextClassifier != nil,
		config.SignalTypeStructure:    c.structureClassifier != nil,
		config.SignalTypeComplexity:   c.complexityClassifier != nil,
		config.SignalTypeModality:     len(c.Config.ModalityRules) > 0 && c.Config.ModalityDetector.Enabled,
		config.SignalTypeJailbreak:    len(c.Config.JailbreakRules) > 0 && c.IsJailbreakEnabled(),
		config.SignalTypePII:          len(c.Config.PIIRules) > 0 && c.IsPIIEnabled(),
	}
}

// textForSignalFunc returns a function that resolves the correct text for a given signal type,
// using uncompressed text for signals that must not receive compressed input.
func textForSignalFunc(text, uncompressedText string, skipCompressionSignals map[string]bool) func(string) string {
	return func(signalType string) string {
		if uncompressedText != "" && skipCompressionSignals[signalType] {
			return uncompressedText
		}
		return text
	}
}

func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, nonUserMessages []string, forceEvaluateAll bool, uncompressedText string, skipCompressionSignals map[string]bool, imageURL ...string) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()
	// Determine which signals (type:name) should be evaluated
	var usedSignals map[string]bool
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
		logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
	} else {
		usedSignals = c.getUsedSignals()
	}

	textForSignal := textForSignalFunc(text, uncompressedText, skipCompressionSignals)
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	imgArg := ""
	if len(imageURL) > 0 {
		imgArg = imageURL[0]
	}

	dispatchers := c.buildSignalDispatchers(results, &mu, textForSignal, contextText, nonUserMessages, imgArg)
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)

	wg.Wait()
	results = c.applySignalGroups(results)
	results = c.applySignalComposers(results)
	results = c.applyProjections(results)
	return results
}

func (c *Classifier) evaluateKeywordSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	category, keywords, err := c.keywordClassifier.ClassifyWithKeywords(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypeKeyword, category, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Keyword.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Keyword.Confidence = 1.0 // Rule-based, always 1.0

	logging.Debugf("[Signal Computation] Keyword signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("keyword rule evaluation failed: %v", err)
	} else if category != "" {
		// Record signal match
		metrics.RecordSignalMatch(config.SignalTypeKeyword, category)

		mu.Lock()
		results.MatchedKeywordRules = append(results.MatchedKeywordRules, category)
		results.MatchedKeywords = append(results.MatchedKeywords, keywords...)
		mu.Unlock()
	}
}

func (c *Classifier) evaluateEmbeddingSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	matchedRules, err := c.keywordEmbeddingClassifier.ClassifyAll(text)
	elapsed := time.Since(start)

	// Record metrics
	results.Metrics.Embedding.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Embedding signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("embedding rule evaluation failed: %v", err)
	} else if len(matchedRules) > 0 {
		// Record the highest confidence for metrics display
		var bestConfidence float64
		for _, mr := range matchedRules {
			if mr.Score > bestConfidence {
				bestConfidence = mr.Score
			}
		}
		results.Metrics.Embedding.Confidence = bestConfidence

		mu.Lock()
		// Add the configured top-k matched rules.
		for _, mr := range matchedRules {
			// Record signal extraction and match metrics for each matched rule
			metrics.RecordSignalExtraction(config.SignalTypeEmbedding, mr.RuleName, elapsed.Seconds())
			metrics.RecordSignalMatch(config.SignalTypeEmbedding, mr.RuleName)

			// Append rule name to the matched list
			results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, mr.RuleName)

			results.SignalConfidences["embedding:"+mr.RuleName] = mr.Score

			logging.Debugf("[Signal Computation] Embedding match: rule=%q, score=%.4f, method=%s",
				mr.RuleName, mr.Score, mr.Method)
		}
		mu.Unlock()
	}
}

func (c *Classifier) evaluateDomainSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	domainResult, err := c.categoryInference.ClassifyWithProbabilities(text)
	if err != nil {
		// Fall back to Classify() (top-1 only) when ClassifyWithProbabilities is unavailable.
		logging.Debugf("[Signal Computation] ClassifyWithProbabilities unavailable, falling back to Classify: %v", err)
		basicResult, basicErr := c.categoryInference.Classify(text)
		if basicErr != nil {
			err = basicErr
		} else {
			domainResult = candle_binding.ClassResultWithProbs{
				Class:      basicResult.Class,
				Confidence: basicResult.Confidence,
			}
			err = nil
		}
	}
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	categoryName := ""
	if err == nil {
		if name, ok := c.CategoryMapping.GetCategoryFromIndex(domainResult.Class); ok {
			categoryName = c.translateMMLUToGeneric(name)
		}
	}

	metrics.RecordSignalExtraction(config.SignalTypeDomain, categoryName, latencySeconds)

	// Record metrics
	results.Metrics.Domain.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if categoryName != "" && err == nil {
		results.Metrics.Domain.Confidence = float64(domainResult.Confidence)
	}
	logging.Debugf("[Signal Computation] Domain signal evaluation completed in %v", elapsed)

	if err != nil {
		logging.Errorf("domain rule evaluation failed: %v", err)
	} else {
		matched := c.matchDomainCategories(domainResult, categoryName)
		mu.Lock()
		for _, cat := range matched {
			metrics.RecordSignalMatch(config.SignalTypeDomain, cat.Category)
			results.MatchedDomainRules = append(results.MatchedDomainRules, cat.Category)
			results.SignalConfidences["domain:"+cat.Category] = float64(cat.Probability)
		}
		mu.Unlock()
	}
}

func (c *Classifier) evaluateFactCheckSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	factCheckResult, err := c.ClassifyFactCheck(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Determine which signal to output based on classification result
	signalName := "no_fact_check_needed"
	if err == nil && factCheckResult != nil && factCheckResult.NeedsFactCheck {
		signalName = "needs_fact_check"
	}

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypeFactCheck, signalName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.FactCheck.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if signalName != "" && err == nil && factCheckResult != nil {
		results.Metrics.FactCheck.Confidence = float64(factCheckResult.Confidence)
	}

	logging.Debugf("[Signal Computation] Fact-check signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("fact-check rule evaluation failed: %v", err)
	} else if factCheckResult != nil {
		// Check if this signal is defined in fact_check_rules
		for _, rule := range c.Config.FactCheckRules {
			if rule.Name == signalName {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypeFactCheck, rule.Name)

				mu.Lock()
				results.MatchedFactCheckRules = append(results.MatchedFactCheckRules, rule.Name)
				mu.Unlock()
				break
			}
		}
	}
}

func (c *Classifier) evaluateUserFeedbackSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	feedbackResult, err := c.ClassifyFeedback(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the feedback type directly as the signal name
	signalName := ""
	if err == nil && feedbackResult != nil {
		signalName = feedbackResult.FeedbackType
	}

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypeUserFeedback, signalName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.UserFeedback.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if signalName != "" && err == nil && feedbackResult != nil {
		results.Metrics.UserFeedback.Confidence = float64(feedbackResult.Confidence)
	}

	logging.Debugf("[Signal Computation] User feedback signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("user feedback rule evaluation failed: %v", err)
	} else if feedbackResult != nil {
		// Check if this signal is defined in user_feedback_rules
		for _, rule := range c.Config.UserFeedbackRules {
			if rule.Name == signalName {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypeUserFeedback, rule.Name)

				mu.Lock()
				results.MatchedUserFeedbackRules = append(results.MatchedUserFeedbackRules, rule.Name)
				mu.Unlock()
				break
			}
		}
	}
}

func (c *Classifier) evaluatePreferenceSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	contentBytes, _ := json.Marshal(text)
	conversationJSON := fmt.Sprintf(`[{"role":"user","content":%s}]`, contentBytes)

	preferenceResult, err := c.preferenceClassifier.Classify(conversationJSON)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the preference name directly as the signal name
	preferenceName := ""
	if err == nil && preferenceResult != nil {
		preferenceName = preferenceResult.Preference
	}

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypePreference, preferenceName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Preference.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if preferenceName != "" && err == nil && preferenceResult != nil && preferenceResult.Confidence > 0 {
		results.Metrics.Preference.Confidence = float64(preferenceResult.Confidence)
	}

	logging.Debugf("[Signal Computation] Preference signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("preference rule evaluation failed: %v", err)
	} else if preferenceResult != nil {
		// Check if this preference is defined in preference_rules
		for _, rule := range c.Config.PreferenceRules {
			if rule.Name == preferenceName {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypePreference, rule.Name)

				mu.Lock()
				results.MatchedPreferenceRules = append(results.MatchedPreferenceRules, rule.Name)
				mu.Unlock()
				logging.Debugf("Preference rule matched: %s", rule.Name)
				break
			}
		}
	}
}

func (c *Classifier) evaluateLanguageSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	languageResult, err := c.languageClassifier.Classify(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the language code directly as the signal name
	languageCode := ""
	if err == nil && languageResult != nil {
		languageCode = languageResult.LanguageCode
	}

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypeLanguage, languageCode, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Language.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if languageCode != "" && err == nil && languageResult != nil {
		results.Metrics.Language.Confidence = languageResult.Confidence
	}

	logging.Debugf("[Signal Computation] Language signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("language rule evaluation failed: %v", err)
	} else if languageResult != nil {
		// Check if this language code is defined in language_rules
		for _, rule := range c.Config.LanguageRules {
			if rule.Name == languageCode {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypeLanguage, rule.Name)

				mu.Lock()
				results.MatchedLanguageRules = append(results.MatchedLanguageRules, rule.Name)
				mu.Unlock()
				break
			}
		}
	}
}

func (c *Classifier) evaluateContextSignal(results *SignalResults, mu *sync.Mutex, contextText string) {
	start := time.Now()
	matchedRules, count, err := c.contextClassifier.Classify(contextText)
	elapsed := time.Since(start)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Context.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Context.Confidence = 1.0 // Rule-based, always 1.0

	logging.Debugf("[Signal Computation] Context signal evaluation completed in %v (count=%d)", elapsed, count)
	if err != nil {
		logging.Errorf("context rule evaluation failed: %v", err)
	} else {
		mu.Lock()
		results.MatchedContextRules = matchedRules
		results.TokenCount = count
		mu.Unlock()
	}
}

func (c *Classifier) evaluateComplexitySignal(results *SignalResults, mu *sync.Mutex, text string, imageURL string) {
	start := time.Now()
	matchedRules, err := c.complexityClassifier.ClassifyWithImage(text, imageURL)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Record signal extraction metrics for each matched rule
	for _, ruleName := range matchedRules {
		metrics.RecordSignalExtraction(config.SignalTypeComplexity, ruleName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeComplexity, ruleName)
	}

	// Record metrics (use microseconds for better precision)
	results.Metrics.Complexity.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Complexity.Confidence = 1.0 // Rule-based, always 1.0

	logging.Debugf("[Signal Computation] Complexity signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("complexity rule evaluation failed: %v", err)
	} else {
		mu.Lock()
		results.MatchedComplexityRules = matchedRules
		mu.Unlock()
	}
}

func (c *Classifier) evaluateModalitySignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	modalityResult := c.classifyModality(text, &c.Config.ModalityDetector.ModalityDetectionConfig)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	signalName := modalityResult.Modality

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypeModality, signalName, latencySeconds)

	// Record metrics
	results.Metrics.Modality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Modality.Confidence = float64(modalityResult.Confidence)

	logging.Debugf("[Signal Computation] Modality signal evaluation completed in %v: %s (confidence=%.3f, method=%s)",
		elapsed, signalName, modalityResult.Confidence, modalityResult.Method)

	// Check if this signal name is defined in modality_rules
	for _, rule := range c.Config.ModalityRules {
		if strings.EqualFold(rule.Name, signalName) {
			metrics.RecordSignalMatch(config.SignalTypeModality, rule.Name)
			mu.Lock()
			results.MatchedModalityRules = append(results.MatchedModalityRules, rule.Name)
			mu.Unlock()
			break
		}
	}
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

func (c *Classifier) evaluateJailbreakSignal(results *SignalResults, mu *sync.Mutex, jailbreakText string, nonUserMessages []string) {
	start := time.Now()

	// Step 1: Collect unique content pieces needed by classifier (non-contrastive) rules.
	classifierContents := c.collectJailbreakClassifierContents(jailbreakText, nonUserMessages)

	// Step 2: Run classifier inference exactly once per unique content piece.
	jailbreakCache := make(map[string]cachedJailbreakResult, len(classifierContents))
	for _, content := range classifierContents {
		result, err := c.jailbreakInference.Classify(content)
		jailbreakCache[content] = cachedJailbreakResult{result, err}
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

	metrics.RecordSignalExtraction(config.SignalTypeJailbreak, "jailbreak_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] Jailbreak signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluateJailbreakRule(rule config.JailbreakRule, jailbreakText string, nonUserMessages []string, jailbreakCache map[string]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	contentToAnalyze := buildContentList(jailbreakText, nonUserMessages, rule.IncludeHistory)
	if len(contentToAnalyze) == 0 {
		return
	}

	switch rule.Method {
	case "contrastive":
		c.evaluateContrastiveJailbreakRule(rule, contentToAnalyze, start, results, mu)
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

	metrics.RecordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	metrics.RecordSignalMatch(config.SignalTypeJailbreak, rule.Name)

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

func (c *Classifier) evaluateBERTJailbreakRule(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string]cachedJailbreakResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	bestType, bestConf := c.findBestJailbreakMatch(rule, contentToAnalyze, jailbreakCache)
	if bestConf <= 0 {
		return
	}

	metrics.RecordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
	metrics.RecordSignalMatch(config.SignalTypeJailbreak, rule.Name)

	mu.Lock()
	results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
	if bestConf > results.JailbreakConfidence {
		results.JailbreakDetected = true
		results.JailbreakType = bestType
		results.JailbreakConfidence = bestConf
	}
	results.SignalConfidences["jailbreak:"+rule.Name] = float64(bestConf)
	mu.Unlock()
}

// findBestJailbreakMatch scans cached BERT results and returns the highest-confidence jailbreak match.
func (c *Classifier) findBestJailbreakMatch(rule config.JailbreakRule, contentToAnalyze []string, jailbreakCache map[string]cachedJailbreakResult) (string, float32) {
	var bestType string
	var bestConf float32
	for _, content := range contentToAnalyze {
		if content == "" {
			continue
		}
		cached, ok := jailbreakCache[content]
		if !ok {
			continue
		}
		if cached.err != nil {
			logging.Errorf("[Signal Computation] Jailbreak rule %q: inference error: %v", rule.Name, cached.err)
			continue
		}
		jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(cached.result.Class)
		if !ok {
			logging.Errorf("[Signal Computation] Jailbreak rule %q: unknown class index %d", rule.Name, cached.result.Class)
			continue
		}
		if cached.result.Confidence < rule.Threshold || jailbreakType != "jailbreak" {
			continue
		}
		if cached.result.Confidence > bestConf {
			bestConf = cached.result.Confidence
			bestType = jailbreakType
		}
	}
	return bestType, bestConf
}

func (c *Classifier) evaluatePIISignal(results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string) {
	start := time.Now()

	// Step 1: Collect the union of unique content pieces across all PII rules.
	contentSeen := make(map[string]struct{})
	var uniqueContents []string
	if piiText != "" {
		contentSeen[piiText] = struct{}{}
		uniqueContents = append(uniqueContents, piiText)
	}
	for _, rule := range c.Config.PIIRules {
		if !rule.IncludeHistory {
			continue
		}
		for _, msg := range nonUserMessages {
			if msg == "" {
				continue
			}
			if _, ok := contentSeen[msg]; !ok {
				contentSeen[msg] = struct{}{}
				uniqueContents = append(uniqueContents, msg)
			}
		}
	}

	// Step 2: Run PII token classification exactly once per unique content piece.
	// Entity types are returned as "LABEL_{class_id}" and translated by PIIMapping.
	piiCache := make(map[string]cachedPIIResult, len(uniqueContents))
	for _, content := range uniqueContents {
		tokenResult, err := c.piiInference.ClassifyTokens(content)
		piiCache[content] = cachedPIIResult{tokenResult, err}
	}

	// Step 3: Evaluate each rule concurrently using the cached token results.
	// Each goroutine applies its own threshold and allow-list without re-running the model.
	var ruleWg sync.WaitGroup
	for _, rule := range c.Config.PIIRules {
		ruleWg.Add(1)
		go func() {
			defer ruleWg.Done()
			c.evaluatePIIRule(rule, piiText, nonUserMessages, piiCache, start, results, mu)
		}()
	}
	ruleWg.Wait()

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.PII.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if results.PIIDetected {
		results.Metrics.PII.Confidence = 1.0 // Binary: PII found or not
	}

	metrics.RecordSignalExtraction(config.SignalTypePII, "pii_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] PII signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, piiCache map[string]cachedPIIResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	ruleContents := collectPIIRuleContents(piiText, nonUserMessages, rule.IncludeHistory)
	if len(ruleContents) == 0 {
		return
	}

	entityTypes := c.collectPIIEntityTypes(ruleContents, rule.Name, rule.Threshold, piiCache)
	deniedEntities := findDeniedEntities(entityTypes, rule.PIITypesAllowed)

	if len(deniedEntities) > 0 {
		metrics.RecordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
		metrics.RecordSignalMatch(config.SignalTypePII, rule.Name)

		logging.Debugf("[Signal Computation] PII rule %q matched: denied_entities=%v", rule.Name, deniedEntities)

		mu.Lock()
		results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
		results.PIIDetected = true
		for _, e := range deniedEntities {
			if !slices.Contains(results.PIIEntities, e) {
				results.PIIEntities = append(results.PIIEntities, e)
			}
		}
		mu.Unlock()
	}
}
