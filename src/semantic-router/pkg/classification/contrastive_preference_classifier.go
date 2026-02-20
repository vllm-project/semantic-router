package classification

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ContrastivePreferenceClassifier performs few-shot preference routing using embeddings.
// It preloads embeddings for each preference rule's examples/description and selects
// the route whose support set is most similar to the incoming query.
type ContrastivePreferenceClassifier struct {
	modelType   string
	maxExamples int

	rules []config.PreferenceRule

	// ruleEmbeddings maps rule name to its support embeddings
	ruleEmbeddings map[string][][]float32

	mu sync.RWMutex
}

// NewContrastivePreferenceClassifier builds a contrastive preference classifier.
// modelType follows GetEmbeddingWithModelType (e.g. "qwen3", "gemma", "mmbert").
func NewContrastivePreferenceClassifier(rules []config.PreferenceRule, modelType string, maxExamples int) (*ContrastivePreferenceClassifier, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("contrastive preference rules cannot be empty")
	}

	if modelType == "" {
		modelType = "qwen3"
	}

	c := &ContrastivePreferenceClassifier{
		modelType:      modelType,
		maxExamples:    maxExamples,
		rules:          rules,
		ruleEmbeddings: make(map[string][][]float32),
	}

	if err := c.preloadRuleEmbeddings(); err != nil {
		return nil, err
	}

	return c, nil
}

// preloadRuleEmbeddings computes embeddings for all rule examples concurrently.
func (c *ContrastivePreferenceClassifier) preloadRuleEmbeddings() error {
	start := time.Now()

	type task struct {
		ruleName string
		text     string
	}

	var tasks []task
	for _, rule := range c.rules {
		for _, example := range c.collectExamples(rule) {
			if strings.TrimSpace(example) == "" {
				continue
			}
			tasks = append(tasks, task{ruleName: rule.Name, text: example})
		}
	}

	if len(tasks) == 0 {
		return fmt.Errorf("no examples provided for contrastive preference classifier")
	}

	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(tasks) {
		numWorkers = len(tasks)
	}

	type result struct {
		ruleName  string
		embedding []float32
		err       error
	}

	taskCh := make(chan task, len(tasks))
	resultCh := make(chan result, len(tasks))

	for _, t := range tasks {
		taskCh <- t
	}
	close(taskCh)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range taskCh {
				out, err := getEmbeddingWithModelType(t.text, c.modelType, 0)
				if err != nil {
					resultCh <- result{ruleName: t.ruleName, err: err}
					continue
				}
				resultCh <- result{ruleName: t.ruleName, embedding: out.Embedding}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	loaded := 0
	var firstErr error

	c.mu.Lock()
	defer c.mu.Unlock()

	for res := range resultCh {
		if res.err != nil {
			if firstErr == nil {
				firstErr = res.err
			}
			logging.Warnf("[Preference Contrastive] failed to embed example for %s: %v", res.ruleName, res.err)
			continue
		}
		c.ruleEmbeddings[res.ruleName] = append(c.ruleEmbeddings[res.ruleName], res.embedding)
		loaded++
	}

	logging.Infof("[Preference Contrastive] preloaded %d/%d example embeddings using model=%s in %v", loaded, len(tasks), c.modelType, time.Since(start))

	if firstErr != nil {
		return firstErr
	}

	return nil
}

// Classify picks the preference with the highest similarity to the query.
func (c *ContrastivePreferenceClassifier) Classify(text string) (*PreferenceResult, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}

	c.mu.RLock()
	if len(c.ruleEmbeddings) == 0 {
		c.mu.RUnlock()
		return nil, fmt.Errorf("no embeddings loaded for contrastive preference classifier")
	}
	c.mu.RUnlock()

	out, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	queryEmbedding := out.Embedding

	var (
		bestRule  string
		bestScore float32 = -1
	)

	c.mu.RLock()
	defer c.mu.RUnlock()

	for ruleName, embeddings := range c.ruleEmbeddings {
		if len(embeddings) == 0 {
			continue
		}
		var maxSim float32
		for _, emb := range embeddings {
			sim := cosineSimilarity(queryEmbedding, emb)
			if sim > maxSim {
				maxSim = sim
			}
		}
		logging.Debugf("[Preference Contrastive] rule=%s similarity=%.4f", ruleName, maxSim)
		if maxSim > bestScore {
			bestScore = maxSim
			bestRule = ruleName
		}
	}

	if bestRule == "" {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}

	return &PreferenceResult{
		Preference: bestRule,
		Confidence: bestScore,
	}, nil
}

func (c *ContrastivePreferenceClassifier) collectExamples(rule config.PreferenceRule) []string {
	examples := make([]string, 0, 1+len(rule.Examples))

	if rule.Description != "" {
		examples = append(examples, rule.Description)
	}

	if len(rule.Examples) > 0 {
		limit := len(rule.Examples)
		if c.maxExamples > 0 && c.maxExamples < limit {
			limit = c.maxExamples
		}
		examples = append(examples, rule.Examples[:limit]...)
	}

	return examples
}
