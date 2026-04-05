package classification

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var ErrPreferenceBelowThreshold = errors.New("preference below threshold")

type preferenceEmbeddingTask struct {
	ruleName string
	text     string
}

type preferenceEmbeddingResult struct {
	ruleName  string
	embedding []float32
	err       error
}

// ContrastivePreferenceClassifier performs few-shot preference routing using embeddings.
// It preloads embeddings for each preference rule's examples/description and selects
// the route whose support set is most similar to the incoming query.
type ContrastivePreferenceClassifier struct {
	modelType string

	rules []config.PreferenceRule

	// ruleEmbeddings maps rule name to its support embeddings
	ruleEmbeddings map[string][][]float32
	ruleBanks      map[string]*prototypeBank
	// ruleThresholds stores per-preference similarity thresholds
	ruleThresholds map[string]float32
	prototypeCfg   config.PrototypeScoringConfig

	mu sync.RWMutex
}

type PreferenceRuleScore struct {
	Name           string
	Score          float32
	Best           float32
	Support        float32
	Threshold      float32
	PrototypeCount int
}

type PreferenceClassificationDetails struct {
	Scores       []PreferenceRuleScore
	BestRule     string
	BestScore    float32
	RunnerUpRule string
	RunnerUp     float32
	Margin       float32
}

// NewContrastivePreferenceClassifier builds a contrastive preference classifier.
// modelType follows GetEmbeddingWithModelType (e.g. "qwen3", "gemma", "mmbert").
func NewContrastivePreferenceClassifier(rules []config.PreferenceRule, modelType string) (*ContrastivePreferenceClassifier, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("contrastive preference rules cannot be empty")
	}

	if modelType == "" {
		modelType = "mmbert"
	}

	ruleThresholds := make(map[string]float32, len(rules))
	for _, rule := range rules {
		ruleThresholds[rule.Name] = rule.Threshold
	}

	c := &ContrastivePreferenceClassifier{
		modelType:      modelType,
		rules:          rules,
		ruleEmbeddings: make(map[string][][]float32),
		ruleBanks:      make(map[string]*prototypeBank),
		ruleThresholds: ruleThresholds,
		prototypeCfg:   config.PrototypeScoringConfig{}.WithDefaults(),
	}

	if err := c.preloadRuleEmbeddings(); err != nil {
		return nil, err
	}

	return c, nil
}

func NewContrastivePreferenceClassifierWithConfig(
	rules []config.PreferenceRule,
	modelType string,
	prototypeCfg config.PrototypeScoringConfig,
) (*ContrastivePreferenceClassifier, error) {
	classifier, err := NewContrastivePreferenceClassifier(rules, modelType)
	if err != nil {
		return nil, err
	}
	classifier.prototypeCfg = prototypeCfg.WithDefaults()
	classifier.rebuildRuleBanks()
	return classifier, nil
}

// preloadRuleEmbeddings computes embeddings for all rule examples concurrently.
func (c *ContrastivePreferenceClassifier) preloadRuleEmbeddings() error {
	start := time.Now()
	tasks, err := c.collectEmbeddingTasks()
	if err != nil {
		return err
	}

	resultCh := c.embedRuleExamples(tasks)
	loaded, firstErr := c.collectEmbeddedResults(resultCh)

	logging.Infof("[Preference Contrastive] preloaded %d/%d example embeddings using model=%s in %v", loaded, len(tasks), c.modelType, time.Since(start))

	if firstErr != nil {
		return firstErr
	}

	c.rebuildRuleBanks()

	return nil
}

func (c *ContrastivePreferenceClassifier) collectEmbeddingTasks() ([]preferenceEmbeddingTask, error) {
	tasks := make([]preferenceEmbeddingTask, 0)
	for _, rule := range c.rules {
		for _, example := range c.collectExamples(rule) {
			if strings.TrimSpace(example) == "" {
				continue
			}
			tasks = append(tasks, preferenceEmbeddingTask{ruleName: rule.Name, text: example})
		}
	}

	if len(tasks) == 0 {
		return nil, fmt.Errorf("no examples provided for contrastive preference classifier")
	}
	return tasks, nil
}

func (c *ContrastivePreferenceClassifier) embedRuleExamples(
	tasks []preferenceEmbeddingTask,
) <-chan preferenceEmbeddingResult {
	taskCh := make(chan preferenceEmbeddingTask, len(tasks))
	resultCh := make(chan preferenceEmbeddingResult, len(tasks))

	for _, task := range tasks {
		taskCh <- task
	}
	close(taskCh)

	var wg sync.WaitGroup
	for i := 0; i < c.embeddingWorkerCount(len(tasks)); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskCh {
				out, err := getEmbeddingWithModelType(task.text, c.modelType, 0)
				if err != nil {
					resultCh <- preferenceEmbeddingResult{ruleName: task.ruleName, err: err}
					continue
				}
				resultCh <- preferenceEmbeddingResult{
					ruleName:  task.ruleName,
					embedding: out.Embedding,
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()
	return resultCh
}

func (c *ContrastivePreferenceClassifier) collectEmbeddedResults(
	resultCh <-chan preferenceEmbeddingResult,
) (int, error) {
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

	return loaded, firstErr
}

func (c *ContrastivePreferenceClassifier) embeddingWorkerCount(taskCount int) int {
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > taskCount {
		return taskCount
	}
	return numWorkers
}

// Classify picks the preference with the highest similarity to the query.
func (c *ContrastivePreferenceClassifier) Classify(text string) (*PreferenceResult, error) {
	details, err := c.ClassifyDetailed(text)
	if err != nil {
		return nil, err
	}
	if details.BestRule == "" {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}
	threshold := c.ruleThresholds[details.BestRule]
	if threshold > 0 && details.BestScore < threshold {
		return nil, fmt.Errorf(
			"%w: preference similarity %.3f below threshold %.3f",
			ErrPreferenceBelowThreshold,
			details.BestScore,
			threshold,
		)
	}
	if c.prototypeCfg.WithDefaults().MarginThreshold > 0 && details.Margin < c.prototypeCfg.WithDefaults().MarginThreshold {
		return nil, fmt.Errorf(
			"%w: preference margin %.3f below threshold %.3f",
			ErrPreferenceBelowThreshold,
			details.Margin,
			c.prototypeCfg.WithDefaults().MarginThreshold,
		)
	}
	return &PreferenceResult{
		Preference: details.BestRule,
		Confidence: details.BestScore,
		Margin:     details.Margin,
	}, nil
}

func (c *ContrastivePreferenceClassifier) ClassifyDetailed(text string) (*PreferenceClassificationDetails, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}

	c.mu.RLock()
	if len(c.ruleBanks) == 0 {
		c.mu.RUnlock()
		return nil, fmt.Errorf("no embeddings loaded for contrastive preference classifier")
	}
	c.mu.RUnlock()

	out, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	queryEmbedding := out.Embedding

	c.mu.RLock()
	defer c.mu.RUnlock()

	scores := make([]PreferenceRuleScore, 0, len(c.ruleBanks))
	for _, rule := range c.rules {
		bank, ok := c.ruleBanks[rule.Name]
		if !ok || bank == nil || len(bank.prototypes) == 0 {
			continue
		}
		bankScore := bank.score(queryEmbedding, defaultPrototypeScoreOptions(c.prototypeCfg))
		score := PreferenceRuleScore{
			Name:           rule.Name,
			Score:          float32(bankScore.Score),
			Best:           float32(bankScore.Best),
			Support:        float32(bankScore.Support),
			Threshold:      c.ruleThresholds[rule.Name],
			PrototypeCount: bankScore.PrototypeCount,
		}
		logging.Debugf("[Preference Contrastive] rule=%s score=%.4f best=%.4f support=%.4f prototypes=%d",
			rule.Name, score.Score, score.Best, score.Support, score.PrototypeCount)
		scores = append(scores, score)
	}

	if len(scores) == 0 {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}

	sort.Slice(scores, func(i, j int) bool {
		if scores[i].Score == scores[j].Score {
			return scores[i].Name < scores[j].Name
		}
		return scores[i].Score > scores[j].Score
	})

	details := &PreferenceClassificationDetails{
		Scores:    scores,
		BestRule:  scores[0].Name,
		BestScore: scores[0].Score,
	}
	if len(scores) > 1 {
		details.RunnerUpRule = scores[1].Name
		details.RunnerUp = scores[1].Score
		details.Margin = scores[0].Score - scores[1].Score
	}

	return details, nil
}

func (c *ContrastivePreferenceClassifier) collectExamples(rule config.PreferenceRule) []string {
	examples := make([]string, 0, 1+len(rule.Examples))

	if rule.Description != "" {
		examples = append(examples, rule.Description)
	}

	if len(rule.Examples) > 0 {
		examples = append(examples, rule.Examples...)
	}

	return examples
}

func (c *ContrastivePreferenceClassifier) rebuildRuleBanks() {
	c.ruleBanks = make(map[string]*prototypeBank, len(c.rules))
	for _, rule := range c.rules {
		embeddings := c.ruleEmbeddings[rule.Name]
		examples := c.collectExamples(rule)
		prototypeExamples := make([]prototypeExample, 0, len(embeddings))
		for i, embedding := range embeddings {
			if len(embedding) == 0 {
				continue
			}
			text := rule.Name
			if i < len(examples) {
				text = examples[i]
			}
			prototypeExamples = append(prototypeExamples, prototypeExample{
				Key:       fmt.Sprintf("%s:%d", rule.Name, i),
				Text:      text,
				Embedding: embedding,
			})
		}
		bank := newPrototypeBank(prototypeExamples, c.prototypeCfg)
		c.ruleBanks[rule.Name] = bank
		logPrototypeBankSummary("Preference Contrastive", rule.Name, bank)
	}
}
