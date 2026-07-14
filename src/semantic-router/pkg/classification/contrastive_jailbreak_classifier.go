package classification

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ContrastiveJailbreakResult holds the analysis outcome for a single rule.
type ContrastiveJailbreakResult struct {
	MaxScore       float32 // Highest contrastive score across analysed messages
	WorstMessage   string  // The message that produced MaxScore
	WorstMsgIndex  int     // Index of that message in the input slice
	JailbreakSim   float32 // max_sim(worstMsg, jailbreak_kb) for the worst message
	BenignSim      float32 // max_sim(worstMsg, benign_kb) for the worst message
	TotalMessages  int     // Number of messages analysed
	ProcessingTime time.Duration
}

// ContrastiveJailbreakClassifier implements contrastive embedding similarity
// for jailbreak detection. It mirrors the ComplexityClassifier pattern:
// pre-computed KB embeddings at init, fast cosine scoring at request time.
//
// Score = max_sim(msg, jailbreak_kb) − max_sim(msg, benign_kb)
// When include_history is true, the maximum score across all user messages
// in the conversation is used (multi-turn chain detection).
type ContrastiveJailbreakClassifier struct {
	rule config.JailbreakRule

	// Pre-computed embeddings for the two knowledge bases
	jailbreakEmbeddings map[string][]float32 // pattern text → embedding
	benignEmbeddings    map[string][]float32 // pattern text → embedding

	modelType string
	backend   string
	provider  embedding.Provider
}

type contrastiveJailbreakEmbeddingTask struct {
	text        string
	isJailbreak bool
}

type contrastiveJailbreakEmbeddingResult struct {
	text        string
	embedding   []float32
	isJailbreak bool
	err         error
}

// NewContrastiveJailbreakClassifier creates and initialises a classifier for a
// single contrastive JailbreakRule. KB embeddings are computed eagerly using a
// worker pool (same approach as ComplexityClassifier).
func NewContrastiveJailbreakClassifier(rule config.JailbreakRule, defaultModelType string) (*ContrastiveJailbreakClassifier, error) {
	return NewContrastiveJailbreakClassifierWithProvider(rule, defaultModelType, nil)
}

func NewContrastiveJailbreakClassifierWithProvider(rule config.JailbreakRule, defaultModelType string, provider embedding.Provider) (*ContrastiveJailbreakClassifier, error) {
	return newContrastiveJailbreakClassifierWithBackend(rule, defaultModelType, "", provider)
}

func newContrastiveJailbreakClassifierWithBackend(rule config.JailbreakRule, defaultModelType string, backend string, provider embedding.Provider) (*ContrastiveJailbreakClassifier, error) {
	modelType := defaultModelType
	if modelType == "" {
		modelType = "qwen3"
	}

	c := &ContrastiveJailbreakClassifier{
		rule:                rule,
		jailbreakEmbeddings: make(map[string][]float32),
		benignEmbeddings:    make(map[string][]float32),
		modelType:           modelType,
		backend:             normalizeTextEmbeddingBackend(backend),
		provider:            provider,
	}

	if err := c.preloadKBEmbeddings(); err != nil {
		return nil, fmt.Errorf("contrastive jailbreak rule %q: %w", rule.Name, err)
	}
	return c, nil
}

// AnalyzeMessages computes the contrastive score for each message and returns
// the result with the maximum score (multi-turn chain detection).
// If messages is empty the returned MaxScore is -1.
func (c *ContrastiveJailbreakClassifier) AnalyzeMessages(messages []string) ContrastiveJailbreakResult {
	result, _ := c.AnalyzeMessagesWithError(messages)
	return result
}

// AnalyzeMessagesWithError preserves inference failures so security routing
// cannot silently interpret a broken embedding backend as a benign no-match.
func (c *ContrastiveJailbreakClassifier) AnalyzeMessagesWithError(messages []string) (ContrastiveJailbreakResult, error) {
	start := time.Now()
	result := ContrastiveJailbreakResult{
		MaxScore:      -1,
		WorstMsgIndex: -1,
		TotalMessages: len(messages),
	}

	for i, msg := range messages {
		if msg == "" {
			continue
		}
		msgEmb, err := c.embedText(msg)
		if err != nil {
			return result, fmt.Errorf("contrastive jailbreak embedding failed: %w", err)
		}

		// max similarity to jailbreak KB
		maxJailSim := float32(-1)
		for _, emb := range c.jailbreakEmbeddings {
			if sim := cosineSimilarity(msgEmb, emb); sim > maxJailSim {
				maxJailSim = sim
			}
		}

		// max similarity to benign KB
		maxBenignSim := float32(-1)
		for _, emb := range c.benignEmbeddings {
			if sim := cosineSimilarity(msgEmb, emb); sim > maxBenignSim {
				maxBenignSim = sim
			}
		}

		score := maxJailSim - maxBenignSim
		if score > result.MaxScore {
			result.MaxScore = score
			result.WorstMessage = msg
			result.WorstMsgIndex = i
			result.JailbreakSim = maxJailSim
			result.BenignSim = maxBenignSim
		}
	}

	result.ProcessingTime = time.Since(start)
	return result, nil
}

func (c *ContrastiveJailbreakClassifier) embedText(text string) ([]float32, error) {
	embedding, _, err := executeTextEmbedding(context.Background(), c.backend, c.provider, text, c.modelType, 0)
	return embedding, err
}

// preloadKBEmbeddings concurrently computes embeddings for jailbreak and benign
// pattern knowledge bases, following the same worker-pool approach as
// ComplexityClassifier.preloadCandidateEmbeddings.
func (c *ContrastiveJailbreakClassifier) preloadKBEmbeddings() error {
	startTime := time.Now()

	logging.Infof("[Contrastive Jailbreak] Preloading KB embeddings for rule %q (jailbreak: %d, benign: %d, model: %s)",
		c.rule.Name, len(c.rule.JailbreakPatterns), len(c.rule.BenignPatterns), c.modelType)

	tasks := c.contrastiveJailbreakEmbeddingTasks()
	if len(tasks) == 0 {
		logging.Warnf("[Contrastive Jailbreak] Rule %q has no KB patterns", c.rule.Name)
		return nil
	}

	numWorkers := boundedContrastiveJailbreakWorkers(len(tasks))
	ok, firstErr := c.collectContrastiveJailbreakEmbeddings(c.embedContrastiveJailbreakTasks(tasks, numWorkers))

	elapsed := time.Since(startTime)
	logging.Infof("[Contrastive Jailbreak] Rule %q: preloaded %d/%d KB embeddings in %v (workers: %d)",
		c.rule.Name, ok, len(tasks), elapsed, numWorkers)

	return firstErr
}

func (c *ContrastiveJailbreakClassifier) contrastiveJailbreakEmbeddingTasks() []contrastiveJailbreakEmbeddingTask {
	tasks := make([]contrastiveJailbreakEmbeddingTask, 0, len(c.rule.JailbreakPatterns)+len(c.rule.BenignPatterns))
	for _, pattern := range c.rule.JailbreakPatterns {
		tasks = append(tasks, contrastiveJailbreakEmbeddingTask{text: pattern, isJailbreak: true})
	}
	for _, pattern := range c.rule.BenignPatterns {
		tasks = append(tasks, contrastiveJailbreakEmbeddingTask{text: pattern})
	}
	return tasks
}

func boundedContrastiveJailbreakWorkers(taskCount int) int {
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > taskCount {
		return taskCount
	}
	return numWorkers
}

func (c *ContrastiveJailbreakClassifier) embedContrastiveJailbreakTasks(
	tasks []contrastiveJailbreakEmbeddingTask,
	numWorkers int,
) <-chan contrastiveJailbreakEmbeddingResult {
	taskCh := make(chan contrastiveJailbreakEmbeddingTask, len(tasks))
	resultCh := make(chan contrastiveJailbreakEmbeddingResult, len(tasks))
	for _, task := range tasks {
		taskCh <- task
	}
	close(taskCh)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskCh {
				resultCh <- c.embedContrastiveJailbreakTask(task)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()
	return resultCh
}

func (c *ContrastiveJailbreakClassifier) embedContrastiveJailbreakTask(
	task contrastiveJailbreakEmbeddingTask,
) contrastiveJailbreakEmbeddingResult {
	embedding, err := c.embedText(task.text)
	return contrastiveJailbreakEmbeddingResult{
		text:        task.text,
		embedding:   embedding,
		isJailbreak: task.isJailbreak,
		err:         err,
	}
}

func (c *ContrastiveJailbreakClassifier) collectContrastiveJailbreakEmbeddings(
	results <-chan contrastiveJailbreakEmbeddingResult,
) (int, error) {
	var mu sync.Mutex
	var firstErr error
	ok := 0
	for result := range results {
		if result.err != nil {
			if firstErr == nil {
				firstErr = contrastiveJailbreakEmbeddingError(result)
			}
			continue
		}
		mu.Lock()
		c.storeContrastiveJailbreakEmbedding(result)
		mu.Unlock()
		ok++
	}
	return ok, firstErr
}

func contrastiveJailbreakEmbeddingError(result contrastiveJailbreakEmbeddingResult) error {
	kind := "benign"
	if result.isJailbreak {
		kind = "jailbreak"
	}
	return fmt.Errorf("failed to embed %s pattern %q: %w", kind, result.text, result.err)
}

func (c *ContrastiveJailbreakClassifier) storeContrastiveJailbreakEmbedding(result contrastiveJailbreakEmbeddingResult) {
	if result.isJailbreak {
		c.jailbreakEmbeddings[result.text] = result.embedding
		return
	}
	c.benignEmbeddings[result.text] = result.embedding
}
