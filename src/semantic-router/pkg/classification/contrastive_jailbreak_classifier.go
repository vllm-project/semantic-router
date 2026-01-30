package classification

import (
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ContrastiveJailbreakResult represents the result of contrastive jailbreak analysis
type ContrastiveJailbreakResult struct {
	// MaxScore is the maximum contrastive score across all messages
	MaxScore float32 `json:"max_score"`
	// IsJailbreak indicates if the max score exceeds threshold
	IsJailbreak bool `json:"is_jailbreak"`
	// FlaggedIndex is the index of the message that triggered detection (-1 if none)
	FlaggedIndex int `json:"flagged_index"`
	// ScoresPerTurn contains the contrastive score for each user message
	ScoresPerTurn []ContrastiveTurnScore `json:"scores_per_turn"`
}

// ContrastiveTurnScore represents the score for a single turn
type ContrastiveTurnScore struct {
	Content string  `json:"content"` // First 100 chars of content
	Score   float32 `json:"score"`
}

// ContrastiveJailbreakClassifier performs multi-turn jailbreak detection using
// contrastive embedding similarity against jailbreak and benign knowledge bases.
// This detects gradual escalation attacks that evade per-message classifiers.
type ContrastiveJailbreakClassifier struct {
	jailbreakPatterns []string
	benignPatterns    []string
	threshold         float32

	// Precomputed embeddings for KB patterns
	jailbreakEmbeddings [][]float32
	benignEmbeddings    [][]float32

	modelType string
	mu        sync.RWMutex
}

// NewContrastiveJailbreakClassifier creates a new contrastive jailbreak classifier
func NewContrastiveJailbreakClassifier(cfg *config.ContrastiveJailbreakConfig, modelType string) (*ContrastiveJailbreakClassifier, error) {
	if modelType == "" {
		modelType = "qwen3" // Default to qwen3
	}

	c := &ContrastiveJailbreakClassifier{
		jailbreakPatterns: cfg.JailbreakPatterns,
		benignPatterns:    cfg.BenignPatterns,
		threshold:         cfg.Threshold,
		modelType:         modelType,
	}

	logging.Infof("[Contrastive Jailbreak] Initializing classifier with model type: %s, threshold: %.3f",
		c.modelType, c.threshold)
	logging.Infof("[Contrastive Jailbreak] KB sizes: jailbreak=%d patterns, benign=%d patterns",
		len(c.jailbreakPatterns), len(c.benignPatterns))

	// Precompute KB embeddings at initialization
	if err := c.preloadKBEmbeddings(); err != nil {
		logging.Warnf("[Contrastive Jailbreak] Failed to preload KB embeddings: %v", err)
		return nil, err
	}

	return c, nil
}

// preloadKBEmbeddings computes embeddings for all KB patterns using concurrent processing
func (c *ContrastiveJailbreakClassifier) preloadKBEmbeddings() error {
	startTime := time.Now()

	logging.Infof("[Contrastive Jailbreak] Preloading embeddings for KB patterns using model: %s...", c.modelType)

	// Collect all patterns to process
	type embeddingTask struct {
		pattern    string
		isJailbreak bool
	}

	var tasks []embeddingTask
	for _, pattern := range c.jailbreakPatterns {
		tasks = append(tasks, embeddingTask{pattern: pattern, isJailbreak: true})
	}
	for _, pattern := range c.benignPatterns {
		tasks = append(tasks, embeddingTask{pattern: pattern, isJailbreak: false})
	}

	if len(tasks) == 0 {
		logging.Infof("[Contrastive Jailbreak] No KB patterns to preload")
		return nil
	}

	// Use worker pool for concurrent embedding generation
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(tasks) {
		numWorkers = len(tasks)
	}

	type result struct {
		embedding   []float32
		isJailbreak bool
		err         error
	}

	resultChan := make(chan result, len(tasks))
	taskChan := make(chan embeddingTask, len(tasks))

	// Send all tasks to channel
	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				output, err := getEmbeddingWithModelType(task.pattern, c.modelType, 0)
				if err != nil {
					resultChan <- result{err: err, isJailbreak: task.isJailbreak}
				} else {
					resultChan <- result{embedding: output.Embedding, isJailbreak: task.isJailbreak}
				}
			}
		}(i)
	}

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	c.mu.Lock()
	defer c.mu.Unlock()

	c.jailbreakEmbeddings = make([][]float32, 0, len(c.jailbreakPatterns))
	c.benignEmbeddings = make([][]float32, 0, len(c.benignPatterns))

	var firstError error
	jailbreakCount := 0
	benignCount := 0

	for res := range resultChan {
		if res.err != nil {
			if firstError == nil {
				firstError = res.err
			}
			logging.Warnf("[Contrastive Jailbreak] Failed to compute embedding: %v", res.err)
		} else {
			if res.isJailbreak {
				c.jailbreakEmbeddings = append(c.jailbreakEmbeddings, res.embedding)
				jailbreakCount++
			} else {
				c.benignEmbeddings = append(c.benignEmbeddings, res.embedding)
				benignCount++
			}
		}
	}

	elapsed := time.Since(startTime)
	logging.Infof("[Contrastive Jailbreak] Preloaded %d jailbreak + %d benign KB embeddings in %v (workers: %d)",
		jailbreakCount, benignCount, elapsed, numWorkers)

	if firstError != nil {
		return firstError
	}

	return nil
}

// ComputeContrastiveScore computes the contrastive score for a single text
// Score = max_similarity(text, jailbreak_kb) - max_similarity(text, benign_kb)
// Positive score = closer to jailbreak patterns
// Negative score = closer to benign patterns
func (c *ContrastiveJailbreakClassifier) ComputeContrastiveScore(text string) (float32, error) {
	if text == "" {
		return 0.0, nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	// Compute embedding for input text
	output, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return 0.0, err
	}
	textEmbedding := output.Embedding

	// Compute max similarity to jailbreak patterns
	maxJailbreakSim := float32(-1.0)
	for _, jbEmb := range c.jailbreakEmbeddings {
		sim := cosineSimilarity(textEmbedding, jbEmb)
		if sim > maxJailbreakSim {
			maxJailbreakSim = sim
		}
	}

	// Compute max similarity to benign patterns
	maxBenignSim := float32(-1.0)
	for _, bnEmb := range c.benignEmbeddings {
		sim := cosineSimilarity(textEmbedding, bnEmb)
		if sim > maxBenignSim {
			maxBenignSim = sim
		}
	}

	// Contrastive score: how much more similar to jailbreak than benign
	contrastiveScore := maxJailbreakSim - maxBenignSim

	return contrastiveScore, nil
}

// AnalyzeConversation analyzes all user messages in a conversation for multi-turn jailbreak
// Returns the maximum contrastive score across all turns
func (c *ContrastiveJailbreakClassifier) AnalyzeConversation(userMessages []string) (*ContrastiveJailbreakResult, error) {
	result := &ContrastiveJailbreakResult{
		MaxScore:      0.0,
		IsJailbreak:   false,
		FlaggedIndex:  -1,
		ScoresPerTurn: make([]ContrastiveTurnScore, 0, len(userMessages)),
	}

	if len(userMessages) == 0 {
		return result, nil
	}

	// Compute contrastive score for each user message
	for i, msg := range userMessages {
		if msg == "" {
			continue
		}

		score, err := c.ComputeContrastiveScore(msg)
		if err != nil {
			logging.Warnf("[Contrastive Jailbreak] Failed to compute score for turn %d: %v", i, err)
			continue
		}

		// Truncate content for logging
		content := msg
		if len(content) > 100 {
			content = content[:100] + "..."
		}

		result.ScoresPerTurn = append(result.ScoresPerTurn, ContrastiveTurnScore{
			Content: content,
			Score:   score,
		})

		// Track maximum score
		if score > result.MaxScore {
			result.MaxScore = score
			if score > c.threshold && result.FlaggedIndex == -1 {
				result.FlaggedIndex = i
			}
		}
	}

	// Determine if jailbreak based on max score
	result.IsJailbreak = result.MaxScore > c.threshold

	if result.IsJailbreak {
		logging.Warnf("[Contrastive Jailbreak] MULTI-TURN ATTACK DETECTED: max_score=%.4f, threshold=%.4f, flagged_turn=%d",
			result.MaxScore, c.threshold, result.FlaggedIndex)
	} else {
		logging.Infof("[Contrastive Jailbreak] No multi-turn attack detected: max_score=%.4f, threshold=%.4f",
			result.MaxScore, c.threshold)
	}

	return result, nil
}

// GetThreshold returns the configured threshold
func (c *ContrastiveJailbreakClassifier) GetThreshold() float32 {
	return c.threshold
}

// GetKBSizes returns the sizes of the KB patterns
func (c *ContrastiveJailbreakClassifier) GetKBSizes() (jailbreak int, benign int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.jailbreakEmbeddings), len(c.benignEmbeddings)
}
