package classification

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
}

// NewContrastiveJailbreakClassifier creates and initialises a classifier for a
// single contrastive JailbreakRule. KB embeddings are computed eagerly using a
// worker pool (same approach as ComplexityClassifier).
func NewContrastiveJailbreakClassifier(rule config.JailbreakRule, defaultModelType string) (*ContrastiveJailbreakClassifier, error) {
	modelType := defaultModelType
	if modelType == "" {
		modelType = "qwen3"
	}

	c := &ContrastiveJailbreakClassifier{
		rule:                rule,
		jailbreakEmbeddings: make(map[string][]float32),
		benignEmbeddings:    make(map[string][]float32),
		modelType:           modelType,
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
		output, err := getEmbeddingWithModelType(msg, c.modelType, 0)
		if err != nil {
			logging.Warnf("[Contrastive Jailbreak] Failed to embed message %d: %v", i, err)
			continue
		}
		msgEmb := output.Embedding

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
	return result
}

// preloadKBEmbeddings concurrently computes embeddings for jailbreak and benign
// pattern knowledge bases, following the same worker-pool approach as
// ComplexityClassifier.preloadCandidateEmbeddings.
func (c *ContrastiveJailbreakClassifier) preloadKBEmbeddings() error {
	startTime := time.Now()

	logging.Infof("[Contrastive Jailbreak] Preloading KB embeddings for rule %q (jailbreak: %d, benign: %d, model: %s)",
		c.rule.Name, len(c.rule.JailbreakPatterns), len(c.rule.BenignPatterns), c.modelType)

	type task struct {
		text        string
		isJailbreak bool
	}

	var tasks []task
	for _, p := range c.rule.JailbreakPatterns {
		tasks = append(tasks, task{text: p, isJailbreak: true})
	}
	for _, p := range c.rule.BenignPatterns {
		tasks = append(tasks, task{text: p, isJailbreak: false})
	}

	if len(tasks) == 0 {
		logging.Warnf("[Contrastive Jailbreak] Rule %q has no KB patterns", c.rule.Name)
		return nil
	}

	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(tasks) {
		numWorkers = len(tasks)
	}

	type res struct {
		text        string
		embedding   []float32
		isJailbreak bool
		err         error
	}

	taskCh := make(chan task, len(tasks))
	resCh := make(chan res, len(tasks))

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
				output, err := getEmbeddingWithModelType(t.text, c.modelType, 0)
				if err != nil {
					resCh <- res{text: t.text, isJailbreak: t.isJailbreak, err: err}
				} else {
					resCh <- res{text: t.text, embedding: output.Embedding, isJailbreak: t.isJailbreak}
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resCh)
	}()

	var mu sync.Mutex
	var firstErr error
	ok := 0

	for r := range resCh {
		if r.err != nil {
			if firstErr == nil {
				kind := "benign"
				if r.isJailbreak {
					kind = "jailbreak"
				}
				firstErr = fmt.Errorf("failed to embed %s pattern %q: %w", kind, r.text, r.err)
			}
			continue
		}
		mu.Lock()
		if r.isJailbreak {
			c.jailbreakEmbeddings[r.text] = r.embedding
		} else {
			c.benignEmbeddings[r.text] = r.embedding
		}
		mu.Unlock()
		ok++
	}

	elapsed := time.Since(startTime)
	logging.Infof("[Contrastive Jailbreak] Rule %q: preloaded %d/%d KB embeddings in %v (workers: %d)",
		c.rule.Name, ok, len(tasks), elapsed, numWorkers)

	return firstErr
}
