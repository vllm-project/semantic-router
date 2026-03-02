package classification

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ComplexityClassifier performs complexity-based classification using embedding similarity.
// Each rule independently classifies difficulty level using hard/easy candidates.
// Supports both text candidates (via text embedding model) and image candidates
// (via the multimodal embedding model) for contrastive knowledge base comparison.
// Results are filtered by composer conditions in the classifier layer.
type ComplexityClassifier struct {
	rules []config.ComplexityRule

	// Precomputed text embeddings for hard and easy candidates
	hardEmbeddings map[string]map[string][]float32 // ruleName -> candidate -> embedding
	easyEmbeddings map[string]map[string][]float32 // ruleName -> candidate -> embedding

	// Precomputed image embeddings for hard and easy image candidates (multimodal)
	imageHardEmbeddings map[string]map[string][]float32 // ruleName -> imageRef -> embedding
	imageEasyEmbeddings map[string]map[string][]float32 // ruleName -> imageRef -> embedding

	modelType          string // Model type for text embeddings ("qwen3" or "gemma")
	hasImageCandidates bool   // True if any rule uses image_candidates
}

// NewComplexityClassifier creates a new ComplexityClassifier with precomputed candidate embeddings.
// When rules contain image_candidates, the multimodal model must be initialized beforehand.
func NewComplexityClassifier(rules []config.ComplexityRule, modelType string) (*ComplexityClassifier, error) {
	if modelType == "" {
		modelType = "qwen3"
	}

	c := &ComplexityClassifier{
		rules:               rules,
		hardEmbeddings:      make(map[string]map[string][]float32),
		easyEmbeddings:      make(map[string]map[string][]float32),
		imageHardEmbeddings: make(map[string]map[string][]float32),
		imageEasyEmbeddings: make(map[string]map[string][]float32),
		modelType:           modelType,
		hasImageCandidates:  config.HasImageCandidatesInRules(rules),
	}

	if c.hasImageCandidates {
		logging.Infof("ComplexityClassifier initialized with model type: %s + multimodal (image candidates detected)", c.modelType)
	} else {
		logging.Infof("ComplexityClassifier initialized with model type: %s", c.modelType)
	}

	if err := c.preloadCandidateEmbeddings(); err != nil {
		logging.Warnf("Failed to preload complexity candidate embeddings: %v", err)
		return nil, err
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all hard/easy candidates (text + image).
// Uses concurrent processing for better performance.
func (c *ComplexityClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()

	logging.Infof("[Complexity Signal] Preloading embeddings for hard/easy candidates using model: %s with concurrent processing...", c.modelType)

	type candidateTask struct {
		ruleName  string
		candidate string
		isHard    bool
		isImage   bool // true for image candidates (use multimodal model)
	}

	var tasks []candidateTask
	for _, rule := range c.rules {
		c.hardEmbeddings[rule.Name] = make(map[string][]float32)
		c.easyEmbeddings[rule.Name] = make(map[string][]float32)
		c.imageHardEmbeddings[rule.Name] = make(map[string][]float32)
		c.imageEasyEmbeddings[rule.Name] = make(map[string][]float32)

		for _, candidate := range rule.Hard.Candidates {
			tasks = append(tasks, candidateTask{
				ruleName: rule.Name, candidate: candidate, isHard: true, isImage: false,
			})
		}
		for _, candidate := range rule.Easy.Candidates {
			tasks = append(tasks, candidateTask{
				ruleName: rule.Name, candidate: candidate, isHard: false, isImage: false,
			})
		}
		for _, imageRef := range rule.Hard.ImageCandidates {
			tasks = append(tasks, candidateTask{
				ruleName: rule.Name, candidate: imageRef, isHard: true, isImage: true,
			})
		}
		for _, imageRef := range rule.Easy.ImageCandidates {
			tasks = append(tasks, candidateTask{
				ruleName: rule.Name, candidate: imageRef, isHard: false, isImage: true,
			})
		}
	}

	if len(tasks) == 0 {
		logging.Infof("[Complexity Signal] No candidates to preload")
		return nil
	}

	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(tasks) {
		numWorkers = len(tasks)
	}

	type result struct {
		ruleName  string
		candidate string
		embedding []float32
		isHard    bool
		isImage   bool
		err       error
	}

	resultChan := make(chan result, len(tasks))
	taskChan := make(chan candidateTask, len(tasks))

	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	var wg sync.WaitGroup
	var mu sync.Mutex

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				var emb []float32
				var err error

				if task.isImage {
					emb, err = getMultiModalImageEmbedding(task.candidate, 0)
				} else {
					output, embErr := getEmbeddingWithModelType(task.candidate, c.modelType, 0)
					if embErr != nil {
						err = embErr
					} else {
						emb = output.Embedding
					}
				}

				resultChan <- result{
					ruleName:  task.ruleName,
					candidate: task.candidate,
					embedding: emb,
					isHard:    task.isHard,
					isImage:   task.isImage,
					err:       err,
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	var firstError error
	successCount := 0
	for res := range resultChan {
		if res.err != nil {
			kind := "easy"
			if res.isHard {
				kind = "hard"
			}
			modality := "text"
			if res.isImage {
				modality = "image"
			}
			if firstError == nil {
				firstError = fmt.Errorf("failed to compute %s %s embedding for candidate '%s': %w",
					modality, kind, res.candidate, res.err)
			}
			logging.Warnf("Failed to compute %s %s embedding for candidate '%s': %v",
				modality, kind, res.candidate, res.err)
		} else {
			mu.Lock()
			if res.isImage {
				if res.isHard {
					c.imageHardEmbeddings[res.ruleName][res.candidate] = res.embedding
				} else {
					c.imageEasyEmbeddings[res.ruleName][res.candidate] = res.embedding
				}
			} else {
				if res.isHard {
					c.hardEmbeddings[res.ruleName][res.candidate] = res.embedding
				} else {
					c.easyEmbeddings[res.ruleName][res.candidate] = res.embedding
				}
			}
			mu.Unlock()
			successCount++
		}
	}

	elapsed := time.Since(startTime)
	logging.Infof("[Complexity Signal] Preloaded %d/%d complexity embeddings (text+image hard/easy candidates) using model %s in %v (workers: %d)",
		successCount, len(tasks), c.modelType, elapsed, numWorkers)

	if firstError != nil {
		return firstError
	}

	return nil
}

// Classify evaluates the query against ALL complexity rules independently (text-only).
// For CUA requests with screenshots, use ClassifyWithImage instead.
func (c *ComplexityClassifier) Classify(query string) ([]string, error) {
	return c.ClassifyWithImage(query, "")
}

// ClassifyWithImage evaluates the query (and optionally a request image) against
// ALL complexity rules independently.
//
// When imageURL is provided (e.g. a base64 data-URI screenshot from a CUA request),
// SigLIP encodes the image and compares it against the image knowledge base.
// The text query is always compared against the text knowledge base.
// The difficulty score fuses both channels: d(t) = max(|d_vis|, |d_sem|).
//
// Returns: all matched rules in format "rulename:difficulty"
// (e.g., ["cua_difficulty:hard", "cua_difficulty:easy"])
func (c *ComplexityClassifier) ClassifyWithImage(query string, imageURL string) ([]string, error) {
	if len(c.rules) == 0 {
		return nil, nil
	}

	// Compute text query embedding once
	queryOutput, err := getEmbeddingWithModelType(query, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	// Compute multimodal text embedding for text-vs-image-candidate comparison
	var mmTextEmbedding []float32
	if c.hasImageCandidates {
		mmEmb, mmErr := getMultiModalTextEmbedding(query, 0)
		if mmErr != nil {
			logging.Warnf("[Complexity Signal] Failed to compute multimodal text embedding: %v", mmErr)
		} else {
			mmTextEmbedding = mmEmb
		}
	}

	// Compute multimodal image embedding of the request screenshot (SigLIP)
	var mmImageEmbedding []float32
	if imageURL != "" && c.hasImageCandidates {
		imgEmb, imgErr := getMultiModalImageEmbedding(imageURL, 0)
		if imgErr != nil {
			logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", imgErr)
		} else {
			mmImageEmbedding = imgEmb
		}
	}

	var matchedRules []string

	for i := range c.rules {
		rule := &c.rules[i]

		// --- Text signal (d_sem): text query vs text candidates ---
		maxHardSim := float32(-1.0)
		for _, hardEmb := range c.hardEmbeddings[rule.Name] {
			if sim := cosineSimilarity(queryEmbedding, hardEmb); sim > maxHardSim {
				maxHardSim = sim
			}
		}

		maxEasySim := float32(-1.0)
		for _, easyEmb := range c.easyEmbeddings[rule.Name] {
			if sim := cosineSimilarity(queryEmbedding, easyEmb); sim > maxEasySim {
				maxEasySim = sim
			}
		}

		textSignal := maxHardSim - maxEasySim

		// --- Image signal (d_vis): request screenshot vs image candidates ---
		// Uses the request image when available (image-to-image), falls back to
		// multimodal text embedding (text-to-image) when no screenshot is present.
		imageSignal := float32(0.0)
		hasImage := false
		queryEmb := mmImageEmbedding
		if queryEmb == nil {
			queryEmb = mmTextEmbedding
		}
		if queryEmb != nil {
			imgMaxHard := float32(-1.0)
			for _, hardEmb := range c.imageHardEmbeddings[rule.Name] {
				if sim := cosineSimilarity(queryEmb, hardEmb); sim > imgMaxHard {
					imgMaxHard = sim
				}
			}
			imgMaxEasy := float32(-1.0)
			for _, easyEmb := range c.imageEasyEmbeddings[rule.Name] {
				if sim := cosineSimilarity(queryEmb, easyEmb); sim > imgMaxEasy {
					imgMaxEasy = sim
				}
			}
			if imgMaxHard > -1.0 && imgMaxEasy > -1.0 {
				imageSignal = imgMaxHard - imgMaxEasy
				hasImage = true
			}
		}

		// Fuse signals: d(t) = max(|d_vis|, |d_sem|)
		difficultySignal := textSignal
		signalSource := "text"
		if hasImage && float32(math.Abs(float64(imageSignal))) > float32(math.Abs(float64(textSignal))) {
			difficultySignal = imageSignal
			signalSource = "image"
		}

		var difficulty string
		if difficultySignal > rule.Threshold {
			difficulty = "hard"
		} else if difficultySignal < -rule.Threshold {
			difficulty = "easy"
		} else {
			difficulty = "medium"
		}

		if hasImage {
			logging.Infof("Complexity rule '%s': text_signal=%.3f, image_signal=%.3f (src=%s), fused=%s(%.3f), difficulty=%s",
				rule.Name, textSignal, imageSignal,
				func() string {
					if mmImageEmbedding != nil {
						return "screenshot"
					}
					return "mm_text"
				}(),
				signalSource, difficultySignal, difficulty)
		} else {
			logging.Infof("Complexity rule '%s': hard_sim=%.3f, easy_sim=%.3f, signal=%.3f, difficulty=%s",
				rule.Name, maxHardSim, maxEasySim, difficultySignal, difficulty)
		}

		matchedRules = append(matchedRules, fmt.Sprintf("%s:%s", rule.Name, difficulty))
	}

	return matchedRules, nil
}
