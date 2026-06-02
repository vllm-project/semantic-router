package classification

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type complexityCandidateTask struct {
	ruleName  string
	candidate string
	isHard    bool
	isImage   bool
}

type complexityCandidateResult struct {
	ruleName  string
	candidate string
	embedding []float32
	isHard    bool
	isImage   bool
	err       error
}

// preloadCandidateEmbeddings computes embeddings for all hard/easy candidates (text + image).
// Uses concurrent processing for better performance.
func (c *ComplexityClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()
	logging.ComponentDebugEvent("classifier", "complexity_candidates_preload_started", map[string]interface{}{
		"model_type":       c.modelType,
		"image_candidates": c.hasImageCandidates,
	})
	tasks := c.buildCandidateTasks()
	if len(tasks) == 0 {
		logging.ComponentDebugEvent("classifier", "complexity_candidates_preload_skipped", map[string]interface{}{
			"reason": "no_candidates",
		})
		return nil
	}

	numWorkers := complexityWorkerCount(len(tasks))
	successCount, firstError := c.collectCandidateEmbeddingResults(c.startCandidateEmbeddingWorkers(tasks, numWorkers))

	elapsed := time.Since(startTime)
	logging.ComponentEvent("classifier", "complexity_candidates_preloaded", map[string]interface{}{
		"candidates":       successCount,
		"total_candidates": len(tasks),
		"model_type":       c.modelType,
		"image_candidates": c.hasImageCandidates,
		"workers":          numWorkers,
		"elapsed_ms":       elapsed.Milliseconds(),
	})

	if firstError != nil {
		return firstError
	}

	c.rebuildPrototypeBanks()

	return nil
}

func (c *ComplexityClassifier) buildCandidateTasks() []complexityCandidateTask {
	tasks := make([]complexityCandidateTask, 0)
	for _, rule := range c.rules {
		c.hardEmbeddings[rule.Name] = make(map[string][]float32)
		c.easyEmbeddings[rule.Name] = make(map[string][]float32)
		c.imageHardEmbeddings[rule.Name] = make(map[string][]float32)
		c.imageEasyEmbeddings[rule.Name] = make(map[string][]float32)

		tasks = appendComplexityTasks(tasks, rule.Name, rule.Hard.Candidates, true, false)
		tasks = appendComplexityTasks(tasks, rule.Name, rule.Easy.Candidates, false, false)
		tasks = appendComplexityTasks(tasks, rule.Name, rule.Hard.ImageCandidates, true, true)
		tasks = appendComplexityTasks(tasks, rule.Name, rule.Easy.ImageCandidates, false, true)
	}
	return tasks
}

func appendComplexityTasks(
	tasks []complexityCandidateTask,
	ruleName string,
	candidates []string,
	isHard bool,
	isImage bool,
) []complexityCandidateTask {
	for _, candidate := range candidates {
		tasks = append(tasks, complexityCandidateTask{
			ruleName:  ruleName,
			candidate: candidate,
			isHard:    isHard,
			isImage:   isImage,
		})
	}
	return tasks
}

func complexityWorkerCount(taskCount int) int {
	if taskCount <= 1 {
		return 1
	}
	backend := embeddingBackendOverride()
	if backend == "" || backend == "candle" {
		return 1
	}
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > taskCount {
		return taskCount
	}
	return numWorkers
}

func (c *ComplexityClassifier) startCandidateEmbeddingWorkers(
	tasks []complexityCandidateTask,
	numWorkers int,
) <-chan complexityCandidateResult {
	resultChan := make(chan complexityCandidateResult, len(tasks))
	taskChan := make(chan complexityCandidateTask, len(tasks))
	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				embedding, err := c.computeCandidateEmbedding(task)
				resultChan <- complexityCandidateResult{
					ruleName:  task.ruleName,
					candidate: task.candidate,
					embedding: embedding,
					isHard:    task.isHard,
					isImage:   task.isImage,
					err:       err,
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()
	return resultChan
}

func (c *ComplexityClassifier) computeCandidateEmbedding(task complexityCandidateTask) ([]float32, error) {
	if task.isImage {
		return getMultiModalImageEmbedding(task.candidate, 0)
	}
	output, err := getEmbeddingWithModelType(task.candidate, c.modelType, 0)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

func (c *ComplexityClassifier) collectCandidateEmbeddingResults(
	resultChan <-chan complexityCandidateResult,
) (int, error) {
	successCount := 0
	var firstError error
	for res := range resultChan {
		if res.err != nil {
			if firstError == nil {
				firstError = fmt.Errorf(
					"failed to compute %s %s embedding for candidate '%s': %w",
					complexityCandidateModality(res),
					complexityCandidateKind(res),
					res.candidate,
					res.err,
				)
			}
			logging.Warnf(
				"Failed to compute %s %s embedding for candidate '%s': %v",
				complexityCandidateModality(res),
				complexityCandidateKind(res),
				res.candidate,
				res.err,
			)
			continue
		}
		c.storeCandidateEmbeddingResult(res)
		successCount++
	}
	return successCount, firstError
}

func complexityCandidateKind(result complexityCandidateResult) string {
	if result.isHard {
		return "hard"
	}
	return "easy"
}

func complexityCandidateModality(result complexityCandidateResult) string {
	if result.isImage {
		return "image"
	}
	return "text"
}

func (c *ComplexityClassifier) storeCandidateEmbeddingResult(result complexityCandidateResult) {
	if result.isImage {
		if result.isHard {
			c.imageHardEmbeddings[result.ruleName][result.candidate] = result.embedding
			return
		}
		c.imageEasyEmbeddings[result.ruleName][result.candidate] = result.embedding
		return
	}
	if result.isHard {
		c.hardEmbeddings[result.ruleName][result.candidate] = result.embedding
		return
	}
	c.easyEmbeddings[result.ruleName][result.candidate] = result.embedding
}

func (c *ComplexityClassifier) rebuildPrototypeBanks() {
	for _, rule := range c.rules {
		hardExamples := make([]prototypeExample, 0, len(c.hardEmbeddings[rule.Name]))
		for candidate, embedding := range c.hardEmbeddings[rule.Name] {
			hardExamples = append(hardExamples, prototypeExample{Key: rule.Name + ":hard:" + candidate, Text: candidate, Embedding: embedding})
		}
		easyExamples := make([]prototypeExample, 0, len(c.easyEmbeddings[rule.Name]))
		for candidate, embedding := range c.easyEmbeddings[rule.Name] {
			easyExamples = append(easyExamples, prototypeExample{Key: rule.Name + ":easy:" + candidate, Text: candidate, Embedding: embedding})
		}
		imageHardExamples := make([]prototypeExample, 0, len(c.imageHardEmbeddings[rule.Name]))
		for candidate, embedding := range c.imageHardEmbeddings[rule.Name] {
			imageHardExamples = append(imageHardExamples, prototypeExample{Key: rule.Name + ":image-hard:" + candidate, Text: candidate, Embedding: embedding})
		}
		imageEasyExamples := make([]prototypeExample, 0, len(c.imageEasyEmbeddings[rule.Name]))
		for candidate, embedding := range c.imageEasyEmbeddings[rule.Name] {
			imageEasyExamples = append(imageEasyExamples, prototypeExample{Key: rule.Name + ":image-easy:" + candidate, Text: candidate, Embedding: embedding})
		}
		hardBank := newPrototypeBank(hardExamples, c.prototypeCfg)
		easyBank := newPrototypeBank(easyExamples, c.prototypeCfg)
		imageHardBank := newPrototypeBank(imageHardExamples, c.prototypeCfg)
		imageEasyBank := newPrototypeBank(imageEasyExamples, c.prototypeCfg)
		c.hardPrototypeBanks[rule.Name] = hardBank
		c.easyPrototypeBanks[rule.Name] = easyBank
		c.imageHardPrototypeBanks[rule.Name] = imageHardBank
		c.imageEasyPrototypeBanks[rule.Name] = imageEasyBank
		logPrototypeBankSummary("Complexity hard", rule.Name, hardBank)
		logPrototypeBankSummary("Complexity easy", rule.Name, easyBank)
		if len(imageHardExamples) > 0 || len(imageEasyExamples) > 0 {
			logPrototypeBankSummary("Complexity image-hard", rule.Name, imageHardBank)
			logPrototypeBankSummary("Complexity image-easy", rule.Name, imageEasyBank)
		}
	}
}
