package classification

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

type complexityQueryEmbeddings struct {
	text   []float32
	mmText []float32
	image  []float32
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

func (c *ComplexityClassifier) loadQueryEmbeddings(query string, imageURL string) (complexityQueryEmbeddings, error) {
	queryOutput, err := getEmbeddingWithModelType(query, c.modelType, 0)
	if err != nil {
		return complexityQueryEmbeddings{}, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	embeddings := complexityQueryEmbeddings{text: queryOutput.Embedding}
	if !c.hasImageCandidates {
		return embeddings, nil
	}

	embeddings.mmText = c.loadOptionalMultiModalTextEmbedding(query)
	if imageURL != "" {
		embeddings.image = c.loadOptionalMultiModalImageEmbedding(imageURL)
	}
	return embeddings, nil
}

func (c *ComplexityClassifier) loadOptionalMultiModalTextEmbedding(query string) []float32 {
	embedding, err := getMultiModalTextEmbedding(query, 0)
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute multimodal text embedding: %v", err)
		return nil
	}
	return embedding
}

func (c *ComplexityClassifier) loadOptionalMultiModalImageEmbedding(imageURL string) []float32 {
	embedding, err := getMultiModalImageEmbedding(imageURL, 0)
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
		return nil
	}
	return embedding
}

func (c *ComplexityClassifier) classifyRuleWithEmbeddings(
	rule config.ComplexityRule,
	queryEmbeddings complexityQueryEmbeddings,
	scoreOptions prototypeScoreOptions,
) ComplexityRuleResult {
	textHardScore, textEasyScore, textSignal := c.scoreTextSignal(rule.Name, queryEmbeddings.text, scoreOptions)
	imageHardScore, imageEasyScore, imageSignal, hasImage := c.scoreImageSignal(rule.Name, queryEmbeddings, scoreOptions)
	fusedSignal, signalSource := selectComplexitySignal(textSignal, imageSignal, hasImage)

	return ComplexityRuleResult{
		RuleName:       rule.Name,
		Difficulty:     classifyComplexityDifficulty(rule.Threshold, fusedSignal),
		TextHardScore:  textHardScore.Score,
		TextEasyScore:  textEasyScore.Score,
		TextMargin:     textSignal,
		ImageHardScore: imageHardScore.Score,
		ImageEasyScore: imageEasyScore.Score,
		ImageMargin:    imageSignal,
		FusedMargin:    fusedSignal,
		Confidence:     math.Abs(fusedSignal),
		SignalSource:   signalSource,
	}
}

func (c *ComplexityClassifier) scoreTextSignal(
	ruleName string,
	queryEmbedding []float32,
	scoreOptions prototypeScoreOptions,
) (prototypeBankScore, prototypeBankScore, float64) {
	hardScore := c.hardPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	easyScore := c.easyPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	return hardScore, easyScore, hardScore.Score - easyScore.Score
}

func (c *ComplexityClassifier) scoreImageSignal(
	ruleName string,
	queryEmbeddings complexityQueryEmbeddings,
	scoreOptions prototypeScoreOptions,
) (prototypeBankScore, prototypeBankScore, float64, bool) {
	queryEmbedding := queryEmbeddings.image
	if queryEmbedding == nil {
		queryEmbedding = queryEmbeddings.mmText
	}
	if queryEmbedding == nil {
		return prototypeBankScore{}, prototypeBankScore{}, 0, false
	}

	hardScore := c.imageHardPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	easyScore := c.imageEasyPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	if hardScore.PrototypeCount == 0 || easyScore.PrototypeCount == 0 {
		return hardScore, easyScore, 0, false
	}

	return hardScore, easyScore, hardScore.Score - easyScore.Score, true
}

func selectComplexitySignal(textSignal float64, imageSignal float64, hasImage bool) (float64, string) {
	if hasImage && math.Abs(imageSignal) > math.Abs(textSignal) {
		return imageSignal, "image"
	}
	return textSignal, "text"
}

func classifyComplexityDifficulty(threshold float32, signal float64) string {
	if signal > float64(threshold) {
		return "hard"
	}
	if signal < -float64(threshold) {
		return "easy"
	}
	return "medium"
}

func logComplexityRuleResult(
	rule config.ComplexityRule,
	result ComplexityRuleResult,
	requestImageProvided bool,
) {
	if result.SignalSource == "image" || result.ImageHardScore > 0 || result.ImageEasyScore > 0 {
		logging.Infof(
			"Complexity rule '%s': text_signal=%.3f, image_signal=%.3f (src=%s), fused=%s(%.3f), difficulty=%s",
			rule.Name,
			result.TextMargin,
			result.ImageMargin,
			complexityImageSourceLabel(requestImageProvided),
			result.SignalSource,
			result.FusedMargin,
			result.Difficulty,
		)
		return
	}
	logging.Infof(
		"Complexity rule '%s': hard_score=%.3f, easy_score=%.3f, signal=%.3f, difficulty=%s",
		rule.Name,
		result.TextHardScore,
		result.TextEasyScore,
		result.FusedMargin,
		result.Difficulty,
	)
}

func complexityImageSourceLabel(requestImageProvided bool) string {
	if requestImageProvided {
		return "screenshot"
	}
	return "mm_text"
}
