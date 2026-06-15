package classification

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type preferenceEmbeddingTask struct {
	ruleName string
	text     string
}

type preferenceEmbeddingResult struct {
	ruleName  string
	embedding []float32
	err       error
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
