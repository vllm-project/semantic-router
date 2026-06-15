package classification

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type exemplarRef struct {
	label string
	index int
	text  string
}

type embeddingResult struct {
	ref       exemplarRef
	embedding []float32
	err       error
}

func (c *KnowledgeBaseClassifier) collectExemplarRefs() []exemplarRef {
	refs := make([]exemplarRef, 0)
	for label, data := range c.labels {
		data.Embeddings = make([][]float32, len(data.Exemplars))
		for i, text := range data.Exemplars {
			refs = append(refs, exemplarRef{label: label, index: i, text: text})
		}
	}
	return refs
}

func (c *KnowledgeBaseClassifier) embedOneExemplar(backend, modelType string, targetDim int, ref exemplarRef) embeddingResult {
	if backend == "openvino" {
		embedding, err := getOpenVINOEmbedding(modelType, ref.text, targetDim)
		if err != nil {
			return embeddingResult{ref: ref, err: err}
		}
		return embeddingResult{ref: ref, embedding: embedding}
	}
	output, err := getEmbeddingWithModelType(ref.text, modelType, targetDim)
	if err != nil {
		return embeddingResult{ref: ref, err: err}
	}
	return embeddingResult{ref: ref, embedding: output.Embedding}
}

func (c *KnowledgeBaseClassifier) embedExemplarsParallel(refs []exemplarRef) <-chan embeddingResult {
	numWorkers := runtime.NumCPU()
	backend := embeddingBackendOverride()
	if backend == "candle" {
		numWorkers = 1
	} else if numWorkers > 8 {
		numWorkers = 8
	}
	if numWorkers > len(refs) {
		numWorkers = len(refs)
	}
	if numWorkers == 0 {
		numWorkers = 1
	}

	resultChan := make(chan embeddingResult, len(refs))
	refChan := make(chan exemplarRef, len(refs))
	for _, ref := range refs {
		refChan <- ref
	}
	close(refChan)

	modelType := c.modelType
	targetDim := 0

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ref := range refChan {
				resultChan <- c.embedOneExemplar(backend, modelType, targetDim, ref)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	return resultChan
}

func (c *KnowledgeBaseClassifier) preloadEmbeddings() error {
	startTime := time.Now()
	logging.ComponentEvent("classifier", "knowledge_base_embeddings_preload_started", map[string]interface{}{
		"knowledge_base": c.rule.Name,
		"labels":         len(c.labels),
		"backend":        c.currentBackend(),
	})
	refs := c.collectExemplarRefs()
	resultChan := c.embedExemplarsParallel(refs)

	failCount := 0
	for res := range resultChan {
		if res.err != nil {
			failCount++
			logging.Warnf("[KnowledgeBase:%s] Failed to embed exemplar %q in %s: %v", c.rule.Name, res.ref.text, res.ref.label, res.err)
			continue
		}
		c.labels[res.ref.label].Embeddings[res.ref.index] = res.embedding
	}

	logging.ComponentEvent("classifier", "knowledge_base_embeddings_preloaded", map[string]interface{}{
		"knowledge_base": c.rule.Name,
		"exemplars":      len(refs) - failCount,
		"labels":         len(c.labels),
		"latency_ms":     time.Since(startTime).Milliseconds(),
	})
	c.rebuildLabelPrototypeBanks()
	c.preloaded = true
	return nil
}

func (c *KnowledgeBaseClassifier) rebuildLabelPrototypeBanks() {
	for labelName, data := range c.labels {
		examples := make([]prototypeExample, 0, len(data.Exemplars))
		for i, text := range data.Exemplars {
			if i >= len(data.Embeddings) || len(data.Embeddings[i]) == 0 {
				continue
			}
			examples = append(examples, prototypeExample{
				Key:       fmt.Sprintf("%s:%d", labelName, i),
				Text:      text,
				Embedding: data.Embeddings[i],
			})
		}
		bank := newPrototypeBank(examples, c.rule.PrototypeScoring)
		data.Prototype = bank
		logPrototypeBankSummary("Knowledge Base "+c.rule.Name, labelName, bank)
	}
}
