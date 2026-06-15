package classification

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type kbLabelData struct {
	Description string
	Exemplars   []string
	Embeddings  [][]float32
	Prototype   *prototypeBank
}

// KBClassifyResult contains the structured output of one embedding KB evaluation.
type KBClassifyResult struct {
	BestLabel             string
	BestSimilarity        float64
	BestLabelMargin       float64
	BestMatchedLabel      string
	BestMatchedSimilarity float64
	BestGroup             string
	BestMatchedGroup      string
	MatchedLabels         []string
	MatchedGroups         []string
	LabelConfidences      map[string]float64
	LabelBestScores       map[string]float64
	LabelSupportScores    map[string]float64
	GroupScores           map[string]float64
	MetricValues          map[string]float64
}

// KnowledgeBaseClassifier performs exemplar-based KB classification.
type KnowledgeBaseClassifier struct {
	rule        config.KnowledgeBaseConfig
	definition  config.KnowledgeBaseDefinition
	labels      map[string]*kbLabelData
	modelType   string
	baseDir     string
	preloadOnce sync.Once
	preloadErr  error
	preloaded   bool
}

func NewKnowledgeBaseClassifier(rule config.KnowledgeBaseConfig, modelType string, baseDir string) (*KnowledgeBaseClassifier, error) {
	rule = rule.WithDefaults()
	c := &KnowledgeBaseClassifier{
		rule:      rule,
		labels:    make(map[string]*kbLabelData),
		modelType: modelType,
		baseDir:   baseDir,
	}

	if err := c.loadDefinition(); err != nil {
		return nil, fmt.Errorf("failed to load KB manifest from %s: %w", rule.Source.Path, err)
	}
	if c.shouldDeferPreload() {
		logging.ComponentEvent("classifier", "knowledge_base_preload_deferred", map[string]interface{}{
			"knowledge_base": c.rule.Name,
			"labels":         len(c.labels),
			"backend":        c.currentBackend(),
		})
	} else if err := c.preloadEmbeddings(); err != nil {
		return nil, fmt.Errorf("failed to preload KB embeddings: %w", err)
	}
	return c, nil
}

func (c *KnowledgeBaseClassifier) currentBackend() string {
	return embeddingBackendOverride()
}

func (c *KnowledgeBaseClassifier) shouldDeferPreload() bool {
	backend := c.currentBackend()
	return backend == "" || backend == "candle"
}

func (c *KnowledgeBaseClassifier) ensureEmbeddingsPreloaded() error {
	if c.preloaded {
		return nil
	}
	c.preloadOnce.Do(func() {
		c.preloadErr = c.preloadEmbeddings()
		if c.preloadErr == nil {
			c.preloaded = true
		}
	})
	return c.preloadErr
}

func (c *KnowledgeBaseClassifier) loadDefinition() error {
	if c.labels == nil {
		c.labels = make(map[string]*kbLabelData)
	}
	definition, err := config.LoadKnowledgeBaseDefinition(c.baseDir, c.rule.Source)
	if err != nil {
		return err
	}
	if len(definition.Labels) == 0 {
		return fmt.Errorf("KB manifest contains no labels")
	}
	c.definition = definition
	for name, label := range definition.Labels {
		if len(label.Exemplars) == 0 {
			continue
		}
		c.labels[name] = &kbLabelData{
			Description: label.Description,
			Exemplars:   append([]string(nil), label.Exemplars...),
		}
	}
	if len(c.labels) == 0 {
		return fmt.Errorf("no valid labels found in KB manifest")
	}
	return nil
}

func (c *KnowledgeBaseClassifier) Classify(text string) (*KBClassifyResult, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("KB classification: query must be provided")
	}

	startTime := time.Now()
	if err := c.ensureEmbeddingsPreloaded(); err != nil {
		return nil, fmt.Errorf("failed to ensure KB embeddings are loaded: %w", err)
	}
	queryOutput, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	result := c.buildResultFromLabelScores(c.computeLabelScores(queryOutput.Embedding))

	elapsed := time.Since(startTime)
	logging.ComponentDebugEvent("classifier", "knowledge_base_classification_completed", map[string]interface{}{
		"knowledge_base":          c.rule.Name,
		"latency_ms":              elapsed.Milliseconds(),
		"best_label":              result.BestLabel,
		"best_similarity":         result.BestSimilarity,
		"best_matched_label":      result.BestMatchedLabel,
		"best_matched_similarity": result.BestMatchedSimilarity,
		"best_group":              result.BestGroup,
		"best_matched_group":      result.BestMatchedGroup,
	})

	return result, nil
}

func (c *KnowledgeBaseClassifier) LabelCount() int {
	return len(c.labels)
}
