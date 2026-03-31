package classification

import (
	"fmt"
	"runtime"
	"sort"
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
}

// KBClassifyResult contains the structured output of one embedding KB evaluation.
type KBClassifyResult struct {
	BestLabel             string
	BestSimilarity        float64
	BestMatchedLabel      string
	BestMatchedSimilarity float64
	BestGroup             string
	BestMatchedGroup      string
	MatchedLabels         []string
	MatchedGroups         []string
	LabelConfidences      map[string]float64
	GroupScores           map[string]float64
	MetricValues          map[string]float64
}

// KnowledgeBaseClassifier performs exemplar-based KB classification.
type KnowledgeBaseClassifier struct {
	rule       config.KnowledgeBaseConfig
	definition config.KnowledgeBaseDefinition
	labels     map[string]*kbLabelData
	modelType  string
	baseDir    string
}

func NewKnowledgeBaseClassifier(rule config.KnowledgeBaseConfig, modelType string, baseDir string) (*KnowledgeBaseClassifier, error) {
	c := &KnowledgeBaseClassifier{
		rule:      rule,
		labels:    make(map[string]*kbLabelData),
		modelType: modelType,
		baseDir:   baseDir,
	}

	if err := c.loadDefinition(); err != nil {
		return nil, fmt.Errorf("failed to load KB manifest from %s: %w", rule.Source.Path, err)
	}
	if err := c.preloadEmbeddings(); err != nil {
		return nil, fmt.Errorf("failed to preload KB embeddings: %w", err)
	}
	return c, nil
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

func (c *KnowledgeBaseClassifier) embedExemplarsParallel(refs []exemplarRef) <-chan embeddingResult {
	numWorkers := runtime.NumCPU() * 2
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
				output, err := getEmbeddingWithModelType(ref.text, modelType, targetDim)
				if err != nil {
					resultChan <- embeddingResult{ref: ref, err: err}
					continue
				}
				resultChan <- embeddingResult{ref: ref, embedding: output.Embedding}
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

	logging.ComponentDebugEvent("classifier", "knowledge_base_embeddings_preloaded", map[string]interface{}{
		"knowledge_base": c.rule.Name,
		"exemplars":      len(refs) - failCount,
		"labels":         len(c.labels),
		"latency_ms":     time.Since(startTime).Milliseconds(),
	})
	return nil
}

func (c *KnowledgeBaseClassifier) computeLabelSimilarities(queryEmb []float32) map[string]float64 {
	labelScores := make(map[string]float64, len(c.labels))
	for labelName, data := range c.labels {
		maxSim := float32(0)
		for _, emb := range data.Embeddings {
			if emb == nil {
				continue
			}
			if sim := cosineSimilarity(queryEmb, emb); sim > maxSim {
				maxSim = sim
			}
		}
		labelScores[labelName] = float64(maxSim)
	}
	return labelScores
}

func (c *KnowledgeBaseClassifier) effectiveThreshold(label string) float64 {
	if threshold, ok := c.rule.LabelThresholds[label]; ok {
		return float64(threshold)
	}
	return float64(c.rule.Threshold)
}

func (c *KnowledgeBaseClassifier) buildMatchedLabels(labelScores map[string]float64) []string {
	matched := make([]string, 0, len(labelScores))
	for label, score := range labelScores {
		if score >= c.effectiveThreshold(label) {
			matched = append(matched, label)
		}
	}
	sort.Strings(matched)
	return matched
}

func (c *KnowledgeBaseClassifier) computeGroupScores(labelScores map[string]float64) map[string]float64 {
	groupScores := make(map[string]float64, len(c.rule.Groups))
	for group, labels := range c.rule.Groups {
		best := 0.0
		for _, label := range labels {
			if score := labelScores[label]; score > best {
				best = score
			}
		}
		groupScores[group] = best
	}
	return groupScores
}

func (c *KnowledgeBaseClassifier) collectMatchedGroups(matchedLabels []string) []string {
	if len(c.rule.Groups) == 0 || len(matchedLabels) == 0 {
		return nil
	}
	labelSet := make(map[string]struct{}, len(matchedLabels))
	for _, label := range matchedLabels {
		labelSet[label] = struct{}{}
	}
	groups := make([]string, 0, len(c.rule.Groups))
	for group, labels := range c.rule.Groups {
		for _, label := range labels {
			if _, ok := labelSet[label]; ok {
				groups = append(groups, group)
				break
			}
		}
	}
	sort.Strings(groups)
	return groups
}

func bestScoredName(scores map[string]float64) (string, float64) {
	if len(scores) == 0 {
		return "", 0
	}
	names := make([]string, 0, len(scores))
	for name := range scores {
		names = append(names, name)
	}
	sort.Strings(names)
	bestName := ""
	bestScore := 0.0
	for _, name := range names {
		score := scores[name]
		if bestName == "" || score > bestScore {
			bestName = name
			bestScore = score
		}
	}
	return bestName, bestScore
}

func (c *KnowledgeBaseClassifier) computeMetricValues(labelScores, groupScores map[string]float64, bestScore, bestMatchedScore float64) map[string]float64 {
	values := map[string]float64{
		config.KBMetricBestScore:        bestScore,
		config.KBMetricBestMatchedScore: bestMatchedScore,
	}
	for _, metric := range c.rule.Metrics {
		if metric.Type != config.KBMetricTypeGroupMargin {
			continue
		}
		values[metric.Name] = groupScores[metric.PositiveGroup] - groupScores[metric.NegativeGroup]
	}
	return values
}

func (c *KnowledgeBaseClassifier) Classify(text string) (*KBClassifyResult, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("KB classification: query must be provided")
	}

	startTime := time.Now()
	queryOutput, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	labelScores := c.computeLabelSimilarities(queryOutput.Embedding)
	bestLabel, bestScore := bestScoredName(labelScores)
	matchedLabels := c.buildMatchedLabels(labelScores)
	matchedLabelScores := make(map[string]float64, len(matchedLabels))
	for _, label := range matchedLabels {
		matchedLabelScores[label] = labelScores[label]
	}
	bestMatchedLabel, bestMatchedScore := bestScoredName(matchedLabelScores)
	groupScores := c.computeGroupScores(labelScores)
	bestGroup, _ := bestScoredName(groupScores)
	matchedGroups := c.collectMatchedGroups(matchedLabels)
	matchedGroupScores := make(map[string]float64, len(matchedGroups))
	for _, group := range matchedGroups {
		matchedGroupScores[group] = groupScores[group]
	}
	bestMatchedGroup, _ := bestScoredName(matchedGroupScores)
	metricValues := c.computeMetricValues(labelScores, groupScores, bestScore, bestMatchedScore)

	elapsed := time.Since(startTime)
	logging.Infof("[KnowledgeBase:%s] Classified in %v: best_label=%s (%.3f), best_matched_label=%s (%.3f), best_group=%s, best_matched_group=%s",
		c.rule.Name, elapsed, bestLabel, bestScore, bestMatchedLabel, bestMatchedScore, bestGroup, bestMatchedGroup)

	return &KBClassifyResult{
		BestLabel:             bestLabel,
		BestSimilarity:        bestScore,
		BestMatchedLabel:      bestMatchedLabel,
		BestMatchedSimilarity: bestMatchedScore,
		BestGroup:             bestGroup,
		BestMatchedGroup:      bestMatchedGroup,
		MatchedLabels:         matchedLabels,
		MatchedGroups:         matchedGroups,
		LabelConfidences:      labelScores,
		GroupScores:           groupScores,
		MetricValues:          metricValues,
	}, nil
}

func (c *KnowledgeBaseClassifier) LabelCount() int {
	return len(c.labels)
}
