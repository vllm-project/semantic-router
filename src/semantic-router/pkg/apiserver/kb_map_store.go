//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var knowledgeBaseMapEmbeddingFunc = candle_binding.GetEmbeddingWithModelType

const kbMapProjectionName = "umap_2d"

type knowledgeBaseMapMetadataResponse struct {
	Name           string              `json:"name"`
	Description    string              `json:"description,omitempty"`
	Projection     string              `json:"projection"`
	ModelType      string              `json:"model_type"`
	PointCount     int                 `json:"point_count"`
	LabelCount     int                 `json:"label_count"`
	GroupCount     int                 `json:"group_count"`
	LabelNames     []string            `json:"label_names"`
	TopicLabelHint []string            `json:"topic_label_hint,omitempty"`
	Groups         map[string][]string `json:"groups,omitempty"`
}

type knowledgeBaseMapArtifacts struct {
	signature string
	metadata  knowledgeBaseMapMetadataResponse
	pointData []byte
}

type knowledgeBaseMapCache struct {
	mu    sync.RWMutex
	items map[string]*knowledgeBaseMapArtifacts
}

type kbRawPoint struct {
	Text       string    `json:"text"`
	LabelName  string    `json:"label_name"`
	LabelIndex int       `json:"label_index"`
	Vector     []float64 `json:"vector"`
}

func newKnowledgeBaseMapCache() *knowledgeBaseMapCache {
	return &knowledgeBaseMapCache{items: make(map[string]*knowledgeBaseMapArtifacts)}
}

func (s *ClassificationAPIServer) kbMapCache() *knowledgeBaseMapCache {
	if s.knowledgeBaseMapCache == nil {
		s.knowledgeBaseMapCache = newKnowledgeBaseMapCache()
	}
	return s.knowledgeBaseMapCache
}

func (c *knowledgeBaseMapCache) get(name, signature string) (*knowledgeBaseMapArtifacts, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	item, ok := c.items[name]
	if !ok || item.signature != signature {
		return nil, false
	}
	return item, true
}

func (c *knowledgeBaseMapCache) put(name string, item *knowledgeBaseMapArtifacts) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[name] = item
}

func knowledgeBaseMapSignature(
	kb config.KnowledgeBaseConfig,
	definition config.KnowledgeBaseDefinition,
	modelType string,
) (string, error) {
	payload := struct {
		KnowledgeBase config.KnowledgeBaseConfig     `json:"kb"`
		Definition    config.KnowledgeBaseDefinition `json:"definition"`
		ModelType     string                         `json:"model_type"`
		Projection    string                         `json:"projection"`
	}{
		KnowledgeBase: kb,
		Definition:    definition,
		ModelType:     modelType,
		Projection:    kbMapProjectionName,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func knowledgeBaseMapModelType(cfg *config.RouterConfig) string {
	modelType := strings.ToLower(strings.TrimSpace(cfg.EmbeddingConfig.ModelType))
	if modelType == "" {
		return "qwen3"
	}
	return modelType
}

func (s *ClassificationAPIServer) ensureKnowledgeBaseMapArtifacts(
	cfg *config.RouterConfig,
	baseDir string,
	kb config.KnowledgeBaseConfig,
) (*knowledgeBaseMapArtifacts, error) {
	definition, err := config.LoadKnowledgeBaseDefinition(baseDir, kb.Source)
	if err != nil {
		return nil, err
	}
	modelType := knowledgeBaseMapModelType(cfg)
	signature, err := knowledgeBaseMapSignature(kb, definition, modelType)
	if err != nil {
		return nil, err
	}
	if item, ok := s.kbMapCache().get(kb.Name, signature); ok {
		return item, nil
	}

	item, err := buildKnowledgeBaseMapArtifacts(kb, definition, modelType)
	if err != nil {
		return nil, err
	}
	item.signature = signature
	s.kbMapCache().put(kb.Name, item)
	return item, nil
}

func buildKnowledgeBaseMapArtifacts(
	kb config.KnowledgeBaseConfig,
	definition config.KnowledgeBaseDefinition,
	modelType string,
) (*knowledgeBaseMapArtifacts, error) {
	labelNames := sortedKnowledgeBaseLabelNames(definition)
	if len(labelNames) == 0 {
		return nil, fmt.Errorf("knowledge base %q has no labels", kb.Name)
	}

	rawPoints, err := buildKnowledgeBaseRawPoints(definition, labelNames, modelType)
	if err != nil {
		return nil, err
	}
	if len(rawPoints) == 0 {
		return nil, fmt.Errorf("knowledge base %q produced no map points", kb.Name)
	}

	pointData, err := marshalKnowledgeBasePointNDJSON(rawPoints)
	if err != nil {
		return nil, err
	}

	metadata := knowledgeBaseMapMetadataResponse{
		Name:           kb.Name,
		Description:    strings.TrimSpace(definition.Description),
		Projection:     kbMapProjectionName,
		ModelType:      modelType,
		PointCount:     len(rawPoints),
		LabelCount:     len(labelNames),
		GroupCount:     len(kb.Groups),
		LabelNames:     append([]string(nil), labelNames...),
		TopicLabelHint: sortedKnowledgeBaseGroupNames(kb.Groups),
		Groups:         cloneKnowledgeBaseGroups(kb.Groups),
	}

	return &knowledgeBaseMapArtifacts{
		metadata:  metadata,
		pointData: pointData,
	}, nil
}

func sortedKnowledgeBaseLabelNames(definition config.KnowledgeBaseDefinition) []string {
	names := make([]string, 0, len(definition.Labels))
	for name := range definition.Labels {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func sortedKnowledgeBaseGroupNames(groups map[string][]string) []string {
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func buildKnowledgeBaseRawPoints(
	definition config.KnowledgeBaseDefinition,
	labelNames []string,
	modelType string,
) ([]kbRawPoint, error) {
	rawPoints := make([]kbRawPoint, 0)
	for labelIndex, labelName := range labelNames {
		label := definition.Labels[labelName]
		for _, exemplar := range label.Exemplars {
			text := strings.TrimSpace(exemplar)
			if text == "" {
				continue
			}
			output, err := knowledgeBaseMapEmbeddingFunc(text, modelType, 0)
			if err != nil {
				return nil, fmt.Errorf("embed exemplar for label %q: %w", labelName, err)
			}
			vector := make([]float64, 0, len(output.Embedding))
			for _, value := range output.Embedding {
				vector = append(vector, float64(value))
			}
			rawPoints = append(rawPoints, kbRawPoint{
				Text:       text,
				LabelName:  labelName,
				LabelIndex: labelIndex,
				Vector:     vector,
			})
		}
	}
	return rawPoints, nil
}

func marshalKnowledgeBasePointNDJSON(points []kbRawPoint) ([]byte, error) {
	var buffer bytes.Buffer
	for _, point := range points {
		line, err := json.Marshal(point)
		if err != nil {
			return nil, err
		}
		buffer.Write(line)
		buffer.WriteByte('\n')
	}
	return buffer.Bytes(), nil
}
