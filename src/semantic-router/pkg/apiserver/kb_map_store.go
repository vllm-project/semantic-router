//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var knowledgeBaseMapEmbeddingFunc = candle_binding.GetEmbeddingWithModelType

const (
	kbMapGridSize       = 64
	kbMapProjectionName = "pca_2d"
)

type knowledgeBaseMapMetadataResponse struct {
	Name           string   `json:"name"`
	Description    string   `json:"description,omitempty"`
	Projection     string   `json:"projection"`
	ModelType      string   `json:"model_type"`
	PointCount     int      `json:"point_count"`
	LabelCount     int      `json:"label_count"`
	GroupCount     int      `json:"group_count"`
	LabelNames     []string `json:"label_names"`
	TopicLabelHint []string `json:"topic_label_hint,omitempty"`
}

type knowledgeBaseMapArtifacts struct {
	signature string
	metadata  knowledgeBaseMapMetadataResponse
	pointData []byte
	gridData  []byte
	topicData []byte
}

type knowledgeBaseMapCache struct {
	mu    sync.RWMutex
	items map[string]*knowledgeBaseMapArtifacts
}

type knowledgeBaseMapGridResponse struct {
	Grid                 [][]float64            `json:"grid"`
	XRange               [2]float64             `json:"xRange"`
	YRange               [2]float64             `json:"yRange"`
	Padded               bool                   `json:"padded"`
	SampleSize           int                    `json:"sampleSize"`
	TotalPointSize       int                    `json:"totalPointSize"`
	EmbeddingName        string                 `json:"embeddingName"`
	GroupGrids           map[string][][]float64 `json:"groupGrids,omitempty"`
	GroupTotalPointSizes map[string]int         `json:"groupTotalPointSizes,omitempty"`
	GroupNames           []string               `json:"groupNames,omitempty"`
}

type knowledgeBaseMapTopicResponse struct {
	Extent [2][2]float64          `json:"extent"`
	Data   map[string][][]any `json:"data"`
}

type kbProjectedPoint struct {
	X         float64
	Y         float64
	Text      string
	LabelName string
	GroupID   int
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
		KnowledgeBase config.KnowledgeBaseConfig      `json:"kb"`
		Definition    config.KnowledgeBaseDefinition  `json:"definition"`
		ModelType     string                          `json:"model_type"`
		Projection    string                          `json:"projection"`
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

	points, err := buildKnowledgeBaseProjectedPoints(definition, labelNames, modelType)
	if err != nil {
		return nil, err
	}
	if len(points) == 0 {
		return nil, fmt.Errorf("knowledge base %q produced no map points", kb.Name)
	}

	xRange, yRange := knowledgeBaseMapRanges(points)
	groupGrids, groupTotals, groupNames, grid := buildKnowledgeBaseDensityGrids(points, labelNames, xRange, yRange)
	topicData := buildKnowledgeBaseTopicResponse(points, labelNames, kb.Groups, xRange, yRange)

	pointData, err := marshalKnowledgeBasePointNDJSON(points)
	if err != nil {
		return nil, err
	}
	gridData, err := json.Marshal(knowledgeBaseMapGridResponse{
		Grid:                 grid,
		XRange:               xRange,
		YRange:               yRange,
		Padded:               false,
		SampleSize:           len(points),
		TotalPointSize:       len(points),
		EmbeddingName:        fmt.Sprintf("%s (%s)", kb.Name, modelType),
		GroupGrids:           groupGrids,
		GroupTotalPointSizes: groupTotals,
		GroupNames:           groupNames,
	})
	if err != nil {
		return nil, err
	}
	topicBytes, err := json.Marshal(topicData)
	if err != nil {
		return nil, err
	}

	metadata := knowledgeBaseMapMetadataResponse{
		Name:           kb.Name,
		Description:    strings.TrimSpace(definition.Description),
		Projection:     kbMapProjectionName,
		ModelType:      modelType,
		PointCount:     len(points),
		LabelCount:     len(labelNames),
		GroupCount:     len(kb.Groups),
		LabelNames:     append([]string(nil), labelNames...),
		TopicLabelHint: sortedKnowledgeBaseGroupNames(kb.Groups),
	}

	return &knowledgeBaseMapArtifacts{
		metadata:  metadata,
		pointData: pointData,
		gridData:  append(gridData, '\n'),
		topicData: append(topicBytes, '\n'),
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

func buildKnowledgeBaseProjectedPoints(
	definition config.KnowledgeBaseDefinition,
	labelNames []string,
	modelType string,
) ([]kbProjectedPoint, error) {
	vectors := make([][]float64, 0)
	rawPoints := make([]kbProjectedPoint, 0)
	for groupID, labelName := range labelNames {
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
			vectors = append(vectors, vector)
			rawPoints = append(rawPoints, kbProjectedPoint{
				Text:      text,
				LabelName: labelName,
				GroupID:   groupID,
			})
		}
	}
	projected := projectKnowledgeBaseEmbeddings(vectors)
	for i := range rawPoints {
		rawPoints[i].X = projected[i][0]
		rawPoints[i].Y = projected[i][1]
	}
	return rawPoints, nil
}

func projectKnowledgeBaseEmbeddings(vectors [][]float64) [][2]float64 {
	projected := make([][2]float64, len(vectors))
	if len(vectors) == 0 {
		return projected
	}
	if len(vectors) == 1 || len(vectors[0]) == 0 {
		return projected
	}

	centered := centerEmbeddingVectors(vectors)
	components := principalComponents(centered, 2)
	if len(components) == 0 {
		for i, vector := range centered {
			projected[i] = fallbackProjectedPoint(vector)
		}
		return projected
	}

	for i, vector := range centered {
		projected[i][0] = dotProduct(vector, components[0])
		if len(components) > 1 {
			projected[i][1] = dotProduct(vector, components[1])
		} else {
			projected[i][1] = fallbackProjectedPoint(vector)[1]
		}
	}
	return projected
}

func centerEmbeddingVectors(vectors [][]float64) [][]float64 {
	count := len(vectors)
	dim := len(vectors[0])
	centered := make([][]float64, count)
	means := make([]float64, dim)
	for _, vector := range vectors {
		for dimIndex, value := range vector {
			means[dimIndex] += value
		}
	}
	for dimIndex := range means {
		means[dimIndex] /= float64(count)
	}
	for index, vector := range vectors {
		row := make([]float64, dim)
		for dimIndex, value := range vector {
			row[dimIndex] = value - means[dimIndex]
		}
		centered[index] = row
	}
	return centered
}

func principalComponents(centered [][]float64, count int) [][]float64 {
	if len(centered) == 0 || len(centered[0]) == 0 || count <= 0 {
		return nil
	}
	dim := len(centered[0])
	components := make([][]float64, 0, count)
	for componentIndex := 0; componentIndex < count; componentIndex++ {
		vector := initialPrincipalComponentVector(dim, componentIndex)
		if normalizeInPlace(vector) == 0 {
			break
		}
		for range 24 {
			next := covarianceMultiply(centered, vector)
			for _, prior := range components {
				projectOut(next, prior)
			}
			if normalizeInPlace(next) == 0 {
				break
			}
			vector = next
		}
		for _, prior := range components {
			projectOut(vector, prior)
		}
		if normalizeInPlace(vector) == 0 {
			continue
		}
		components = append(components, vector)
	}
	return components
}

func initialPrincipalComponentVector(dim int, seed int) []float64 {
	vector := make([]float64, dim)
	for index := range vector {
		phase := float64(index + 1 + seed)
		vector[index] = math.Sin(phase*0.73) + math.Cos(phase*1.19)
	}
	return vector
}

func covarianceMultiply(centered [][]float64, vector []float64) []float64 {
	result := make([]float64, len(vector))
	if len(centered) == 0 {
		return result
	}
	for _, row := range centered {
		scale := dotProduct(row, vector)
		if scale == 0 {
			continue
		}
		for index, value := range row {
			result[index] += value * scale
		}
	}
	if len(centered) > 1 {
		scale := 1.0 / float64(len(centered)-1)
		for index := range result {
			result[index] *= scale
		}
	}
	return result
}

func projectOut(target []float64, basis []float64) {
	scale := dotProduct(target, basis)
	for index := range target {
		target[index] -= scale * basis[index]
	}
}

func normalizeInPlace(vector []float64) float64 {
	norm := math.Sqrt(dotProduct(vector, vector))
	if norm == 0 {
		return 0
	}
	for index := range vector {
		vector[index] /= norm
	}
	return norm
}

func dotProduct(left []float64, right []float64) float64 {
	sum := 0.0
	for index, value := range left {
		sum += value * right[index]
	}
	return sum
}

func fallbackProjectedPoint(vector []float64) [2]float64 {
	point := [2]float64{}
	if len(vector) > 0 {
		point[0] = vector[0]
	}
	if len(vector) > 1 {
		point[1] = vector[1]
	}
	return point
}

func knowledgeBaseMapRanges(points []kbProjectedPoint) ([2]float64, [2]float64) {
	minX, maxX := points[0].X, points[0].X
	minY, maxY := points[0].Y, points[0].Y
	for _, point := range points[1:] {
		minX = math.Min(minX, point.X)
		maxX = math.Max(maxX, point.X)
		minY = math.Min(minY, point.Y)
		maxY = math.Max(maxY, point.Y)
	}
	if minX == maxX {
		minX -= 1
		maxX += 1
	}
	if minY == maxY {
		minY -= 1
		maxY += 1
	}
	return [2]float64{minX, maxX}, [2]float64{minY, maxY}
}

func buildKnowledgeBaseDensityGrids(
	points []kbProjectedPoint,
	groupNames []string,
	xRange [2]float64,
	yRange [2]float64,
) (map[string][][]float64, map[string]int, []string, [][]float64) {
	grid := newGrid(kbMapGridSize)
	groupGrids := make(map[string][][]float64, len(groupNames))
	groupTotals := make(map[string]int, len(groupNames))
	for _, name := range groupNames {
		groupGrids[name] = newGrid(kbMapGridSize)
	}

	xSpan := xRange[1] - xRange[0]
	ySpan := yRange[1] - yRange[0]
	for _, point := range points {
		xIndex := clampGridIndex(int(math.Round(((point.X - xRange[0]) / xSpan) * float64(kbMapGridSize-1))))
		yIndex := clampGridIndex(int(math.Round(((point.Y - yRange[0]) / ySpan) * float64(kbMapGridSize-1))))
		grid[yIndex][xIndex]++
		groupName := groupNames[point.GroupID]
		groupGrids[groupName][yIndex][xIndex]++
		groupTotals[groupName]++
	}

	grid = smoothGrid(grid, 2)
	for name, groupGrid := range groupGrids {
		groupGrids[name] = smoothGrid(groupGrid, 1)
	}
	return groupGrids, groupTotals, append([]string(nil), groupNames...), grid
}

func newGrid(size int) [][]float64 {
	grid := make([][]float64, size)
	for rowIndex := range grid {
		grid[rowIndex] = make([]float64, size)
	}
	return grid
}

func clampGridIndex(index int) int {
	if index < 0 {
		return 0
	}
	if index >= kbMapGridSize {
		return kbMapGridSize - 1
	}
	return index
}

func smoothGrid(grid [][]float64, passes int) [][]float64 {
	current := grid
	for range passes {
		next := newGrid(len(current))
		for rowIndex := range current {
			for colIndex := range current[rowIndex] {
				total := 0.0
				weight := 0.0
				for rowOffset := -1; rowOffset <= 1; rowOffset++ {
					for colOffset := -1; colOffset <= 1; colOffset++ {
						nextRow := rowIndex + rowOffset
						nextCol := colIndex + colOffset
						if nextRow < 0 || nextRow >= len(current) || nextCol < 0 || nextCol >= len(current[rowIndex]) {
							continue
						}
						cellWeight := 1.0
						if rowOffset == 0 {
							cellWeight++
						}
						if colOffset == 0 {
							cellWeight++
						}
						total += current[nextRow][nextCol] * cellWeight
						weight += cellWeight
					}
				}
				if weight > 0 {
					next[rowIndex][colIndex] = total / weight
				}
			}
		}
		current = next
	}
	return current
}

func buildKnowledgeBaseTopicResponse(
	points []kbProjectedPoint,
	labelNames []string,
	groups map[string][]string,
	xRange [2]float64,
	yRange [2]float64,
) knowledgeBaseMapTopicResponse {
	labelPoints := make(map[string][]kbProjectedPoint, len(labelNames))
	for _, point := range points {
		labelPoints[point.LabelName] = append(labelPoints[point.LabelName], point)
	}

	data := make(map[string][][]any)
	levelOne := make([][]any, 0, len(groups))
	for _, groupName := range sortedKnowledgeBaseGroupNames(groups) {
		memberLabels := groups[groupName]
		memberPoints := make([]kbProjectedPoint, 0)
		for _, labelName := range memberLabels {
			memberPoints = append(memberPoints, labelPoints[labelName]...)
		}
		if len(memberPoints) == 0 {
			continue
		}
		x, y := centroidForKnowledgeBasePoints(memberPoints)
		levelOne = append(levelOne, []any{x, y, groupName})
	}
	if len(levelOne) > 0 {
		data["1"] = levelOne
	}

	levelTwo := make([][]any, 0, len(labelNames))
	for _, labelName := range labelNames {
		memberPoints := labelPoints[labelName]
		if len(memberPoints) == 0 {
			continue
		}
		x, y := centroidForKnowledgeBasePoints(memberPoints)
		levelTwo = append(levelTwo, []any{x, y, labelName})
	}
	if len(levelTwo) > 0 {
		data["2"] = levelTwo
	}

	if len(data) == 0 {
		data["1"] = [][]any{{0.0, 0.0, "knowledge"}}
	}

	return knowledgeBaseMapTopicResponse{
		Extent: [2][2]float64{
			{xRange[0], yRange[0]},
			{xRange[1], yRange[1]},
		},
		Data: data,
	}
}

func centroidForKnowledgeBasePoints(points []kbProjectedPoint) (float64, float64) {
	sumX := 0.0
	sumY := 0.0
	for _, point := range points {
		sumX += point.X
		sumY += point.Y
	}
	return sumX / float64(len(points)), sumY / float64(len(points))
}

func marshalKnowledgeBasePointNDJSON(points []kbProjectedPoint) ([]byte, error) {
	var buffer bytes.Buffer
	for _, point := range points {
		record := []any{point.X, point.Y, point.Text, "", point.GroupID}
		line, err := json.Marshal(record)
		if err != nil {
			return nil, err
		}
		buffer.Write(line)
		buffer.WriteByte('\n')
	}
	return buffer.Bytes(), nil
}
