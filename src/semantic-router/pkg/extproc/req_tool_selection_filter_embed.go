package extproc

import (
	"fmt"
	"sort"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// toolEmbeddingText builds a phrase for embedding from an OpenAI-format tool (name + optional description).
func toolEmbeddingText(t openai.ChatCompletionToolParam) string {
	name := strings.TrimSpace(t.Function.Name)
	var parts []string
	if name != "" {
		parts = append(parts, name)
	}
	if !param.IsOmitted(t.Function.Description) {
		if d := strings.TrimSpace(t.Function.Description.Value); d != "" {
			parts = append(parts, d)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

func dotProductFloat32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float32
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

type scoredRequestTool struct {
	tool  openai.ChatCompletionToolParam
	score float32
	order int
}

// filterRequestToolsAgainstQuerySemantic scores each OpenAI-format tool definition against queryText
// using embedding dot-products (same pipeline as ToolDatabase retrieval).
func filterRequestToolsAgainstQuerySemantic(
	queryText string,
	requestTools []openai.ChatCompletionToolParam,
	modelType string,
	targetDim int,
	relevanceThreshold float32,
	preserveCount int,
) ([]openai.ChatCompletionToolParam, float32, error) {
	if len(requestTools) == 0 {
		return nil, 0, nil
	}
	trimmedQuery := strings.TrimSpace(queryText)
	if trimmedQuery == "" {
		out := make([]openai.ChatCompletionToolParam, len(requestTools))
		copy(out, requestTools)
		return out, 0, nil
	}
	qOut, err := candle_binding.GetEmbeddingWithModelType(trimmedQuery, modelType, targetDim)
	if err != nil {
		return nil, 0, fmt.Errorf("tool_selection filter: query embedding: %w", err)
	}

	scored, err := scoreToolsBySemanticSimilarity(requestTools, qOut.Embedding, modelType, targetDim)
	if err != nil {
		return nil, 0, err
	}

	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].score == scored[j].score {
			return scored[i].order < scored[j].order
		}
		return scored[i].score > scored[j].score
	})
	maxScore := float32(0)
	if len(scored) > 0 {
		maxScore = scored[0].score
	}

	kept := keepByRelevanceThreshold(scored, relevanceThreshold)

	if preserveCount <= 0 || len(kept) >= preserveCount {
		return kept, maxScore, nil
	}

	kept = preserveTopScoredTools(scored, kept, preserveCount)
	return kept, maxScore, nil
}

func scoreToolsBySemanticSimilarity(
	requestTools []openai.ChatCompletionToolParam,
	queryEmbedding []float32,
	modelType string,
	targetDim int,
) ([]scoredRequestTool, error) {
	scored := make([]scoredRequestTool, 0, len(requestTools))
	for i, tool := range requestTools {
		embeddingText := toolEmbeddingText(tool)
		if embeddingText == "" {
			embeddingText = tool.Function.Name
		}
		tOut, err := candle_binding.GetEmbeddingWithModelType(embeddingText, modelType, targetDim)
		if err != nil {
			return nil, fmt.Errorf("tool_selection filter: embedding tool %q: %w", tool.Function.Name, err)
		}
		scored = append(scored, scoredRequestTool{
			tool:  tool,
			score: dotProductFloat32(queryEmbedding, tOut.Embedding),
			order: i,
		})
	}
	return scored, nil
}

func keepByRelevanceThreshold(scored []scoredRequestTool, relevanceThreshold float32) []openai.ChatCompletionToolParam {
	kept := make([]openai.ChatCompletionToolParam, 0, len(scored))
	for _, s := range scored {
		if s.score >= relevanceThreshold {
			kept = append(kept, s.tool)
		}
	}
	return kept
}

func preserveTopScoredTools(
	scored []scoredRequestTool,
	kept []openai.ChatCompletionToolParam,
	preserveCount int,
) []openai.ChatCompletionToolParam {
	needed := preserveCount - len(kept)
	seen := make(map[string]struct{}, len(kept))
	for _, t := range kept {
		seen[strings.ToLower(strings.TrimSpace(t.Function.Name))] = struct{}{}
	}
	for _, s := range scored {
		if needed == 0 {
			break
		}
		key := strings.ToLower(strings.TrimSpace(s.tool.Function.Name))
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		kept = append(kept, s.tool)
		needed--
	}
	return kept
}
