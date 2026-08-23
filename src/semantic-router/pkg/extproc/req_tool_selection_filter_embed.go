package extproc

import (
	"context"
	"fmt"
	"sort"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// toolEmbeddingText builds a phrase for embedding from a neutral function tool.
func toolEmbeddingText(t llmprotocol.Tool) string {
	name := strings.TrimSpace(t.Name)
	var parts []string
	if name != "" {
		parts = append(parts, name)
	}
	if description := strings.TrimSpace(t.Description); description != "" {
		parts = append(parts, description)
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
	tool  llmprotocol.Tool
	score float32
	order int
}

// filterRequestToolsAgainstQuerySemantic scores each OpenAI-format tool definition against queryText
// using embedding dot-products (same pipeline as ToolDatabase retrieval).
func filterRequestToolsAgainstQuerySemantic(
	queryText string,
	requestTools []llmprotocol.Tool,
	modelType string,
	targetDim int,
	provider embedding.Provider,
	relevanceThreshold float32,
	preserveCount int,
) ([]llmprotocol.Tool, float32, error) {
	if len(requestTools) == 0 {
		return nil, 0, nil
	}
	trimmedQuery := strings.TrimSpace(queryText)
	if trimmedQuery == "" {
		out := make([]llmprotocol.Tool, len(requestTools))
		copy(out, requestTools)
		return out, 0, nil
	}
	queryEmbedding, err := embedToolSelectionText(provider, trimmedQuery, modelType, targetDim)
	if err != nil {
		return nil, 0, fmt.Errorf("tool_selection filter: query embedding: %w", err)
	}

	scored, err := scoreToolsBySemanticSimilarity(requestTools, queryEmbedding, modelType, targetDim, provider)
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
	requestTools []llmprotocol.Tool,
	queryEmbedding []float32,
	modelType string,
	targetDim int,
	provider embedding.Provider,
) ([]scoredRequestTool, error) {
	scored := make([]scoredRequestTool, 0, len(requestTools))
	for i, tool := range requestTools {
		embeddingText := toolEmbeddingText(tool)
		if embeddingText == "" {
			embeddingText = tool.Name
		}
		toolEmbedding, err := embedToolSelectionText(provider, embeddingText, modelType, targetDim)
		if err != nil {
			return nil, fmt.Errorf("tool_selection filter: embedding tool %q: %w", tool.Name, err)
		}
		scored = append(scored, scoredRequestTool{
			tool:  tool,
			score: dotProductFloat32(queryEmbedding, toolEmbedding),
			order: i,
		})
	}
	return scored, nil
}

func embedToolSelectionText(provider embedding.Provider, text string, modelType string, targetDim int) ([]float32, error) {
	if provider != nil {
		return provider.Embed(context.Background(), text)
	}
	output, err := candle_binding.GetEmbeddingWithModelType(text, modelType, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

func keepByRelevanceThreshold(scored []scoredRequestTool, relevanceThreshold float32) []llmprotocol.Tool {
	kept := make([]llmprotocol.Tool, 0, len(scored))
	for _, s := range scored {
		if s.score >= relevanceThreshold {
			kept = append(kept, s.tool)
		}
	}
	return kept
}

func preserveTopScoredTools(
	scored []scoredRequestTool,
	kept []llmprotocol.Tool,
	preserveCount int,
) []llmprotocol.Tool {
	needed := preserveCount - len(kept)
	seen := make(map[string]struct{}, len(kept))
	for _, t := range kept {
		seen[strings.ToLower(strings.TrimSpace(t.Name))] = struct{}{}
	}
	for _, s := range scored {
		if needed == 0 {
			break
		}
		key := strings.ToLower(strings.TrimSpace(s.tool.Name))
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		kept = append(kept, s.tool)
		needed--
	}
	return kept
}
