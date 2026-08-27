package extproc

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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
//
// The query and every tool are embedded in a single pass through emb, which
// serves unchanged tool definitions from its memo and batches the remaining
// misses. Scoring, ordering and threshold semantics are unaffected by whether a
// vector came from the memo or was just computed.
func filterRequestToolsAgainstQuerySemantic(
	ctx context.Context,
	queryText string,
	requestTools []openai.ChatCompletionToolParam,
	emb *cachedToolEmbedder,
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
	if emb == nil {
		return nil, 0, fmt.Errorf("tool_selection filter: embedder is not initialized")
	}

	toolTexts := make([]string, len(requestTools))
	for i, tool := range requestTools {
		text := toolEmbeddingText(tool)
		if text == "" {
			text = tool.Function.Name
		}
		toolTexts[i] = text
	}

	queryEmbedding, toolEmbeddings, err := emb.embedQueryAndTools(ctx, trimmedQuery, toolTexts)
	if err != nil {
		return nil, 0, err
	}

	scored := make([]scoredRequestTool, 0, len(requestTools))
	for i, tool := range requestTools {
		scored = append(scored, scoredRequestTool{
			tool:  tool,
			score: dotProductFloat32(queryEmbedding, toolEmbeddings[i]),
			order: i,
		})
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
