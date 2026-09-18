package native

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// DiagnosticResult retains the exact prepared identity beside its typed output.
// No endpoint or resource compatibility key is included in this descriptor.
type DiagnosticResult[T any] struct {
	Binding binding.PreparedBinding `json:"binding"`
	Result  T                       `json:"result"`
}
type DiagnosticLabels struct {
	Distribution *tasks.LabelDistribution         `json:"distribution,omitempty"`
	Windows      *tasks.WindowedLabelDistribution `json:"windows,omitempty"`
}
type DiagnosticScores struct {
	Scores       []float32                  `json:"scores,omitempty"`
	Windows      *tasks.WindowedLabelScores `json:"windows,omitempty"`
	Input        *tasks.InputUsage          `json:"input,omitempty"`
	PolicySHA256 string                     `json:"policy_sha256,omitempty"`
	Thresholds   []float32                  `json:"thresholds,omitempty"`
	WindowRanges [][2]int                   `json:"window_ranges,omitempty"`
}
type DiagnosticTokens struct {
	Spans         tasks.TokenClassificationResult `json:"spans"`
	Windows       [][2]int                        `json:"windows,omitempty"`
	ContentTokens int                             `json:"content_tokens,omitempty"`
}

func diagnosticWindow(metadata binding.PreparedBinding, text string) (tasks.TextWindowsRequest, error) {
	window := metadata.Capability.Window
	if window == nil {
		return tasks.TextWindowsRequest{}, fmt.Errorf("%w: prepared window geometry unavailable", binding.ErrCapability)
	}
	return tasks.TextWindowsRequest{Text: text, Size: window.Size, Overlap: window.Overlap}, nil
}

func (r *Runtime) DiagnoseLabels(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticLabels], error) {
	var response DiagnosticResult[DiagnosticLabels]
	handle, metadata, err := binding.LookupPrepared[string, tasks.LabelDistribution](r.prepared, recipe, name, "label_distribution.v1")
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticLabels]{Binding: metadata, Result: DiagnosticLabels{Distribution: &result}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution](r.prepared, recipe, name, "label_distribution.v1")
	if err != nil {
		return response, err
	}
	input, err := diagnosticWindow(metadata, text)
	if err != nil {
		return response, err
	}
	result, err := windowed.Call(ctx, recipe, input)
	return DiagnosticResult[DiagnosticLabels]{Binding: metadata, Result: DiagnosticLabels{Windows: &result}}, err
}

func (r *Runtime) DiagnoseScores(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticScores], error) {
	var response DiagnosticResult[DiagnosticScores]
	handle, metadata, err := binding.LookupPrepared[string, tasks.LabelScores](r.prepared, recipe, name, "label_scores.v1")
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{Scores: result.Scores, Input: result.Input}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedLabelScores](r.prepared, recipe, name, "label_scores.v1")
	if err != nil {
		return response, err
	}
	r.mu.Lock()
	policy := r.operatingPoints[windowed]
	r.mu.Unlock()
	if policy != nil {
		result, callErr := policy.Score(ctx, recipe, text)
		return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{Scores: result.Scores, Input: result.Input, PolicySHA256: policy.PolicySHA256(), Thresholds: policy.Thresholds(), WindowRanges: result.Windows}}, callErr
	}
	input, err := diagnosticWindow(metadata, text)
	if err != nil {
		return response, err
	}
	result, err := windowed.Call(ctx, recipe, input)
	return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{Windows: &result}}, err
}

func (r *Runtime) DiagnoseTokens(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticTokens], error) {
	var response DiagnosticResult[DiagnosticTokens]
	handle, metadata, err := binding.LookupPrepared[string, tasks.TokenClassificationResult](r.prepared, recipe, name, "token_spans.v1")
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticTokens]{Binding: metadata, Result: DiagnosticTokens{Spans: result}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedTokenClassification](r.prepared, recipe, name, "token_spans.v1")
	if err != nil {
		return response, err
	}
	input, err := diagnosticWindow(metadata, text)
	if err != nil {
		return response, err
	}
	result, err := windowed.Call(ctx, recipe, input)
	return DiagnosticResult[DiagnosticTokens]{Binding: metadata, Result: DiagnosticTokens{Spans: result.Result, Windows: result.Windows, ContentTokens: result.ContentTokens}}, err
}

func (r *Runtime) DiagnoseEmbedding(ctx context.Context, recipe, name, text string) (DiagnosticResult[tasks.EmbeddingResult], error) {
	var response DiagnosticResult[tasks.EmbeddingResult]
	handle, metadata, err := binding.LookupPrepared[embedding.TextRequest, tasks.EmbeddingResult](r.prepared, recipe, name, "embedding.v1")
	if err != nil {
		return response, err
	}
	options := embedding.Options{}
	if metadata.Capability.Embedding != nil {
		options.Dimension = metadata.Capability.Embedding.Dimension
		options.Layer = metadata.Capability.Embedding.Layer
	}
	result, err := handle.Call(ctx, recipe, embedding.TextRequest{Text: text, Options: options})
	return DiagnosticResult[tasks.EmbeddingResult]{Binding: metadata, Result: result}, err
}

func (r *Runtime) DiagnoseRerank(ctx context.Context, recipe, name string, pairs []tasks.QueryDocument) (DiagnosticResult[tasks.RelevanceScores], error) {
	var response DiagnosticResult[tasks.RelevanceScores]
	handle, metadata, err := binding.LookupPrepared[[]tasks.QueryDocument, tasks.RelevanceScores](r.prepared, recipe, name, "relevance_scores.v1")
	if err != nil {
		return response, err
	}
	result, err := handle.Call(ctx, recipe, pairs)
	return DiagnosticResult[tasks.RelevanceScores]{Binding: metadata, Result: result}, err
}
