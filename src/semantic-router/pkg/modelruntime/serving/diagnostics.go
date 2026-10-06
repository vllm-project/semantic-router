package serving

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// DiagnosticResult retains the exact prepared identity beside its typed output.
// No endpoint or resource compatibility key is included in this descriptor.
type DiagnosticResult[T any] struct {
	Binding binding.PreparedBinding `json:"binding"`
	Result  T                       `json:"result"`
}

// DiagnosticLabels is a categorical result, single-forward or windowed.
type DiagnosticLabels struct {
	Distribution *tasks.LabelDistribution         `json:"distribution,omitempty"`
	Windows      *tasks.WindowedLabelDistribution `json:"windows,omitempty"`
}

// DiagnosticScores is an independent-label result, with its operating point when one applies.
type DiagnosticScores struct {
	Scores       []float32                  `json:"scores,omitempty"`
	Windows      *tasks.WindowedLabelScores `json:"windows,omitempty"`
	Input        *tasks.InputUsage          `json:"input,omitempty"`
	PolicySHA256 string                     `json:"policy_sha256,omitempty"`
	Thresholds   []float32                  `json:"thresholds,omitempty"`
	WindowRanges [][2]int                   `json:"window_ranges,omitempty"`
}

// DiagnosticTokens is a span result with its window coverage when windowed.
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

// DiagnoseLabels runs a recipe's prepared categorical binding on text.
func (r *Runtime) DiagnoseLabels(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticLabels], error) {
	var response DiagnosticResult[DiagnosticLabels]
	handle, metadata, err := binding.LookupPrepared[string, tasks.LabelDistribution](r.prepared, recipe, name, config.RemoteClassifierContractLabelDistribution)
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticLabels]{Binding: metadata, Result: DiagnosticLabels{Distribution: &result}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution](r.prepared, recipe, name, config.RemoteClassifierContractLabelDistribution)
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

// DiagnoseScores runs a recipe's prepared independent-label binding on text.
func (r *Runtime) DiagnoseScores(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticScores], error) {
	var response DiagnosticResult[DiagnosticScores]
	handle, metadata, err := binding.LookupPrepared[string, tasks.LabelScores](r.prepared, recipe, name, config.RemoteClassifierContractLabelScores)
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{Scores: result.Scores, Input: result.Input}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedLabelScores](r.prepared, recipe, name, config.RemoteClassifierContractLabelScores)
	if err != nil {
		return response, err
	}
	r.mu.Lock()
	policy := r.operatingPoints[windowed]
	r.mu.Unlock()
	if policy != nil {
		result, callErr := policy.Score(ctx, recipe, text)
		return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{
			Scores: result.Scores, Input: result.Input, PolicySHA256: policy.PolicySHA256(),
			Thresholds: policy.Thresholds(), WindowRanges: result.Windows,
		}}, callErr
	}
	input, err := diagnosticWindow(metadata, text)
	if err != nil {
		return response, err
	}
	result, err := windowed.Call(ctx, recipe, input)
	return DiagnosticResult[DiagnosticScores]{Binding: metadata, Result: DiagnosticScores{Windows: &result}}, err
}

// DiagnoseTokens runs a recipe's prepared span binding on text.
func (r *Runtime) DiagnoseTokens(ctx context.Context, recipe, name, text string) (DiagnosticResult[DiagnosticTokens], error) {
	var response DiagnosticResult[DiagnosticTokens]
	handle, metadata, err := binding.LookupPrepared[string, tasks.TokenClassificationResult](r.prepared, recipe, name, config.RemoteClassifierContractTokenSpans)
	if err == nil {
		result, callErr := handle.Call(ctx, recipe, text)
		return DiagnosticResult[DiagnosticTokens]{Binding: metadata, Result: DiagnosticTokens{Spans: result}}, callErr
	}
	if !errors.Is(err, binding.ErrNotPrepared) {
		return response, err
	}
	windowed, metadata, err := binding.LookupPrepared[tasks.TextWindowsRequest, tasks.WindowedTokenClassification](r.prepared, recipe, name, config.RemoteClassifierContractTokenSpans)
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
