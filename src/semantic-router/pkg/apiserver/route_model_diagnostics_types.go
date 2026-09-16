//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type ModelDiagnosticTarget struct {
	Recipe  string `json:"recipe"`
	Binding string `json:"binding"`
}
type ModelTextDiagnosticRequest struct {
	ModelDiagnosticTarget
	Text string `json:"text"`
}
type ModelRerankPair struct {
	Query    string `json:"query"`
	Document string `json:"document"`
}
type ModelRerankDiagnosticRequest struct {
	ModelDiagnosticTarget
	Pairs []ModelRerankPair `json:"pairs"`
}
type ModelDiagnosticWindow struct {
	Size    int `json:"size"`
	Overlap int `json:"overlap"`
}
type ModelDiagnosticEmbeddingCapability struct {
	Dimension     int      `json:"dimension"`
	Layer         int      `json:"layer"`
	Pooling       string   `json:"pooling"`
	Normalization string   `json:"normalization"`
	Modalities    []string `json:"modalities"`
}
type ModelDiagnosticBinding struct {
	Recipe     string                              `json:"recipe"`
	Name       string                              `json:"name"`
	Deployment string                              `json:"deployment"`
	Contract   string                              `json:"contract"`
	Adapter    string                              `json:"adapter"`
	Head       string                              `json:"head,omitempty"`
	Artifact   string                              `json:"artifact,omitempty"`
	Revision   string                              `json:"revision,omitempty"`
	Provider   string                              `json:"provider"`
	Device     string                              `json:"device"`
	Precision  string                              `json:"precision"`
	Labels     []string                            `json:"labels,omitempty"`
	MaxTokens  int                                 `json:"max_tokens"`
	Overflow   string                              `json:"overflow"`
	Window     *ModelDiagnosticWindow              `json:"window,omitempty"`
	Embedding  *ModelDiagnosticEmbeddingCapability `json:"embedding,omitempty"`
}
type ModelDiagnosticInventory struct {
	Recipe   string                   `json:"recipe"`
	Bindings []ModelDiagnosticBinding `json:"bindings"`
}
type ModelDiagnosticResponse[T any] struct {
	Binding ModelDiagnosticBinding `json:"binding"`
	Result  T                      `json:"result"`
}
type ModelDiagnosticInputUsage struct {
	OriginalTokens  int  `json:"original_tokens"`
	ProcessedTokens int  `json:"processed_tokens"`
	Truncated       bool `json:"truncated"`
}
type ModelDiagnosticLabelWindow struct {
	Start         int       `json:"start"`
	End           int       `json:"end"`
	Probabilities []float32 `json:"probabilities"`
}
type ModelDiagnosticLabelsResult struct {
	Probabilities []float32                    `json:"probabilities,omitempty"`
	Windows       []ModelDiagnosticLabelWindow `json:"windows,omitempty"`
	ContentTokens int                          `json:"content_tokens,omitempty"`
	Input         *ModelDiagnosticInputUsage   `json:"input,omitempty"`
}
type ModelDiagnosticScoreWindow struct {
	Start  int       `json:"start"`
	End    int       `json:"end"`
	Scores []float32 `json:"scores"`
}
type ModelDiagnosticScoresResult struct {
	Scores       []float32                    `json:"scores,omitempty"`
	Windows      []ModelDiagnosticScoreWindow `json:"windows,omitempty"`
	Input        *ModelDiagnosticInputUsage   `json:"input,omitempty"`
	PolicySHA256 string                       `json:"policy_sha256,omitempty"`
	Thresholds   []float32                    `json:"thresholds,omitempty"`
	WindowRanges [][2]int                     `json:"window_ranges,omitempty"`
}
type ModelDiagnosticEntity struct {
	Type       string   `json:"type"`
	Start      int      `json:"start"`
	End        int      `json:"end"`
	Text       string   `json:"text"`
	Confidence *float32 `json:"confidence,omitempty"`
}
type ModelDiagnosticTokensResult struct {
	Entities        []ModelDiagnosticEntity    `json:"entities"`
	Input           *ModelDiagnosticInputUsage `json:"input,omitempty"`
	ScoresAvailable bool                       `json:"scores_available"`
	ScanIncomplete  bool                       `json:"scan_incomplete"`
	TruncatedAt     *int                       `json:"truncated_at,omitempty"`
	Windows         [][2]int                   `json:"windows,omitempty"`
	ContentTokens   int                        `json:"content_tokens,omitempty"`
}
type ModelDiagnosticEmbeddingResult struct {
	Embedding []float32                  `json:"embedding"`
	Input     *ModelDiagnosticInputUsage `json:"input,omitempty"`
}
type ModelDiagnosticRerankResult struct {
	Scores    []float32                    `json:"scores"`
	ScoreType string                       `json:"score_type"`
	Inputs    []*ModelDiagnosticInputUsage `json:"inputs"`
}

func diagnosticBinding(info binding.PreparedBinding) ModelDiagnosticBinding {
	id, c := info.Identity, info.Capability
	out := ModelDiagnosticBinding{Recipe: id.Recipe, Name: id.Name, Deployment: id.Deployment, Contract: id.Contract, Adapter: id.Adapter, Head: id.Head, Artifact: info.Artifact, Revision: info.Revision, Provider: c.Provider, Device: c.Device, Precision: c.Precision, Labels: c.Labels, MaxTokens: c.Limits.EffectiveTokens(), Overflow: c.Limits.Overflow}
	if c.Window != nil {
		out.Window = &ModelDiagnosticWindow{Size: c.Window.Size, Overlap: c.Window.Overlap}
	}
	if c.Embedding != nil {
		e := c.Embedding
		out.Embedding = &ModelDiagnosticEmbeddingCapability{Dimension: e.Dimension, Layer: e.Layer, Pooling: e.Pooling, Normalization: e.Normalization, Modalities: e.Modalities}
	}
	return out
}

func diagnosticInput(input *tasks.InputUsage) *ModelDiagnosticInputUsage {
	if input == nil {
		return nil
	}
	return &ModelDiagnosticInputUsage{OriginalTokens: input.OriginalTokens, ProcessedTokens: input.ProcessedTokens, Truncated: input.Truncated}
}
