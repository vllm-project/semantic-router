//go:build !windows

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
	Dimensions    []int    `json:"dimensions,omitempty"`
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
type ModelDiagnosticEmbeddingResult struct {
	Embedding []float32                  `json:"embedding"`
	Input     *ModelDiagnosticInputUsage `json:"input,omitempty"`
}

func diagnosticBinding(info binding.PreparedBinding) ModelDiagnosticBinding {
	id, c := info.Identity, info.Capability
	out := ModelDiagnosticBinding{Recipe: id.Recipe, Name: id.Name, Deployment: id.Deployment, Contract: id.Contract, Adapter: id.Adapter, Head: id.Head, Artifact: info.Artifact, Revision: info.Revision, Provider: c.Provider, Device: c.Device, Precision: c.Precision, Labels: c.Labels, MaxTokens: c.Limits.EffectiveTokens(), Overflow: c.Limits.Overflow}
	if c.Window != nil {
		out.Window = &ModelDiagnosticWindow{Size: c.Window.Size, Overlap: c.Window.Overlap}
	}
	if c.Embedding != nil {
		e := c.Embedding
		out.Embedding = &ModelDiagnosticEmbeddingCapability{Dimension: e.Dimension, Layer: e.Layer, Pooling: e.Pooling, Normalization: e.Normalization, Modalities: e.Modalities, Dimensions: e.AvailableDimensions}
	}
	return out
}

func diagnosticInput(input *tasks.InputUsage) *ModelDiagnosticInputUsage {
	if input == nil {
		return nil
	}
	return &ModelDiagnosticInputUsage{OriginalTokens: input.OriginalTokens, ProcessedTokens: input.ProcessedTokens, Truncated: input.Truncated}
}
