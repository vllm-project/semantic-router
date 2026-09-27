package pluginruntime

import (
	"context"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// RetrievalPreviewRuntime binds diagnostics to the published Router's plugin
// policies and prepared dependencies. No operation invokes a selected tool.
type RetrievalPreviewRuntime interface {
	PreviewRAG(context.Context, RAGPreviewRequest) (RAGPreviewResponse, error)
	PreviewTools(context.Context, ToolsPreviewRequest) (ToolsPreviewResponse, error)
}

type RAGPreviewRequest struct {
	Binding         Binding                `json:"binding"`
	Mode            ExecutionMode          `json:"mode,omitempty"`
	Format          llmprotocol.WireFormat `json:"format,omitempty"`
	RequestBody     json.RawMessage        `json:"request_body"`
	SuppliedContext string                 `json:"supplied_context,omitempty"`
}

type RAGPreviewResponse struct {
	Guarantees
	Binding         Binding         `json:"binding"`
	Enabled         bool            `json:"enabled"`
	Stage           string          `json:"stage"`
	Backend         string          `json:"backend"`
	Context         string          `json:"context"`
	SimilarityScore float32         `json:"similarity_score"`
	RequestBody     json.RawMessage `json:"request_body"`
}

type ToolsPreviewRequest struct {
	Binding     Binding                `json:"binding"`
	Mode        ExecutionMode          `json:"mode,omitempty"`
	Format      llmprotocol.WireFormat `json:"format,omitempty"`
	RequestBody json.RawMessage        `json:"request_body"`
	// Plugin is selected by the registered operation, never by an input field.
	Plugin string `json:"-"`
}

type ToolsPreviewResponse struct {
	Guarantees
	Binding       Binding         `json:"binding"`
	Plugin        string          `json:"plugin"`
	Enabled       bool            `json:"enabled"`
	SelectedTools []string        `json:"selected_tools"`
	RequestBody   json.RawMessage `json:"request_body"`
}
