package extproc

import (
	"context"
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

var _ pluginruntime.RetrievalPreviewRuntime = (*OpenAIRouter)(nil)

func (r *OpenAIRouter) PreviewRAG(ctx context.Context, input pluginruntime.RAGPreviewRequest) (pluginruntime.RAGPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, err
	}
	requestContext, err := r.pluginPreviewContext(ctx, input.Binding, "rag")
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, err
	}
	format := previewRequestFormat(input.Format)
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(format, input.RequestBody)
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, fmt.Errorf("invalid request_body: %w", err)
	}
	requestContext.SemanticRequest = &request
	requestContext.RequestID = "plugin-preview"
	requestContext.UserContent = extractSemanticRequestSignals(&request).UserContent
	policy := requestContext.VSRSelectedDecision.GetRAGConfig()
	if policy == nil {
		return pluginruntime.RAGPreviewResponse{}, pluginruntime.ErrInvalidBinding
	}
	result := pluginruntime.RAGPreviewResponse{
		Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding,
		Enabled: policy.Enabled, Stage: "injection", Backend: policy.Backend,
	}
	contextText := input.SuppliedContext
	if policy.Enabled && mode == pluginruntime.ModeProbe {
		if input.SuppliedContext != "" {
			return result, fmt.Errorf("supplied_context is only accepted in preview mode")
		}
		result.Stage = "retrieval_and_injection"
		result.BackendCalls = true
		if policy.Backend == "openai" {
			backend, configErr := policy.OpenAIBackendConfig()
			if configErr != nil {
				return result, configErr
			}
			if backend.WorkflowMode == "tool_based" {
				result.BackendCalls = false
			}
		}
		// Intentionally call the shared backend implementation directly: the normal
		// wrapper reads/writes the RAG result cache and changes request metrics.
		contextText, err = r.retrieveContextFromBackend(ctx, requestContext, policy)
		if err != nil {
			return result, fmt.Errorf("configured RAG retrieval failed: %w", err)
		}
	}
	if policy.Enabled {
		if err = r.injectRAGContext(requestContext, contextText, policy); err != nil {
			return result, err
		}
	}
	encoded, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		return result, err
	}
	result.Context = requestContext.RAGRetrievedContext
	result.SimilarityScore = requestContext.RAGSimilarityScore
	result.RequestBody = encoded.Body
	return result, nil
}

func (r *OpenAIRouter) PreviewTools(ctx context.Context, input pluginruntime.ToolsPreviewRequest) (pluginruntime.ToolsPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, err
	}
	if input.Plugin != config.DecisionPluginTools && input.Plugin != config.DecisionPluginToolSelection {
		return pluginruntime.ToolsPreviewResponse{}, pluginruntime.ErrInvalidBinding
	}
	requestContext, err := r.pluginPreviewContext(ctx, input.Binding, input.Plugin)
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, err
	}
	format := previewRequestFormat(input.Format)
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(format, input.RequestBody)
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, fmt.Errorf("invalid request_body: %w", err)
	}
	requestContext.SemanticRequest = &request
	toolsPolicy := requestContext.VSRSelectedDecision.GetToolsConfig()
	selectionPolicy := requestContext.VSRSelectedDecision.GetToolSelectionConfig()
	enabled := toolsPolicy != nil && toolsPolicy.Enabled
	if input.Plugin == config.DecisionPluginToolSelection {
		enabled = selectionPolicy != nil && selectionPolicy.Enabled
	}
	result := pluginruntime.ToolsPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding, Plugin: input.Plugin, Enabled: enabled, SelectedTools: []string{}}
	if enabled {
		// Apply the same early static policy as dispatch, on private decoded state.
		effectivePolicy := toolsPolicy
		if effectivePolicy == nil || !effectivePolicy.Enabled {
			effectivePolicy = &config.ToolsPluginConfig{Enabled: true, Mode: config.ToolsPluginModePassthrough}
		}
		var response *ext_proc.ProcessingResponse
		canSelect, policyErr := r.handleEarlyToolModes(&request, &response, requestContext, effectivePolicy)
		if policyErr != nil {
			return result, policyErr
		}
		signals := extractSemanticRequestSignals(&request)
		query, _, hasQuery := buildToolClassificationText(signals.UserContent, signals.NonUserMessages)
		selectionEnabled := (selectionPolicy != nil && selectionPolicy.Enabled) || (toolsPolicy != nil && toolsPolicy.SelectionEnabled())
		if selectionPolicy != nil && selectionPolicy.Enabled && selectionPolicy.Mode == config.ToolSelectionModeFilter && len(request.Tools) == 0 {
			selectionEnabled = false
		}
		if canSelect && selectionEnabled && hasQuery {
			if mode != pluginruntime.ModeProbe {
				return result, pluginruntime.ErrProbeRequired
			}
			if selectionPolicy != nil && selectionPolicy.Enabled {
				if selectionPolicy.Mode == config.ToolSelectionModeFilter {
					if len(request.Tools) == 0 {
						selectionEnabled = false
					} else if r.toolEmbedder == nil {
						return result, pluginruntime.ErrUnavailable
					}
				} else {
					database, _, databaseErr := r.toolDatabaseForSelectionPlugin(selectionPolicy)
					if databaseErr != nil {
						return result, databaseErr
					}
					if database == nil || !database.IsEnabled() {
						return result, pluginruntime.ErrUnavailable
					}
				}
			} else if r.ToolsDatabase == nil || !r.ToolsDatabase.IsEnabled() {
				return result, pluginruntime.ErrUnavailable
			}
			result.BackendCalls = selectionEnabled
			// Existing selection implementations may memoize embeddings and report
			// retrieval metrics, but never execute tools or record conversation state.
			if err = r.handleToolSelection(&request, query, signals.NonUserMessages, &response, requestContext); err != nil {
				return result, err
			}
		}
		clearSemanticToolChoiceWhenNoTools(&request)
	}
	encoded, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		return result, err
	}
	result.RequestBody = encoded.Body
	for _, tool := range request.Tools {
		result.SelectedTools = append(result.SelectedTools, tool.Name)
	}
	return result, nil
}

func previewRequestFormat(format llmprotocol.WireFormat) llmprotocol.WireFormat {
	if format == "" {
		return llmprotocol.OpenAIChatV1
	}
	return format
}
