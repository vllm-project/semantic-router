package extproc

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

var _ pluginruntime.RetrievalPreviewRuntime = (*OpenAIRouter)(nil)

// PreviewRAG previews the rag source of a decision's context plugin: in
// probe mode it recalls and judges the documents, without the result cache,
// and either way injects the context as the plugin does.
func (r *OpenAIRouter) PreviewRAG(ctx context.Context, input pluginruntime.RAGPreviewRequest) (pluginruntime.RAGPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, err
	}
	requestContext, err := r.pluginPreviewContext(ctx, input.Binding, config.DecisionPluginContext)
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, err
	}
	format := previewRequestFormat(input.Format)
	engine, err := r.protocolEngine()
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, err
	}
	request, envelope, _, err := engine.DecodeRequestForMutation(format, input.RequestBody)
	if err != nil {
		return pluginruntime.RAGPreviewResponse{}, fmt.Errorf("invalid request_body: %w", err)
	}
	requestContext.SemanticRequest = &request
	requestContext.RequestID = "plugin-preview"
	snapshot := extractSemanticRequestSignalsWith(&request, r.routingInputSelector())
	requestContext.UserContent, requestContext.RoutingInput = snapshot.UserContent, snapshot.RoutingInput
	plugin := requestContext.VSRSelectedDecision.GetContextConfig()
	source := plugin.Source(config.ContextSourceRAG)
	if source == nil {
		return pluginruntime.RAGPreviewResponse{}, pluginruntime.ErrInvalidBinding
	}
	result := pluginruntime.RAGPreviewResponse{
		Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding,
		Enabled: plugin.Enabled, Stage: "injection", Backend: source.Backend,
	}
	contextText := input.SuppliedContext
	if plugin.Enabled && mode == pluginruntime.ModeProbe {
		if input.SuppliedContext != "" {
			return result, fmt.Errorf("supplied_context is only accepted in preview mode")
		}
		result.Stage = "retrieval_and_injection"
		result.BackendCalls = true
		matched, matchErr := r.matcher().Match(ctx, config.MatchRAG, source.Match, match.Input{
			Request: requestContext.UserContent, Index: ragIndex{router: r, request: requestContext, source: source},
		})
		if matchErr != nil {
			return result, fmt.Errorf("configured RAG retrieval failed: %w", matchErr)
		}
		texts := make([]string, len(matched.Kept))
		for index, document := range matched.Kept {
			texts[index] = document.Text
			requestContext.RAGSimilarityScore = max(requestContext.RAGSimilarityScore, float32(document.Similarity))
		}
		contextText = strings.Join(texts, "\n\n---\n\n")
	}
	if plugin.Enabled && contextText != "" {
		contextText = truncateRAGContext(contextText, source)
		if err = r.injectContext(requestContext, plugin, config.ContextSourceRAG, contextText); err != nil {
			return result, err
		}
		requestContext.RAGRetrievedContext = contextText
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
	if input.Plugin != config.DecisionPluginTools {
		return pluginruntime.ToolsPreviewResponse{}, pluginruntime.ErrInvalidBinding
	}
	requestContext, err := r.pluginPreviewContext(ctx, input.Binding, input.Plugin)
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, err
	}
	format := previewRequestFormat(input.Format)
	engine, err := r.protocolEngine()
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, err
	}
	request, envelope, _, err := engine.DecodeRequestForMutation(format, input.RequestBody)
	if err != nil {
		return pluginruntime.ToolsPreviewResponse{}, fmt.Errorf("invalid request_body: %w", err)
	}
	requestContext.SemanticRequest = &request
	plugin := requestContext.VSRSelectedDecision.GetToolsConfig()
	result := pluginruntime.ToolsPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding, Plugin: input.Plugin, Enabled: plugin != nil && plugin.Enabled, SelectedTools: []string{}}
	// The gate and the policy run as dispatch runs them, on private decoded
	// state; the selection asks the decision model and never runs a tool.
	if result.Enabled && !r.applyTrustedFactsGate(&request, requestContext, plugin, trustedFactsRequestStage) &&
		applyToolsPolicy(&request, plugin.Policy) && plugin.Select != nil && leavesToolChoiceToModel(&request) {
		if mode != pluginruntime.ModeProbe {
			return result, pluginruntime.ErrProbeRequired
		}
		result.BackendCalls = true
		if failure := r.selectTools(&request, plugin, requestContext); failure != nil {
			return result, fmt.Errorf("%w: %s", pluginruntime.ErrUnavailable, failure.message)
		}
	}
	clearSemanticToolChoiceWhenNoTools(&request)
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
