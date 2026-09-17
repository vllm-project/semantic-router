//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func apiPluginRetrievalRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/rag/preview", Method: "POST", Description: "Preview configured RAG context injection using supplied_context; mode=probe retrieves from the configured backend without the RAG result cache or generation"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleRAGPluginPreview,
			strictJSONBodyFor[pluginruntime.RAGPreviewRequest](), jsonResponse[pluginruntime.RAGPreviewResponse](http.StatusOK, "Recipe-bound RAG preview and explicit effects"), errorResponses(400, 404, 409, 429, 500, 503, 504),
			pluginOperationFor("rag", "preview"), pluginOperationFor("rag", "probe")),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/tools/preview", Method: "POST", Description: "Preview the configured tools policy; mode=probe permits semantic tool retrieval but never executes tools"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleToolsPluginPreview,
			strictJSONBodyFor[pluginruntime.ToolsPreviewRequest](), jsonResponse[pluginruntime.ToolsPreviewResponse](http.StatusOK, "Recipe-bound tool policy result and explicit effects"), errorResponses(400, 404, 409, 429, 500, 503, 504),
			pluginOperationFor("tools", "preview"), pluginOperationFor("tools", "probe")),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/tool_selection/preview", Method: "POST", Description: "Preview the configured tool_selection policy; mode=probe permits configured retrieval and embedding calls but never executes tools"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleToolSelectionPluginPreview,
			strictJSONBodyFor[pluginruntime.ToolsPreviewRequest](), jsonResponse[pluginruntime.ToolsPreviewResponse](http.StatusOK, "Recipe-bound tool selection result and explicit effects"), errorResponses(400, 404, 409, 429, 500, 503, 504),
			pluginOperationFor("tool_selection", "preview"), pluginOperationFor("tool_selection", "probe")),
	}
}

func (s *ClassificationAPIServer) handleRAGPluginPreview(w http.ResponseWriter, r *http.Request) {
	var input pluginruntime.RAGPreviewRequest
	if err := s.parseStrictJSONRequest(r, &input); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	runPluginPreview(s, w, r, func(ctx context.Context, capabilities pluginruntime.Capabilities) (pluginruntime.RAGPreviewResponse, error) {
		if capabilities.Retrieval == nil {
			return pluginruntime.RAGPreviewResponse{}, pluginruntime.ErrUnavailable
		}
		return capabilities.Retrieval.PreviewRAG(ctx, input)
	})
}

func (s *ClassificationAPIServer) handleToolsPluginPreview(w http.ResponseWriter, r *http.Request) {
	s.handleToolPreview(w, r, "tools")
}

func (s *ClassificationAPIServer) handleToolSelectionPluginPreview(w http.ResponseWriter, r *http.Request) {
	s.handleToolPreview(w, r, "tool_selection")
}

func (s *ClassificationAPIServer) handleToolPreview(w http.ResponseWriter, r *http.Request, plugin string) {
	var input pluginruntime.ToolsPreviewRequest
	if err := s.parseStrictJSONRequest(r, &input); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	input.Plugin = plugin
	runPluginPreview(s, w, r, func(ctx context.Context, capabilities pluginruntime.Capabilities) (pluginruntime.ToolsPreviewResponse, error) {
		if capabilities.Retrieval == nil {
			return pluginruntime.ToolsPreviewResponse{}, pluginruntime.ErrUnavailable
		}
		return capabilities.Retrieval.PreviewTools(ctx, input)
	})
}
