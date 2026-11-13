package extproc

import (
	"net/url"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const videosCollectionPath = "/v1/videos"

// splitVideoID splits at the first colon so a provider id may itself contain
// colons. The route half is the model route the create call was dispatched on.
func splitVideoID(id string) (route, providerID string, ok bool) {
	route, providerID, ok = strings.Cut(id, ":")
	return route, providerID, ok && route != "" && providerID != ""
}

// stampVideoID keeps an id already stamped for this route, while a provider id
// containing its own colon still receives the route prefix.
func stampVideoID(route, providerID string) string {
	if existingRoute, _, ok := splitVideoID(providerID); ok && existingRoute == route {
		return providerID
	}
	return route + ":" + providerID
}

// parseVideosObjectPath returns the encoded client id and whether the path is
// the content endpoint. It accepts exactly the status/delete and content forms.
func parseVideosObjectPath(path string) (id string, content, ok bool) {
	path = normalizeRequestPath(path)
	rest, found := strings.CutPrefix(path, videosCollectionPath+"/")
	if !found || rest == "" {
		return "", false, false
	}
	if strings.HasSuffix(rest, "/content") {
		id = strings.TrimSuffix(rest, "/content")
		content = true
	} else {
		id = rest
	}
	if id == "" || strings.Contains(id, "/") {
		return "", false, false
	}
	decoded, err := url.PathUnescape(id)
	if err != nil || decoded == "" {
		return "", false, false
	}
	return decoded, content, true
}

func isVideosPath(path string) bool {
	path = normalizeRequestPath(path)
	if path == videosCollectionPath {
		return true
	}
	_, _, ok := parseVideosObjectPath(path)
	return ok
}

func (r *OpenAIRouter) handleVideosRequestHeaders(
	method string,
	path string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	id, content, ok := parseVideosObjectPath(path)
	if !ok {
		return nil, nil
	}
	allowed := method == "GET" || (!content && method == "DELETE")
	if !allowed {
		return nil, nil
	}
	route, providerID, ok := splitVideoID(id)
	if !ok {
		return r.videoNotFound(ctx), nil
	}
	capabilities, known := r.declaredModelCapabilities(route)
	if !known || !capabilities.Supports(llmprotocol.CapabilityVideoGeneration) {
		return r.videoNotFound(ctx), nil
	}

	// The object path never reaches the inference entrypoints, so the header
	// stage is the only place that can declare the format both sides speak.
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIVideosV1, llmprotocol.OpenAIVideosV1
	ctx.RequestModel, ctx.videosRoute, ctx.videosContent = route, route, content
	mutation := r.buildRequestHeadersMutation(ctx, path)
	appendRoutingHeaders(&mutation.SetHeaders, route)
	if content {
		mutation.SetHeaders = append(mutation.SetHeaders, overwriteRequestHeader(":path", videosCollectionPath+"/"+url.PathEscape(providerID)+"/content"))
	} else {
		mutation.SetHeaders = append(mutation.SetHeaders, overwriteRequestHeader(":path", videosCollectionPath+"/"+url.PathEscape(providerID)))
	}
	response := newContinueRequestHeadersResponse(mutation)
	response.GetRequestHeaders().GetResponse().ClearRouteCache = true
	return response, nil
}

func (r *OpenAIRouter) videoNotFound(ctx *RequestContext) *ext_proc.ProcessingResponse {
	protocolError := llmprotocol.NewError(llmprotocol.ErrorNotFound, "video_not_found", "video not found", nil)
	if ctx != nil {
		ctx.ImmediateProtocolError = protocolError
	}
	return r.createErrorResponse(404, protocolError.Message)
}

func isVideoContentRequest(ctx *RequestContext) bool {
	return ctx != nil && ctx.SourceFormat == llmprotocol.OpenAIVideosV1 && ctx.videosContent
}

func stampVideoResponse(ctx *RequestContext, response *llmprotocol.Response) bool {
	if ctx == nil || response == nil {
		return false
	}
	route := ctx.videosRoute
	if route == "" && ctx.videosCreate {
		route = ctx.RequestModel
	}
	if route == "" {
		return false
	}
	changed := false
	for itemIndex := range response.Output {
		for contentIndex := range response.Output[itemIndex].Content {
			video := response.Output[itemIndex].Content[contentIndex].GeneratedVideo
			if video == nil || video.ID == "" {
				continue
			}
			stamped := stampVideoID(route, video.ID)
			if stamped != video.ID {
				video.ID = stamped
				changed = true
			}
		}
	}
	return changed
}
