package extproc

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func (r *OpenAIRouter) handleAuthorizedModelsRequest(
	ctx context.Context,
	request *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if !r.nativeAccessEnabled() {
		return r.handleModelsRequest("/v1/models")
	}
	if r.InferenceAccess == nil || request == nil || request.InferenceAccess == nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnauthenticated, nil), nil
	}
	discovery, err := r.InferenceAccess.DiscoverCatalog(ctx, accessruntime.CatalogDiscoveryRequest{
		Session: request.InferenceAccess.session,
		Queries: []accessruntime.DiscoveryQuery{
			{ResourceType: accesscontrol.GrantResourceEntrypoint, Permission: accesscontrol.GrantPermissionDiscover},
			{ResourceType: accesscontrol.GrantResourceModel, Permission: accesscontrol.GrantPermissionDiscover},
		},
	})
	if err != nil || !discovery.Result.Allowed() {
		return r.createInferenceAccessError(discovery.Result.Disposition, nil), nil
	}
	if discovery.Tenant.NamespaceID == "" {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil), nil
	}
	forPath, err := discoveryInferencePath(request)
	if err != nil {
		return r.createErrorResponse(http.StatusBadRequest, "invalid discovery path"), nil
	}
	allowedEntrypoints := inferenceResourceSet(discovery.Resources[accesscontrol.GrantResourceEntrypoint])
	allowedModels := inferenceResourceSet(discovery.Resources[accesscontrol.GrantResourceModel])
	claims := entrypointClaims(discovery.Tenant.RoutingClaims)
	catalog := publicmodels.NewOpenAIModelList(r.Config, time.Now().Unix())
	filtered := catalog.Data[:0]
	for _, item := range catalog.Data {
		if r.modelCatalogItemAllowed(item.ID, forPath, claims, allowedEntrypoints, allowedModels) {
			filtered = append(filtered, item)
		}
	}
	catalog.Data = filtered
	return r.createJSONResponse(http.StatusOK, catalog), nil
}

func (r *OpenAIRouter) modelCatalogItemAllowed(
	name string,
	forPath string,
	claims map[string]config.EntrypointClaimValue,
	entrypoints map[string]struct{},
	models map[string]struct{},
) bool {
	if entrypoint := r.entrypointForAlias(name); entrypoint != nil {
		if _, ok := entrypoints[entrypoint.ID]; !ok {
			return false
		}
		resolution, err := r.Config.ResolveEntrypoint(name, forPath, claims)
		return err == nil && resolution.Outcome == config.EntrypointResolveMatched &&
			resolution.Entrypoint != nil && resolution.Entrypoint.ID == entrypoint.ID
	}
	params, ok := r.Config.ModelConfig[name]
	if !ok || params.ResourceID == "" {
		return false
	}
	_, ok = models[params.ResourceID]
	return ok
}

func discoveryInferencePath(ctx *RequestContext) (string, error) {
	const defaultPath = "/v1/chat/completions"
	if ctx == nil {
		return defaultPath, nil
	}
	requestURI := strings.TrimSpace(headerValueCI(ctx, ":path"))
	if requestURI == "" {
		return defaultPath, nil
	}
	parsed, err := url.ParseRequestURI(requestURI)
	if err != nil {
		return "", err
	}
	values := parsed.Query()["for_path"]
	if len(values) == 0 || strings.TrimSpace(values[0]) == "" {
		return defaultPath, nil
	}
	if len(values) != 1 {
		return "", fmt.Errorf("for_path must appear once")
	}
	path := strings.TrimSpace(values[0])
	if len(path) > 2048 || !strings.HasPrefix(path, "/") {
		return "", fmt.Errorf("for_path must be an absolute path")
	}
	target, err := url.ParseRequestURI(path)
	if err != nil || target.IsAbs() || target.RawQuery != "" || target.Fragment != "" {
		return "", fmt.Errorf("for_path must not contain an origin, query, or fragment")
	}
	return target.Path, nil
}

func inferenceResourceSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}
