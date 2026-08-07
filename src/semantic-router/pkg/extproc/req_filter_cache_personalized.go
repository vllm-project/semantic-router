package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

func personalizedExactCacheEnabled(ctx *RequestContext) bool {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	plugin := ctx.VSRSelectedDecision.GetResponseCacheConfig()
	return plugin != nil &&
		plugin.Enabled &&
		plugin.Personalized != nil &&
		strings.TrimSpace(plugin.Personalized.Mode) == "exact"
}

func (r *OpenAIRouter) lookupPersonalizedExactCache(
	ctx *RequestContext,
	categoryName string,
	selectedModel string,
) (*ext_proc.ProcessingResponse, bool) {
	if !canUsePersonalizedExactCache(ctx) {
		return nil, false
	}
	if !preparePersonalizedCacheIdentity(ctx, selectedModel) {
		return nil, false
	}
	applyRequestCacheControls(ctx)
	if ctx.CacheReadBypass {
		return nil, false
	}
	return r.performExactCacheLookup(ctx, categoryName, true)
}

func canUsePersonalizedExactCache(ctx *RequestContext) bool {
	if !personalizedExactCacheEnabled(ctx) {
		return false
	}
	personalized, reason := ctx.HasPersonalizedContext()
	if !personalized || (reason != "rag_context" && reason != "memory_context") {
		return false
	}
	if responseCacheScope(ctx) == "global" ||
		strings.TrimSpace(responseCacheScopeIdentity(ctx)) == "" {
		return false
	}
	return len(ctx.workingRequestBody()) > 0
}

func preparePersonalizedCacheIdentity(ctx *RequestContext, selectedModel string) bool {
	workingBody := ctx.workingRequestBody()
	identity, err := cache.BuildRequestIdentity(workingBody)
	if err != nil {
		return false
	}
	contextRevision, err := cache.FingerprintValue(map[string]interface{}{
		"working_body": workingBody,
		"rag_context":  ctx.RAGRetrievedContext,
		"memory":       ctx.MemoryContext,
	})
	if err != nil || contextRevision == "" {
		return false
	}
	policyFingerprint := responseCachePolicyFingerprint(ctx)
	ctx.CacheRequestModel = identity.Model
	ctx.CacheQuery = identity.Query
	ctx.CacheSelectedModel = selectedCacheModel(identity.Model, []string{selectedModel})
	ctx.CacheExactFingerprint = cache.CombineFingerprints(
		identity.ExactFingerprint,
		policyFingerprint,
		contextRevision,
	)
	ctx.CacheCompatibilityFingerprint = cache.CombineFingerprints(
		identity.CompatibilityFingerprint,
		policyFingerprint,
		contextRevision,
	)
	ctx.CacheSemanticSafe = false
	ctx.CacheIdentity = responseCacheIdentity(ctx, identity.Model)
	return true
}

func personalizedCacheWriteAllowed(ctx *RequestContext, reason string) bool {
	return personalizedExactCacheEnabled(ctx) &&
		(reason == "rag_context" || reason == "memory_context") &&
		ctx.CacheIdentity.ExactFingerprint != ""
}
