package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) kvAddressRegistry() kvtransfer.AddressRegistry {
	if r == nil {
		return kvtransfer.NoopAddressRegistry{}
	}
	r.kvAddressRegistryMu.Lock()
	defer r.kvAddressRegistryMu.Unlock()
	if r.kvAddressRegistryStore != nil {
		return r.kvAddressRegistryStore
	}
	if r.Config == nil {
		r.kvAddressRegistryStore = kvtransfer.NoopAddressRegistry{}
		return r.kvAddressRegistryStore
	}
	registry, err := kvtransfer.NewAddressRegistryFromResponseCacheStore(r.Config.SemanticCache)
	if err != nil {
		logging.Warnf("KV address registry unavailable: %v", err)
		r.kvAddressRegistryStore = kvtransfer.NoopAddressRegistry{}
		return r.kvAddressRegistryStore
	}
	r.kvAddressRegistryStore = registry
	return r.kvAddressRegistryStore
}

func (r *OpenAIRouter) updateKVAddressRegistry(ctx *RequestContext) {
	if r == nil || ctx == nil || !shouldWriteKVAddressRegistry(ctx) {
		return
	}
	record, ok := kvAddressRecordFromContext(ctx)
	if !ok {
		return
	}
	registry := r.kvAddressRegistry()
	if registry == nil {
		return
	}
	writeCtx := cacheWriteContext(ctx)
	if err := registry.Write(writeCtx, record); err != nil {
		logging.Warnf("Failed to write KV address registry for request %s: %v", ctx.RequestID, err)
	}
}

func shouldWriteKVAddressRegistry(ctx *RequestContext) bool {
	if ctx == nil || ctx.LooperRequest || requestBypassesRouting(ctx) {
		return false
	}
	if skip, _ := shouldSkipCacheWriteForStatus(ctx); skip {
		return false
	}
	if routingSessionStateKey(ctx) == "" {
		return false
	}
	return strings.TrimSpace(ctx.UpstreamBackendAddress) != ""
}

func kvAddressRecordFromContext(ctx *RequestContext) (kvtransfer.AddressRecord, bool) {
	if ctx == nil {
		return kvtransfer.AddressRecord{}, false
	}
	sessionKey := routingSessionStateKey(ctx)
	if sessionKey == "" || ctx.UpstreamBackendAddress == "" || ctx.RequestModel == "" {
		return kvtransfer.AddressRecord{}, false
	}
	turnCount := ctx.TurnIndex + 1
	if turnCount < 1 {
		turnCount = 1
	}
	return kvtransfer.AddressRecord{
		SessionID: sessionKey,
		SourcePod: ctx.UpstreamBackendAddress,
		Model:     ctx.RequestModel,
		Namespace: kvAddressNamespace(ctx),
		TurnCount: turnCount,
		UpdatedAt: time.Now().UTC(),
	}, true
}

func kvAddressNamespace(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	scope := responseCacheScope(ctx)
	scopeIdentity := responseCacheScopeIdentity(ctx)
	if scopeIdentity != "" {
		scopeIdentity = scope + ":" + scopeIdentity
	}
	return cache.UserScopeNamespace(scopeIdentity)
}
