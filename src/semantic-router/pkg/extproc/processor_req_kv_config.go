package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
)

type configuredKVBackends struct{ backends []config.KVTransferBackend }

func (b configuredKVBackends) ResolveKVTarget(model, backend string) (kvtransfer.ModelIdentity, bool, bool, bool) {
	for _, item := range b.backends {
		if item.ModelAlias == model && item.BackendName == backend {
			return item.Identity, item.CanExport, item.CanLoad, true
		}
	}
	return kvtransfer.ModelIdentity{}, false, false, false
}

type registryKVSourceLookup struct {
	router *OpenAIRouter
}

func (l registryKVSourceLookup) LookupSource(ctx context.Context, namespace, session string) (*kvtransfer.SourceCache, error) {
	record, err := l.router.kvAddressRegistry().Lookup(ctx, namespace, session)
	if err != nil || record == nil {
		return nil, err
	}
	if record.CacheID == "" || record.Identity == nil || record.ExpiresAt.IsZero() {
		return nil, nil
	}
	backends := configuredKVBackends{backends: l.router.Config.KVTransfer.Backends}
	identity, export, _, ok := backends.ResolveKVTarget(record.Model, record.BackendName)
	if !ok || !export || identity != *record.Identity {
		return nil, nil
	}
	return &kvtransfer.SourceCache{Namespace: record.Namespace, SessionID: record.SessionID, CacheID: record.CacheID, Endpoint: record.SourcePod, Model: *record.Identity, ExpiresAt: record.ExpiresAt}, nil
}

func (r *OpenAIRouter) initializeKVHandoff() {
	if r.Config == nil || r.Config.KVTransfer == nil || !r.Config.KVTransfer.Enabled {
		return
	}
	backends := configuredKVBackends{backends: r.Config.KVTransfer.Backends}
	coordinator := &kvtransfer.Coordinator{Lookup: registryKVSourceLookup{router: r}, Policies: kvtransfer.PoliciesFromConfig(r.Config.KVTransfer)}
	coordinator.CanExport = func(source kvtransfer.SourceCache) bool {
		for _, backend := range backends.backends {
			if backend.Identity == source.Model && backend.CanExport {
				return true
			}
		}
		return false
	}
	r.KVHandoff = &CoordinatedKVHandoff{Coordinator: coordinator, Targets: backends}
}

const kvSourceCandidateTTL = 5 * time.Minute
