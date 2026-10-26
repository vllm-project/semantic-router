package config

import "testing"

func TestKVTransferRejectsUnpinnedBackend(t *testing.T) {
	cfg := &RouterConfig{KVTransfer: &KVTransferConfig{Enabled: true, Backends: []KVTransferBackend{{ModelAlias: "source", BackendName: "source", Identity: KVServingIdentity{Model: "source", WeightRevision: "main"}}}, Pairs: []KVTransferPair{{Enabled: false}}}}
	if validateKVTransferConfig(cfg) == nil {
		t.Fatal("accepted mutable identity")
	}
	cfg.KVTransfer.Enabled = false
	if err := validateKVTransferConfig(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestKVTransferCanonicalRoundTrip(t *testing.T) {
	original := &KVTransferConfig{Enabled: true, Backends: []KVTransferBackend{{ModelAlias: "source", BackendName: "backend", CanExport: true}}, Pairs: []KVTransferPair{{MapperID: "mapper", MaxTransferTurn: 3}}}
	cfg := &RouterConfig{}
	applyCanonicalIntegrationGlobal(cfg, CanonicalIntegrationGlobal{KVTransfer: original})
	exported := CanonicalGlobalFromRouterConfig(cfg).Integrations.KVTransfer
	if exported == nil || exported.Backends[0].BackendName != "backend" || exported.Pairs[0].MaxTransferTurn != 3 {
		t.Fatal("lost transfer configuration")
	}
}
