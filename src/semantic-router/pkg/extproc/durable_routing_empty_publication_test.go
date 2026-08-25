package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestDurableRoutingRegistryActivatesCanonicalEmptyPublication(t *testing.T) {
	base, err := config.ParseYAMLBytes([]byte(durableRoutingBootstrapYAML))
	if err != nil {
		t.Fatal(err)
	}
	base.SkipExternalAssetValidation = true
	registry, err := NewDurableRoutingRegistry(DurableRoutingRegistryOptions{
		BootstrapConfig: base,
		Dependencies: RuntimeDependencies{
			InferenceAccess:      &fakeInferenceAccess{},
			DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
			OutcomeFeedback:      &outcomeRuntimeStub{},
			OutcomeProjection:    outcomeProjectionRuntimeStub{},
			ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
			ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = registry.Close() })

	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "namespace-empty", Name: "Empty", QuotaPartitionID: "partition-empty",
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive,
		Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	compiled, err := accesspublisher.Compile(accesspublisher.DesiredState{
		Namespace: namespace, Revision: 1, RevisionTime: now,
		Routing: routingsnapshot.Bundle{
			NamespaceID: string(namespace.ID), Revision: 1, Currency: "USD",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	loaded := accesspublisher.LoadedRoutingPublication{
		Identity: accesspublisher.RuntimePublicationIdentity{
			PublicationID: compiled.ID, NamespaceID: compiled.NamespaceID,
			QuotaPartition: compiled.QuotaPartition, DesiredRevision: compiled.DesiredRevision,
			RuntimeEpoch: compiled.RuntimeEpoch, PublicationDigest: compiled.Digest,
			ManifestDigest: compiled.Manifest.Digest, RoutingDigest: compiled.Routing.Digest,
			State: accesspublisher.PublicationStateActive,
		},
		Manifest: compiled.Manifest, Routing: compiled.Routing, Snapshot: compiled.Routing.Snapshot,
	}
	if activateErr := registry.Activate(context.Background(), loaded); activateErr != nil {
		t.Fatalf("activate empty publication: %v", activateErr)
	}
	lease, err := registry.Acquire(durableRoutingPin(loaded))
	if err != nil {
		t.Fatalf("acquire empty publication: %v", err)
	}
	lease.Release()
}
