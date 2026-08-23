package managedruntime

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendcredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const standaloneCapabilityKeyVersion = "process-v1"

type standaloneRoutingSource struct {
	snapshot *routingsnapshot.Snapshot
}

func (source standaloneRoutingSource) Snapshot(
	_ context.Context,
	pin routingcontext.Generation,
) (*routingsnapshot.Snapshot, error) {
	// Standalone has no routing document envelope; standalonePublication
	// intentionally defines its routing digest as the snapshot bundle digest.
	publication := standalonePublication(source.snapshot)
	if source.snapshot == nil || publication == nil || source.snapshot.NamespaceID != pin.NamespaceID ||
		source.snapshot.Revision != pin.SnapshotRevision || source.snapshot.Digest != pin.RoutingDigest ||
		publication.QuotaPartition != pin.QuotaPartition || publication.PublicationID != pin.PublicationID ||
		publication.RuntimeEpoch != pin.RuntimeEpoch {
		return nil, fmt.Errorf("standalone routing snapshot revision is unavailable")
	}
	return source.snapshot, nil
}

func newStandalone(
	cfg *config.RouterConfig,
	options Options,
) (_ *Runtime, resultErr error) {
	if cfg.RoutingSnapshot == nil {
		return nil, errors.New("standalone mode requires one compiled routing snapshot")
	}
	egressPolicy, newStandaloneErr := backendegress.LoadFile(cfg.BackendEgress.PolicyFile)
	if newStandaloneErr != nil {
		return nil, fmt.Errorf("load backend egress policy: %w", newStandaloneErr)
	}
	credentials, newStandaloneErr := backendcredential.NewResolver(cfg.BackendCredentials.Standalone)
	if newStandaloneErr != nil {
		return nil, newStandaloneErr
	}
	runtime := &Runtime{
		mode:              config.ControlPlaneModeStandalone,
		credentialCloser:  credentials,
		responseTerminals: backendinvoker.NewLocalResponseTerminalStore(),
		protocolCodecs:    protocolcodec.NewBuiltinRegistry(),
	}
	defer func() {
		if resultErr != nil {
			_ = runtime.Close()
		}
	}()

	snapshot := cfg.RoutingSnapshot
	source := standaloneRoutingSource{snapshot: snapshot}
	runtime.standalonePublication = standalonePublication(snapshot)
	capabilityKeyring, newStandaloneErr := ephemeralCapabilityKeyring()
	if newStandaloneErr != nil {
		return nil, newStandaloneErr
	}
	defer zeroSymmetric(&capabilityKeyring)
	capabilityLifetime, newStandaloneErr := cfg.BackendDispatch.CapabilityLifetime()
	if newStandaloneErr != nil {
		return nil, newStandaloneErr
	}
	runtime.dispatchCapabilities, newStandaloneErr = dispatchauthority.NewRoutingOnlyRuntime(
		dispatchauthority.RoutingOnlyAuthorityOptions{
			NamespaceID:  snapshot.NamespaceID,
			Publications: runtime,
			Issuer: backendinvoker.CapabilityIssuerOptions{
				Audience: cfg.BackendDispatch.Audience,
				Keyring: backendinvoker.SigningKeyring{
					ActiveVersion: capabilityKeyring.ActiveVersion,
					Keys:          cloneSymmetricKeys(capabilityKeyring.Keys),
					MaxLifetime:   capabilityLifetime,
				},
				Lifetime: capabilityLifetime,
			},
		},
	)
	if newStandaloneErr != nil {
		return nil, fmt.Errorf("compose standalone backend dispatch authority: %w", newStandaloneErr)
	}
	if err := runtime.dispatchCapabilities.AttachRoutingSnapshots(source); err != nil {
		return nil, err
	}
	dispatch, newStandaloneErr := newBackendDispatchComposition(
		cfg.BackendDispatch,
		credentials,
		runtime.protocolCodecs,
		backendinvoker.ProcessLocalJournal{},
		runtime.responseTerminals,
		egressPolicy,
		options.BackendDialTimeout,
	)
	if newStandaloneErr != nil {
		return nil, newStandaloneErr
	}
	if err := dispatch.Attach(source, securitykeyring.Symmetric{
		ActiveVersion: capabilityKeyring.ActiveVersion,
		Keys:          cloneSymmetricKeys(capabilityKeyring.Keys),
	}); err != nil {
		_ = dispatch.Close()
		return nil, err
	}
	runtime.backendDispatch = dispatch
	return runtime, nil
}

func standalonePublication(snapshot *routingsnapshot.Snapshot) *accesspublisher.RuntimePublicationIdentity {
	if snapshot == nil {
		return nil
	}
	identity := accesspublisher.RuntimePublicationIdentity{
		PublicationID: "standalone-" + snapshot.Digest[:24],
		NamespaceID:   snapshot.NamespaceID, QuotaPartition: snapshot.NamespaceID,
		DesiredRevision: uint64(snapshot.Revision), RuntimeEpoch: 1,
		PublicationDigest: snapshot.Digest, ManifestDigest: snapshot.Digest,
		RoutingDigest: snapshot.Digest, State: accesspublisher.PublicationStateActive,
	}
	return &identity
}

func ephemeralCapabilityKeyring() (securitykeyring.Symmetric, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return securitykeyring.Symmetric{}, fmt.Errorf("generate process-local backend capability key: %w", err)
	}
	return securitykeyring.Symmetric{
		ActiveVersion: standaloneCapabilityKeyVersion,
		Keys:          map[string][]byte{standaloneCapabilityKeyVersion: key},
	}, nil
}

func cloneSymmetricKeys(source map[string][]byte) map[string][]byte {
	result := make(map[string][]byte, len(source))
	for version, key := range source {
		result[version] = append([]byte(nil), key...)
	}
	return result
}

var (
	_ backendinvoker.RoutingSnapshotSource = standaloneRoutingSource{}
	_ io.Closer                            = (*backendcredential.Resolver)(nil)
)
