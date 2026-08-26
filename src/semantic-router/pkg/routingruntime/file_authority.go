package routingruntime

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const fileCapabilityKeyVersion = "process-v1"

type fileRoutingSource struct {
	snapshot *routingsnapshot.Snapshot
}

func (source fileRoutingSource) Snapshot(
	_ context.Context,
	pin routingcontext.Generation,
) (*routingsnapshot.Snapshot, error) {
	// File authority has no routing document envelope; filePublication
	// intentionally defines its routing digest as the snapshot bundle digest.
	publication := filePublication(source.snapshot)
	if source.snapshot == nil || publication == nil || source.snapshot.NamespaceID != pin.NamespaceID ||
		source.snapshot.Revision != pin.SnapshotRevision || source.snapshot.Digest != pin.RoutingDigest ||
		publication.QuotaPartition != pin.QuotaPartition || publication.PublicationID != pin.PublicationID ||
		publication.RuntimeEpoch != pin.RuntimeEpoch {
		return nil, fmt.Errorf("file routing snapshot revision is unavailable")
	}
	return source.snapshot, nil
}

func newFileAuthorityRuntime(
	cfg *config.RouterConfig,
	options Options,
) (_ *Runtime, resultErr error) {
	if cfg.RoutingSnapshot == nil {
		return nil, errors.New("file routing requires one compiled routing snapshot")
	}
	egressPolicy, composeErr := backendegress.LoadFile(cfg.BackendEgress.PolicyFile)
	if composeErr != nil {
		return nil, fmt.Errorf("load backend egress policy: %w", composeErr)
	}
	credentials, composeErr := backendcredential.NewResolver(cfg.BackendCredentials.File)
	if composeErr != nil {
		return nil, composeErr
	}
	runtime := &Runtime{
		capabilities:      runtimecapabilities.RuntimeCapabilities{FileRouting: true},
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
	source := fileRoutingSource{snapshot: snapshot}
	runtime.filePublication = filePublication(snapshot)
	capabilityKeyring, composeErr := ephemeralCapabilityKeyring()
	if composeErr != nil {
		return nil, composeErr
	}
	defer zeroSymmetric(&capabilityKeyring)
	capabilityLifetime, composeErr := cfg.BackendDispatch.CapabilityLifetime()
	if composeErr != nil {
		return nil, composeErr
	}
	runtime.dispatchCapabilities, composeErr = dispatchauthority.NewRoutingOnlyRuntime(
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
	if composeErr != nil {
		return nil, fmt.Errorf("compose file-authority backend dispatch authority: %w", composeErr)
	}
	if err := runtime.dispatchCapabilities.AttachRoutingSnapshots(source); err != nil {
		return nil, err
	}
	dispatch, composeErr := newBackendDispatchComposition(
		cfg.BackendDispatch,
		credentials,
		runtime.protocolCodecs,
		backendinvoker.ProcessLocalJournal{},
		runtime.responseTerminals,
		egressPolicy,
		options.BackendDialTimeout,
		runtime.Ready,
	)
	if composeErr != nil {
		return nil, composeErr
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

func filePublication(snapshot *routingsnapshot.Snapshot) *accesspublisher.RuntimePublicationIdentity {
	if snapshot == nil || snapshot.Revision <= 0 {
		return nil
	}
	// #nosec G115 -- valid routing snapshots carry a positive int64 revision.
	desiredRevision := uint64(snapshot.Revision)
	identity := accesspublisher.RuntimePublicationIdentity{
		PublicationID: "file-" + snapshot.Digest[:24],
		NamespaceID:   snapshot.NamespaceID, QuotaPartition: snapshot.NamespaceID,
		DesiredRevision: desiredRevision, RuntimeEpoch: 1,
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
		ActiveVersion: fileCapabilityKeyVersion,
		Keys:          map[string][]byte{fileCapabilityKeyVersion: key},
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
	_ backendinvoker.RoutingSnapshotSource = fileRoutingSource{}
	_ io.Closer                            = (*backendcredential.Resolver)(nil)
)
