package managementcomposition

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

// accessRoutingPublicationReader resolves the same immutable runtime
// publication consumed by inference. It deliberately does not consult mutable
// authoring state or select a latest routing snapshot.
type accessRoutingPublicationReader struct {
	store routingPublicationStore
}

type routingPublicationStore interface {
	GetPublicationNamespace(context.Context, string) (accesspublisher.NamespacePublication, error)
	ReadPublicationHeads(context.Context, accesspublisher.NamespacePublication) (accesspublisher.PublicationHeads, error)
	LoadRoutingPublication(context.Context, accesspublisher.RuntimePublicationIdentity) (accesspublisher.LoadedRoutingPublication, error)
}

func newAccessRoutingPublicationReader(
	client redis.UniversalClient,
	keyPrefix string,
) (*accessRoutingPublicationReader, error) {
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: client, KeyPrefix: keyPrefix,
	})
	if err != nil {
		return nil, err
	}
	return &accessRoutingPublicationReader{store: store}, nil
}

func (reader *accessRoutingPublicationReader) ReadRoutingPublication(
	ctx context.Context,
	pin accessmanagement.RoutingPublicationPin,
) (*accessmanagement.RoutingPublication, error) {
	if reader == nil || reader.store == nil || !validRoutingPublicationPin(pin) {
		return nil, errors.New("routing publication reader or pin is invalid")
	}
	reference, err := reader.store.GetPublicationNamespace(ctx, pin.NamespaceID)
	if err != nil {
		return nil, fmt.Errorf("locate routing publication namespace: %w", err)
	}
	if reference.QuotaPartition != pin.QuotaPartition {
		return nil, errors.New("routing publication partition does not match the applied policy")
	}

	identity, err := reader.readPinnedActive(ctx, reference, pin)
	if err != nil {
		return nil, err
	}
	loaded, err := reader.store.LoadRoutingPublication(ctx, identity)
	if err != nil {
		return nil, fmt.Errorf("load exact routing publication: %w", err)
	}
	if !loaded.Identity.SameGeneration(identity) ||
		loaded.Snapshot.NamespaceID != pin.NamespaceID ||
		loaded.Snapshot.Revision != pin.RoutingRevision ||
		loaded.Identity.RoutingDigest != pin.RoutingDocumentDigest {
		return nil, errors.New("loaded routing publication does not match the applied policy")
	}
	// Close the head-to-document race. A concurrent activation is surfaced as
	// unavailable so callers retry from a fresh applied policy; an older
	// publication is never substituted with the new head.
	confirmed, err := reader.readPinnedActive(ctx, reference, pin)
	if err != nil {
		return nil, err
	}
	if !confirmed.SameGeneration(identity) {
		return nil, accesspublisher.ErrPublicationChanged
	}
	return &accessmanagement.RoutingPublication{
		RoutingDocumentDigest: identity.RoutingDigest,
		Snapshot:              loaded.Snapshot,
	}, nil
}

func (reader *accessRoutingPublicationReader) readPinnedActive(
	ctx context.Context,
	reference accesspublisher.NamespacePublication,
	pin accessmanagement.RoutingPublicationPin,
) (accesspublisher.RuntimePublicationIdentity, error) {
	heads, err := reader.store.ReadPublicationHeads(ctx, reference)
	if err != nil {
		return accesspublisher.RuntimePublicationIdentity{}, fmt.Errorf("read active routing publication: %w", err)
	}
	if heads.Active == nil || !runtimePublicationMatchesPin(*heads.Active, pin) {
		return accesspublisher.RuntimePublicationIdentity{}, accesspublisher.ErrPublicationChanged
	}
	return *heads.Active, nil
}

func runtimePublicationMatchesPin(
	identity accesspublisher.RuntimePublicationIdentity,
	pin accessmanagement.RoutingPublicationPin,
) bool {
	if pin.RoutingRevision <= 0 {
		return false
	}
	// #nosec G115 -- the positive int64 bound is checked above.
	revision := uint64(pin.RoutingRevision)
	return identity.Activated() && identity.PublicationID == pin.PublicationID &&
		identity.NamespaceID == pin.NamespaceID && identity.QuotaPartition == pin.QuotaPartition &&
		identity.RuntimeEpoch == pin.RuntimeEpoch && identity.DesiredRevision == revision &&
		identity.RoutingDigest == pin.RoutingDocumentDigest
}

func validRoutingPublicationPin(pin accessmanagement.RoutingPublicationPin) bool {
	return strings.TrimSpace(pin.NamespaceID) != "" && strings.TrimSpace(pin.QuotaPartition) != "" &&
		strings.TrimSpace(pin.PublicationID) != "" && pin.RuntimeEpoch > 0 &&
		pin.RoutingRevision > 0 && validRoutingDigest(pin.RoutingDocumentDigest)
}

func validRoutingDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

var _ accessmanagement.RoutingPublicationReader = (*accessRoutingPublicationReader)(nil)
