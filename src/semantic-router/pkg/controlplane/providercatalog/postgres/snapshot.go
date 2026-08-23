package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

type catalogDocument struct {
	Providers []providercatalog.Definition `json:"providers"`
}

type persistedSnapshot struct {
	revision                   string
	payload                    []byte
	payloadDigest              []byte
	integrationReferences      []providercatalog.IntegrationReference
	catalog                    catalogDocument
	requiredWireFormats        []string
	requiredCredentialAdapters []string
	requiredDiscoveryAdapters  []string
}

func compilePersistedSnapshot(snapshot *providercatalog.Snapshot, registry *providercatalog.Registry) (persistedSnapshot, error) {
	if snapshot == nil || !validRevision(snapshot.Revision()) {
		return persistedSnapshot{}, fmt.Errorf("provider catalog snapshot is required")
	}
	payload, err := snapshot.MarshalBinary()
	if err != nil {
		return persistedSnapshot{}, err
	}
	restored, err := providercatalog.RestoreSnapshot(payload, registry)
	if err != nil {
		return persistedSnapshot{}, fmt.Errorf("validate provider catalog snapshot: %w", err)
	}
	canonical, err := restored.MarshalBinary()
	if err != nil || !bytes.Equal(payload, canonical) || restored.Revision() != snapshot.Revision() {
		return persistedSnapshot{}, fmt.Errorf("provider catalog snapshot is not canonical")
	}
	digest := sha256.Sum256(payload)
	wireFormats, credential, discovery := requiredRuntimeCapabilities(restored.List())
	return persistedSnapshot{
		revision: restored.Revision(), payload: payload, payloadDigest: append([]byte(nil), digest[:]...),
		integrationReferences: restored.IntegrationReferences(), catalog: catalogDocument{Providers: restored.List()},
		requiredWireFormats: wireFormats, requiredCredentialAdapters: credential,
		requiredDiscoveryAdapters: discovery,
	}, nil
}

func requiredRuntimeCapabilities(providers []providercatalog.Definition) ([]string, []string, []string) {
	wireFormats := make(map[string]struct{})
	credential := make(map[string]struct{})
	discovery := make(map[string]struct{})
	for _, provider := range providers {
		for _, providerInterface := range provider.Interfaces {
			wireFormats[string(providerInterface.WireFormat)] = struct{}{}
		}
		if provider.Credential.AdapterID != "" {
			credential[provider.Credential.AdapterID] = struct{}{}
		}
		if provider.Discovery != nil {
			discovery[provider.Discovery.AdapterID] = struct{}{}
		}
	}
	return sortedSet(wireFormats), sortedSet(credential), sortedSet(discovery)
}

func sortedSet(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func insertImmutableSnapshot(ctx context.Context, tx *sql.Tx, expected persistedSnapshot, registry *providercatalog.Registry) error {
	integrationReferences, insertImmutableSnapshotErr := json.Marshal(expected.integrationReferences)
	if insertImmutableSnapshotErr != nil {
		return insertImmutableSnapshotErr
	}
	catalog, insertImmutableSnapshotErr := json.Marshal(expected.catalog)
	if insertImmutableSnapshotErr != nil {
		return insertImmutableSnapshotErr
	}
	wireFormats, _ := json.Marshal(expected.requiredWireFormats)
	credential, _ := json.Marshal(expected.requiredCredentialAdapters)
	discovery, _ := json.Marshal(expected.requiredDiscoveryAdapters)
	if _, err := tx.ExecContext(ctx, `INSERT INTO provider_catalog_revisions
	  (revision, snapshot_bytes, snapshot_digest, integration_references, catalog,
   required_wire_formats, required_credential_adapters, required_discovery_adapters)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (revision) DO NOTHING`, expected.revision, expected.payload, expected.payloadDigest,
		integrationReferences, catalog, wireFormats, credential, discovery); err != nil {
		return fmt.Errorf("persist immutable provider catalog snapshot: %w", err)
	}
	stored, insertImmutableSnapshotErr := readPersistedSnapshot(ctx, tx, expected.revision)
	if insertImmutableSnapshotErr != nil {
		return insertImmutableSnapshotErr
	}
	if !bytes.Equal(stored.payload, expected.payload) || !bytes.Equal(stored.payloadDigest, expected.payloadDigest) ||
		!reflect.DeepEqual(stored.integrationReferences, expected.integrationReferences) ||
		!reflect.DeepEqual(stored.catalog, expected.catalog) ||
		!reflect.DeepEqual(stored.requiredWireFormats, expected.requiredWireFormats) ||
		!reflect.DeepEqual(stored.requiredCredentialAdapters, expected.requiredCredentialAdapters) ||
		!reflect.DeepEqual(stored.requiredDiscoveryAdapters, expected.requiredDiscoveryAdapters) {
		return fmt.Errorf("%w: revision %s already has different immutable content", providercatalog.ErrPublicationConflict, expected.revision)
	}
	if _, err := restorePersistedSnapshot(stored, registry); err != nil {
		return err
	}
	return nil
}

type rowQuerier interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func readPersistedSnapshot(ctx context.Context, query rowQuerier, revision string) (persistedSnapshot, error) {
	var stored persistedSnapshot
	stored.revision = revision
	var references, catalog, wireFormats, credential, discovery []byte
	err := query.QueryRowContext(ctx, `SELECT snapshot_bytes, snapshot_digest, integration_references, catalog,
required_wire_formats, required_credential_adapters, required_discovery_adapters
FROM provider_catalog_revisions WHERE revision = $1`, revision).Scan(
		&stored.payload, &stored.payloadDigest, &references, &catalog, &wireFormats, &credential, &discovery,
	)
	if err == sql.ErrNoRows {
		return persistedSnapshot{}, fmt.Errorf("%w: persisted revision %s is absent", ErrCorruptSnapshot, revision)
	}
	if err != nil {
		return persistedSnapshot{}, fmt.Errorf("read provider catalog snapshot: %w", err)
	}
	for label, source := range map[string]struct {
		payload []byte
		target  any
	}{
		"integration references": {references, &stored.integrationReferences},
		"catalog":                {catalog, &stored.catalog},
		"wire formats":           {wireFormats, &stored.requiredWireFormats},
		"credential adapters":    {credential, &stored.requiredCredentialAdapters},
		"discovery adapters":     {discovery, &stored.requiredDiscoveryAdapters},
	} {
		if err := decodeStrict(source.payload, source.target); err != nil {
			return persistedSnapshot{}, fmt.Errorf("%w: decode %s: %w", ErrCorruptSnapshot, label, err)
		}
	}
	return stored, nil
}

func restorePersistedSnapshot(stored persistedSnapshot, registry *providercatalog.Registry) (*providercatalog.Snapshot, error) {
	if !validRevision(stored.revision) || len(stored.payloadDigest) != sha256.Size {
		return nil, fmt.Errorf("%w: revision or payload digest is invalid", ErrCorruptSnapshot)
	}
	digest := sha256.Sum256(stored.payload)
	if !bytes.Equal(digest[:], stored.payloadDigest) {
		return nil, fmt.Errorf("%w: snapshot byte digest differs", ErrCorruptSnapshot)
	}
	snapshot, err := providercatalog.RestoreSnapshot(stored.payload, registry)
	if err != nil || snapshot.Revision() != stored.revision {
		return nil, fmt.Errorf("%w: restore revision %s: %w", ErrCorruptSnapshot, stored.revision, err)
	}
	wireFormats, credential, discovery := requiredRuntimeCapabilities(snapshot.List())
	if !reflect.DeepEqual(snapshot.IntegrationReferences(), stored.integrationReferences) ||
		!reflect.DeepEqual(catalogDocument{Providers: snapshot.List()}, stored.catalog) ||
		!reflect.DeepEqual(wireFormats, stored.requiredWireFormats) ||
		!reflect.DeepEqual(credential, stored.requiredCredentialAdapters) ||
		!reflect.DeepEqual(discovery, stored.requiredDiscoveryAdapters) {
		return nil, fmt.Errorf("%w: snapshot metadata differs from immutable bytes", ErrCorruptSnapshot)
	}
	return snapshot, nil
}

func decodeStrict(payload []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("JSON contains trailing values")
	}
	return nil
}

func normalizeRolloutGroups(input []providercatalog.RolloutGroup) ([]providercatalog.RolloutGroup, error) {
	return providercatalog.CanonicalRolloutGroups(input)
}

func validateReplicaID(value string) error {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 256 {
		return fmt.Errorf("replica ID is required, canonical, and bounded")
	}
	for _, char := range value {
		if char < 0x20 || char == 0x7f {
			return fmt.Errorf("replica ID contains a control character")
		}
	}
	return nil
}
