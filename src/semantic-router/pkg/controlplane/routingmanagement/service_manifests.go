package routingmanagement

import (
	"context"
	"fmt"
	"slices"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const maximumRoutingManifestBytes = 3 << 20

// PrepareManifest is the sole human-name to immutable-identity boundary for a
// Management import. The returned snapshot is the exact value authorized by
// the transport and later submitted to PostgreSQL.
func (service *Service) PrepareManifest(
	ctx context.Context,
	namespaceID string,
	document []byte,
) (PreparedManifest, error) {
	if service == nil || service.manifests == nil || !canonicalUUIDText(namespaceID) ||
		len(document) == 0 || len(document) > maximumRoutingManifestBytes {
		return PreparedManifest{}, ErrInvalid
	}
	source, err := service.manifests.Decode(document)
	if err != nil {
		return PreparedManifest{}, fmt.Errorf("%w: %w", ErrManifest, err)
	}
	if source == nil {
		return PreparedManifest{}, fmt.Errorf("%w: decoded routing manifest is empty", ErrManifest)
	}
	names := manifestProviderCredentialReferences(source)
	identities, err := service.store.ProviderCredentialIDsByName(ctx, namespaceID, names)
	if err != nil {
		return PreparedManifest{}, err
	}
	resolved, err := remapManifestProviderCredentials(source, namespaceID, identities)
	if err != nil {
		return PreparedManifest{}, err
	}
	credentialIDs := make([]string, 0, len(identities))
	for _, name := range names {
		credentialIDs = append(credentialIDs, identities[name])
	}
	slices.Sort(credentialIDs)
	credentialIDs = slices.Compact(credentialIDs)
	return PreparedManifest{
		NamespaceID: namespaceID, Snapshot: resolved, CredentialIDs: credentialIDs,
	}, nil
}

func (service *Service) ImportManifest(
	ctx context.Context, namespaceID string, request ManifestImportRequest, mutation MutationContext,
) (ManifestImportResult, error) {
	if service == nil || service.manifests == nil || !canonicalUUIDText(namespaceID) ||
		request.ExpectedRevision < 0 || request.Prepared.NamespaceID != namespaceID || request.Prepared.Snapshot == nil {
		return ManifestImportResult{}, ErrInvalid
	}
	snapshot, err := verifyPreparedManifest(request.Prepared)
	if err != nil {
		return ManifestImportResult{}, err
	}
	if request.DryRun {
		diff, previewErr := service.store.PreviewManifest(ctx, namespaceID, request.ExpectedRevision, snapshot)
		return ManifestImportResult{Diff: diff}, previewErr
	}
	diff, receipt, err := service.store.ImportManifest(ctx, namespaceID, request.ExpectedRevision, snapshot, mutation)
	return ManifestImportResult{Diff: diff, Receipt: receipt}, err
}

func (service *Service) exportCurrentManifest(
	ctx context.Context, namespaceID string,
) ([]byte, int64, error) {
	if service == nil || service.manifests == nil || !canonicalUUIDText(namespaceID) {
		return nil, 0, ErrInvalid
	}
	snapshot, revision, err := service.store.CurrentManifest(ctx, namespaceID)
	if err != nil {
		return nil, 0, err
	}
	ids := manifestProviderCredentialReferences(snapshot)
	names, err := service.store.ProviderCredentialNamesByID(ctx, namespaceID, ids)
	if err != nil {
		return nil, 0, err
	}
	readable, err := remapManifestProviderCredentials(snapshot, namespaceID, names)
	if err != nil {
		return nil, 0, err
	}
	document, err := service.manifests.Encode(readable)
	if err != nil {
		return nil, 0, fmt.Errorf("%w: %w", ErrManifest, err)
	}
	return document, revision, nil
}

func verifyPreparedManifest(prepared PreparedManifest) (*routingsnapshot.Snapshot, error) {
	if !canonicalUUIDText(prepared.NamespaceID) || prepared.Snapshot == nil ||
		prepared.Snapshot.NamespaceID != prepared.NamespaceID {
		return nil, ErrInvalid
	}
	verified, err := routingsnapshot.Compile(prepared.Snapshot.Bundle)
	if err != nil || verified.Digest != prepared.Snapshot.Digest ||
		verified.SemanticDigest != prepared.Snapshot.SemanticDigest {
		return nil, fmt.Errorf("%w: prepared routing manifest digest is invalid", ErrManifest)
	}
	wantIDs := manifestProviderCredentialReferences(verified)
	gotIDs := append([]string(nil), prepared.CredentialIDs...)
	slices.Sort(gotIDs)
	gotIDs = slices.Compact(gotIDs)
	if !slices.Equal(wantIDs, gotIDs) {
		return nil, fmt.Errorf("%w: prepared routing manifest credential closure is invalid", ErrManifest)
	}
	for _, id := range wantIDs {
		if _, parseErr := uuid.Parse(id); parseErr != nil {
			return nil, fmt.Errorf("%w: prepared routing manifest credential identity is invalid", ErrManifest)
		}
	}
	return verified, nil
}

func remapManifestProviderCredentials(
	source *routingsnapshot.Snapshot,
	namespaceID string,
	references map[string]string,
) (*routingsnapshot.Snapshot, error) {
	if source == nil || !canonicalUUIDText(namespaceID) {
		return nil, ErrInvalid
	}
	bundle := source.Bundle
	bundle.NamespaceID = namespaceID
	bundle.Models = append([]routingsnapshot.Model(nil), source.Models...)
	for modelIndex := range bundle.Models {
		model := &bundle.Models[modelIndex]
		model.Backends = append([]routingsnapshot.Backend(nil), model.Backends...)
		for backendIndex := range model.Backends {
			backend := &model.Backends[backendIndex]
			if backend.ProviderCredentialID == "" {
				continue
			}
			resolved, found := references[backend.ProviderCredentialID]
			if !found || resolved == "" {
				return nil, fmt.Errorf("%w: ProviderCredential reference is unavailable", ErrManifest)
			}
			backend.ProviderCredentialID = resolved
		}
	}
	compiled, err := routingsnapshot.Compile(bundle)
	if err != nil {
		return nil, fmt.Errorf("%w: compile resolved routing manifest: %w", ErrManifest, err)
	}
	return compiled, nil
}

func manifestProviderCredentialReferences(snapshot *routingsnapshot.Snapshot) []string {
	if snapshot == nil {
		return nil
	}
	values := make([]string, 0)
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID != "" {
				values = append(values, backend.ProviderCredentialID)
			}
		}
	}
	slices.Sort(values)
	return slices.Compact(values)
}
