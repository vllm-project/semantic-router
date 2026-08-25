package accesspublisher

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const maximumPublishedProviderCredentialVersions = 32

func Compile(state DesiredState) (Publication, error) {
	if state.Revision == 0 || state.RevisionTime.IsZero() {
		return Publication{}, fmt.Errorf("desired revision and revision timestamp are required")
	}
	if err := state.Namespace.Validate(); err != nil {
		return Publication{}, fmt.Errorf("namespace: %w", err)
	}
	if state.Namespace.Status != accesscontrol.NamespaceStatusActive {
		if len(state.Keys) != 0 || len(state.Credentials) != 0 {
			return Publication{}, fmt.Errorf("disabled namespace %s cannot publish active keys or credentials", state.Namespace.ID)
		}
		state.BarrierHints = append(state.BarrierHints, Barrier{
			Kind: "namespace", ResourceID: string(state.Namespace.ID), Reason: "namespace_disabled",
		})
	}
	desiredRevision, err := postgresBigint(state.Revision, "desired revision")
	if err != nil {
		return Publication{}, err
	}
	if state.Routing.NamespaceID != string(state.Namespace.ID) || state.Routing.Revision != desiredRevision {
		return Publication{}, fmt.Errorf("routing bundle does not pin the desired namespace revision")
	}
	routing, compileErr := compileRouting(state.Routing, state.Revision)
	if compileErr != nil {
		return Publication{}, compileErr
	}
	resources := routingResourceSet(routing.Snapshot)
	providerCredentials, compileErr := compileProviderCredentials(
		state.Namespace, state.Revision, state.ProviderCredentials, routing.Snapshot,
	)
	if compileErr != nil {
		return Publication{}, compileErr
	}

	access, keyIDs, compileErr := compileAccessDocuments(state, resources)
	if compileErr != nil {
		return Publication{}, compileErr
	}
	credentials, compileErr := compileCredentialDocuments(state, keyIDs)
	if compileErr != nil {
		return Publication{}, compileErr
	}

	barriers, compileErr := canonicalBarriers(state.BarrierHints)
	if compileErr != nil {
		return Publication{}, fmt.Errorf("barrier hints: %w", compileErr)
	}
	publication := Publication{
		NamespaceID: string(state.Namespace.ID), QuotaPartition: string(state.Namespace.QuotaPartitionID),
		DesiredRevision: state.Revision, RuntimeEpoch: state.Namespace.RuntimeEpoch,
		Access: access, Credentials: credentials, ProviderCredentials: providerCredentials,
		Routing: routing, BarrierHints: barriers,
	}
	publication.Digest, compileErr = canonicalDigest(struct {
		NamespaceID         string                       `json:"namespaceId"`
		QuotaPartition      string                       `json:"quotaPartition"`
		DesiredRevision     uint64                       `json:"desiredRevision"`
		RuntimeEpoch        uint64                       `json:"runtimeEpoch"`
		Access              []AccessDocument             `json:"access"`
		Credentials         []CredentialDocument         `json:"credentials"`
		ProviderCredentials []ProviderCredentialDocument `json:"providerCredentials"`
		Routing             RoutingDocument              `json:"routing"`
	}{
		publication.NamespaceID, publication.QuotaPartition, publication.DesiredRevision,
		publication.RuntimeEpoch, publication.Access, publication.Credentials,
		publication.ProviderCredentials, publication.Routing,
	})
	if compileErr != nil {
		return Publication{}, fmt.Errorf("digest publication: %w", compileErr)
	}
	publication.ID = "pub_" + publication.Digest
	publication.Manifest = compileManifest(publication)
	publication.Manifest.Digest, compileErr = canonicalDigest(publication.Manifest)
	if compileErr != nil {
		return Publication{}, fmt.Errorf("digest manifest: %w", compileErr)
	}
	if err := publication.Validate(); err != nil {
		return Publication{}, err
	}
	return publication, nil
}

func compileAccessDocuments(
	state DesiredState,
	resources map[string]struct{},
) ([]AccessDocument, map[string]struct{}, error) {
	documents := make([]AccessDocument, 0, len(state.Keys))
	keyIDs := make(map[string]struct{}, len(state.Keys))
	for index, candidate := range state.Keys {
		if candidate.Revision != state.Revision {
			return nil, nil, fmt.Errorf("key candidate %d has revision %d, want %d", index, candidate.Revision, state.Revision)
		}
		projection, compileCandidateErr := accessprojection.Compile(candidate, accessprojection.CompileOptions{
			CalendarScheduleStart: state.RevisionTime.UTC().Truncate(time.Millisecond),
		})
		if compileCandidateErr != nil {
			return nil, nil, fmt.Errorf("compile key candidate %d: %w", index, compileCandidateErr)
		}
		if _, exists := keyIDs[projection.KeyID]; exists {
			return nil, nil, fmt.Errorf("duplicate key projection %s", projection.KeyID)
		}
		if err := validateGrantResources(projection, resources); err != nil {
			return nil, nil, fmt.Errorf("key %s: %w", projection.KeyID, err)
		}
		keyIDs[projection.KeyID] = struct{}{}
		document, err := compileAccessDocument(candidate, projection)
		if err != nil {
			return nil, nil, fmt.Errorf("key %s: %w", projection.KeyID, err)
		}
		documents = append(documents, document)
	}
	sort.Slice(documents, func(i, j int) bool { return documents[i].KeyID < documents[j].KeyID })
	return documents, keyIDs, nil
}

func compileCredentialDocuments(
	state DesiredState,
	keyIDs map[string]struct{},
) ([]CredentialDocument, error) {
	documents := make([]CredentialDocument, 0, len(state.Credentials))
	credentialIDs := make(map[string]struct{}, len(state.Credentials))
	for index, candidate := range state.Credentials {
		if strings.TrimSpace(candidate.Kind) == "" {
			return nil, fmt.Errorf("credential candidate %d kind is required", index)
		}
		projection, err := accessprojection.CompileCredential(candidate.Kind, candidate.Credential, candidate.Delegation)
		if err != nil {
			return nil, fmt.Errorf("compile credential candidate %d: %w", index, err)
		}
		if len(projection.SecretHMAC) != 32 {
			return nil, fmt.Errorf("credential %s HMAC must be exactly 32 bytes", projection.KID)
		}
		if _, exists := keyIDs[projection.KeyID]; !exists {
			return nil, fmt.Errorf("credential %s references unpublished key %s", projection.KID, projection.KeyID)
		}
		identity := credentialIdentity(candidate.Kind, projection.KID)
		if _, exists := credentialIDs[identity]; exists {
			return nil, fmt.Errorf("duplicate credential projection %s", identity)
		}
		credentialIDs[identity] = struct{}{}
		document := CredentialDocument{
			NamespaceID: string(state.Namespace.ID), QuotaPartition: string(state.Namespace.QuotaPartitionID),
			DesiredRevision: state.Revision, Kind: candidate.Kind, PublicID: projection.KID, Projection: projection,
		}
		document.Digest, err = canonicalDigest(document)
		if err != nil {
			return nil, fmt.Errorf("digest credential %s: %w", identity, err)
		}
		documents = append(documents, document)
	}
	sort.Slice(documents, func(i, j int) bool {
		return credentialIdentity(documents[i].Kind, documents[i].PublicID) <
			credentialIdentity(documents[j].Kind, documents[j].PublicID)
	})
	return documents, nil
}

func compileProviderCredentials(
	namespace accesscontrol.Namespace,
	revision uint64,
	candidates []ProviderCredentialCandidate,
	snapshot routingsnapshot.Snapshot,
) ([]ProviderCredentialDocument, error) {
	references := providerCredentialReferences(snapshot)
	documents := make([]ProviderCredentialDocument, 0, len(candidates))
	seen := make(map[string]struct{}, len(candidates))
	for index, candidate := range candidates {
		credential := candidate.Credential
		if err := credential.Validate(); err != nil {
			return nil, fmt.Errorf("provider credential candidate %d: %w", index, err)
		}
		if credential.NamespaceID != string(namespace.ID) {
			return nil, fmt.Errorf("provider credential %s belongs to another namespace", credential.ID)
		}
		if _, referenced := references[credential.ID]; !referenced {
			return nil, fmt.Errorf("provider credential %s is not referenced by the routing snapshot", credential.ID)
		}
		if _, duplicate := seen[credential.ID]; duplicate {
			return nil, fmt.Errorf("duplicate provider credential %s", credential.ID)
		}
		seen[credential.ID] = struct{}{}
		versions, err := canonicalProviderCredentialVersions(credential, candidate.Versions)
		if err != nil {
			return nil, fmt.Errorf("provider credential %s: %w", credential.ID, err)
		}
		document := ProviderCredentialDocument{
			NamespaceID: string(namespace.ID), QuotaPartition: string(namespace.QuotaPartitionID),
			DesiredRevision: revision, Credential: credential, Versions: versions,
		}
		document.Digest, err = canonicalDigest(document)
		if err != nil {
			return nil, fmt.Errorf("digest provider credential %s: %w", credential.ID, err)
		}
		documents = append(documents, document)
	}
	if len(seen) != len(references) {
		missing := make([]string, 0, len(references)-len(seen))
		for credentialID := range references {
			if _, exists := seen[credentialID]; !exists {
				missing = append(missing, credentialID)
			}
		}
		sort.Strings(missing)
		return nil, fmt.Errorf("routing snapshot references unpublished provider credentials: %s", strings.Join(missing, ", "))
	}
	byID := make(map[string]ProviderCredentialDocument, len(documents))
	for _, document := range documents {
		byID[document.Credential.ID] = document
	}
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			document := byID[backend.ProviderCredentialID]
			if document.Credential.ProviderID != backend.ProviderID ||
				document.Credential.NormalizedOrigin != backend.Origin {
				return nil, fmt.Errorf("provider credential %s binding differs from backend %s", backend.ProviderCredentialID, backend.ID)
			}
		}
	}
	sort.Slice(documents, func(i, j int) bool {
		return documents[i].Credential.ID < documents[j].Credential.ID
	})
	return documents, nil
}

func canonicalProviderCredentialVersions(
	credential providercredential.Credential,
	input []providercredential.Version,
) ([]providercredential.Version, error) {
	if len(input) > maximumPublishedProviderCredentialVersions {
		return nil, fmt.Errorf("version count exceeds %d", maximumPublishedProviderCredentialVersions)
	}
	versions := append([]providercredential.Version(nil), input...)
	seen := make(map[string]struct{}, len(versions))
	active := 0
	for index, version := range versions {
		if err := version.Validate(); err != nil {
			return nil, fmt.Errorf("version %d: %w", index, err)
		}
		if version.NamespaceID != credential.NamespaceID || version.CredentialID != credential.ID {
			return nil, fmt.Errorf("version %s binding is invalid", version.ID)
		}
		if _, duplicate := seen[version.ID]; duplicate {
			return nil, fmt.Errorf("version %s is duplicated", version.ID)
		}
		seen[version.ID] = struct{}{}
		switch version.Status {
		case providercredential.VersionActive:
			active++
			if credential.ActiveVersionID == nil || *credential.ActiveVersionID != version.ID {
				return nil, fmt.Errorf("version %s is not the credential active version", version.ID)
			}
		case providercredential.VersionRetiring:
		default:
			return nil, fmt.Errorf("version %s is not publishable", version.ID)
		}
	}
	if credential.Status == providercredential.StatusActive {
		if active != 1 || credential.ActiveVersionID == nil {
			return nil, fmt.Errorf("active credential requires exactly one active version")
		}
	} else if len(versions) != 0 || credential.ActiveVersionID != nil {
		return nil, fmt.Errorf("inactive credential cannot publish encrypted versions")
	}
	sort.Slice(versions, func(i, j int) bool {
		if versions[i].Status != versions[j].Status {
			return versions[i].Status < versions[j].Status
		}
		return versions[i].ID < versions[j].ID
	})
	return versions, nil
}

func providerCredentialReferences(snapshot routingsnapshot.Snapshot) map[string]struct{} {
	result := make(map[string]struct{})
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID != "" {
				result[backend.ProviderCredentialID] = struct{}{}
			}
		}
	}
	return result
}

func compileAccessDocument(
	candidate accessprojection.Candidate,
	projection accessprojection.Projection,
) (AccessDocument, error) {
	document := AccessDocument{
		NamespaceID: projection.NamespaceID, QuotaPartition: projection.QuotaPartition,
		DesiredRevision: candidate.Revision, KeyID: projection.KeyID, Projection: projection,
	}
	document.Digest = projection.Digest
	return document, nil
}

func compileRouting(bundle routingsnapshot.Bundle, revision uint64) (RoutingDocument, error) {
	snapshot, err := routingsnapshot.Compile(bundle)
	if err != nil {
		return RoutingDocument{}, fmt.Errorf("compile routing snapshot: %w", err)
	}
	resources := make(map[string]string, len(snapshot.Models)+len(snapshot.Recipes)+len(snapshot.Entrypoints))
	for _, model := range snapshot.Models {
		resources[routingResourceKey("model", model.ID)], err = canonicalDigest(model)
		if err != nil {
			return RoutingDocument{}, err
		}
	}
	for _, recipe := range snapshot.Recipes {
		resources[routingResourceKey("recipe", recipe.ID)], err = canonicalDigest(recipe)
		if err != nil {
			return RoutingDocument{}, err
		}
	}
	for _, entrypoint := range snapshot.Entrypoints {
		resources[routingResourceKey("entrypoint", entrypoint.ID)], err = canonicalDigest(entrypoint)
		if err != nil {
			return RoutingDocument{}, err
		}
	}
	document := RoutingDocument{
		NamespaceID: bundle.NamespaceID, DesiredRevision: revision,
		Snapshot: *snapshot, ResourceDigests: resources,
	}
	document.Digest, err = canonicalDigest(document)
	if err != nil {
		return RoutingDocument{}, fmt.Errorf("digest routing document: %w", err)
	}
	return document, nil
}

func compileManifest(publication Publication) Manifest {
	manifest := Manifest{
		NamespaceID: publication.NamespaceID, QuotaPartition: publication.QuotaPartition,
		DesiredRevision: publication.DesiredRevision, RuntimeEpoch: publication.RuntimeEpoch,
		PublicationID: publication.ID, Access: make(map[string]ManifestEntry, len(publication.Access)),
		Credentials:         make(map[string]ManifestEntry, len(publication.Credentials)),
		ProviderCredentials: make(map[string]ManifestEntry, len(publication.ProviderCredentials)),
		RoutingDigest:       publication.Routing.Digest,
		RoutingResources:    cloneStringMap(publication.Routing.ResourceDigests),
	}
	for _, document := range publication.Access {
		manifest.Access[document.KeyID] = ManifestEntry{Revision: publication.DesiredRevision, Digest: document.Digest}
	}
	for _, document := range publication.Credentials {
		manifest.Credentials[credentialIdentity(document.Kind, document.PublicID)] = ManifestEntry{
			Revision: publication.DesiredRevision, Digest: document.Digest,
		}
	}
	for _, document := range publication.ProviderCredentials {
		manifest.ProviderCredentials[document.Credential.ID] = ManifestEntry{
			Revision: publication.DesiredRevision, Digest: document.Digest,
		}
	}
	return manifest
}

func validateGrantResources(projection accessprojection.Projection, resources map[string]struct{}) error {
	for _, grant := range projection.Grants {
		if _, exists := resources[routingResourceKey(string(grant.ResourceType), grant.ResourceID)]; !exists {
			return fmt.Errorf("grant references routing resource %s/%s absent from the snapshot", grant.ResourceType, grant.ResourceID)
		}
	}
	return nil
}

func routingResourceSet(snapshot routingsnapshot.Snapshot) map[string]struct{} {
	result := make(map[string]struct{}, len(snapshot.Models)+len(snapshot.Entrypoints))
	for _, model := range snapshot.Models {
		result[routingResourceKey("model", model.ID)] = struct{}{}
	}
	for _, entrypoint := range snapshot.Entrypoints {
		result[routingResourceKey("entrypoint", entrypoint.ID)] = struct{}{}
	}
	return result
}

func routingResourceKey(kind, id string) string       { return kind + ":" + id }
func credentialIdentity(kind, publicID string) string { return kind + ":" + publicID }
func canonicalDigest(value any) (string, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:]), nil
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}
	output := make(map[string]string, len(input))
	for key, value := range input {
		output[key] = value
	}
	return output
}
