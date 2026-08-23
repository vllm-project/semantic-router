package accesspublisher

import (
	"fmt"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func verifyManifest(manifest Manifest) error {
	expected := manifest.Digest
	manifest.Digest = ""
	actual, err := canonicalDigest(manifest)
	if err != nil || expected == "" || actual != expected {
		return fmt.Errorf("%w: manifest digest verification failed", ErrStagedCorrupt)
	}
	return nil
}

func verifyCredentialDocument(document CredentialDocument) error {
	expected := document.Digest
	document.Digest = ""
	actual, err := canonicalDigest(document)
	if err != nil || expected == "" || actual != expected {
		return fmt.Errorf("%w: credential document digest verification failed", ErrStagedCorrupt)
	}
	return nil
}

func verifyProviderCredentialDocument(document ProviderCredentialDocument) error {
	expected := document.Digest
	document.Digest = ""
	actual, verifyProviderCredentialDocumentErr := canonicalDigest(document)
	if verifyProviderCredentialDocumentErr != nil || expected == "" || actual != expected {
		return fmt.Errorf("%w: provider credential document digest verification failed", ErrStagedCorrupt)
	}
	if err := document.Credential.Validate(); err != nil {
		return fmt.Errorf("%w: provider credential metadata is invalid", ErrStagedCorrupt)
	}
	canonical, verifyProviderCredentialDocumentErr := canonicalProviderCredentialVersions(document.Credential, document.Versions)
	if verifyProviderCredentialDocumentErr != nil {
		return fmt.Errorf("%w: provider credential versions are invalid: %w", ErrStagedCorrupt, verifyProviderCredentialDocumentErr)
	}
	if !reflect.DeepEqual(canonical, document.Versions) {
		return fmt.Errorf("%w: provider credential versions are not canonical", ErrStagedCorrupt)
	}
	return nil
}

func verifyRoutingDocument(document RoutingDocument) error {
	expected := document.Digest
	document.Digest = ""
	actual, err := canonicalDigest(document)
	if err != nil || expected == "" || actual != expected {
		return fmt.Errorf("%w: routing document digest verification failed", ErrStagedCorrupt)
	}
	compiled, err := routingsnapshot.Compile(document.Snapshot.Bundle)
	if err != nil || document.Snapshot.Digest == "" || compiled.Digest != document.Snapshot.Digest {
		return fmt.Errorf("%w: routing snapshot bundle digest verification failed", ErrStagedCorrupt)
	}
	return nil
}

func verifyPublication(publication Publication) error {
	if err := publication.Validate(); err != nil {
		return err
	}
	for _, document := range publication.Access {
		if document.NamespaceID != publication.NamespaceID || document.QuotaPartition != publication.QuotaPartition ||
			document.DesiredRevision != publication.DesiredRevision || document.KeyID != document.Projection.KeyID ||
			document.Digest != document.Projection.Digest {
			return fmt.Errorf("access document identity differs from publication")
		}
		if err := document.Projection.VerifyDigest(document.Digest); err != nil {
			return err
		}
	}
	for _, document := range publication.Credentials {
		if document.NamespaceID != publication.NamespaceID || document.QuotaPartition != publication.QuotaPartition ||
			document.DesiredRevision != publication.DesiredRevision {
			return fmt.Errorf("credential document identity differs from publication")
		}
		if err := verifyCredentialDocument(document); err != nil {
			return err
		}
	}
	references := providerCredentialReferences(publication.Routing.Snapshot)
	providerDocuments := make(map[string]ProviderCredentialDocument, len(publication.ProviderCredentials))
	for _, document := range publication.ProviderCredentials {
		if document.NamespaceID != publication.NamespaceID || document.QuotaPartition != publication.QuotaPartition ||
			document.DesiredRevision != publication.DesiredRevision {
			return fmt.Errorf("provider credential document identity differs from publication")
		}
		if _, duplicate := providerDocuments[document.Credential.ID]; duplicate {
			return fmt.Errorf("provider credential document is duplicated")
		}
		providerDocuments[document.Credential.ID] = document
		if err := verifyProviderCredentialDocument(document); err != nil {
			return err
		}
	}
	if len(providerDocuments) != len(references) {
		return fmt.Errorf("provider credential documents differ from routing references")
	}
	for _, model := range publication.Routing.Snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			document, exists := providerDocuments[backend.ProviderCredentialID]
			if !exists || document.Credential.ProviderID != backend.ProviderID ||
				document.Credential.NormalizedOrigin != backend.Origin {
				return fmt.Errorf("provider credential document binding differs from routing backend")
			}
		}
	}
	if err := verifyRoutingDocument(publication.Routing); err != nil {
		return err
	}
	if err := verifyManifest(publication.Manifest); err != nil {
		return err
	}
	expectedManifest := compileManifest(publication)
	expectedManifestDigest, err := canonicalDigest(expectedManifest)
	if err != nil || publication.Manifest.Digest != expectedManifestDigest {
		return fmt.Errorf("publication manifest differs from immutable documents")
	}
	expected := publication.Digest
	actual, err := canonicalDigest(struct {
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
	if err != nil || actual != expected {
		return fmt.Errorf("publication digest verification failed")
	}
	return nil
}
