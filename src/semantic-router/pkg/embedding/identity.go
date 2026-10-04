package embedding

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrIdentityUnsupported means this provider has no verified representation
// descriptor. It must not be treated as evidence that two vector spaces match.
var ErrIdentityUnsupported = errors.New("embedding content identity is unsupported")

// ConsumerSettings describes the actual embedding call, including the
// consumer's text preparation. Zero layer/dimension use model defaults.
type ConsumerSettings struct {
	ModelType   string
	Layer       int
	Dimension   int
	InputPolicy string
}

// ArtifactDigest binds the served content: a model package's digest.
type ArtifactDigest struct {
	Role   string `json:"role"`
	SHA256 string `json:"sha256"`
}

// RuntimeDescriptor describes the vectors a model runtime serves. Paths and
// repository revisions are deliberately absent: content defines identity.
type RuntimeDescriptor struct {
	Version           int              `json:"version"`
	ModelType         string           `json:"model_type"`
	Runtime           string           `json:"runtime"`
	Artifacts         []ArtifactDigest `json:"artifacts"`
	Layer             int              `json:"layer"`
	Dimension         int              `json:"dimension"`
	MaxSequenceLength int              `json:"max_sequence_length"`
	PoolingContract   string           `json:"pooling_contract"`
}

// ContentIdentity is suitable for isolating vector namespaces. It is not a model
// download revision, a quality certificate, or a remote provider version claim.
type ContentIdentity struct {
	Fingerprint string
	Descriptor  RuntimeDescriptor
}

// RepresentationProvider reports the identity of the vectors it serves for an
// output view and a versioned input policy, without running inference.
type RepresentationProvider interface {
	RepresentationIdentity(Options, string) (ContentIdentity, error)
}

// ResolveProviderIdentity isolates persisted vectors (semantic cache, memory,
// vector stores) by the prepared provider's representation identity.
func ResolveProviderIdentity(provider Provider, settings ConsumerSettings) (ContentIdentity, error) {
	if strings.TrimSpace(settings.ModelType) == "" {
		return ContentIdentity{}, fmt.Errorf("%w: %s", ErrIdentityUnsupported, settings.ModelType)
	}
	if settings.Layer < 0 || settings.Dimension < 0 || settings.Layer > math.MaxInt32 || settings.Dimension > math.MaxInt32 {
		return ContentIdentity{}, fmt.Errorf("embedding layer and dimension must fit nonnegative int32")
	}
	owned, ok := provider.(RepresentationProvider)
	if !ok {
		return ContentIdentity{}, fmt.Errorf("%w: prepared provider has no content descriptor", ErrIdentityUnsupported)
	}
	return owned.RepresentationIdentity(Options{Layer: settings.Layer, Dimension: settings.Dimension}, settings.InputPolicy)
}

func (s *Set) ResolveIdentity(settings ConsumerSettings) (ContentIdentity, error) {
	provider, err := s.Get(settings.ModelType, 0, 0)
	if err != nil {
		return ContentIdentity{}, err
	}
	return ResolveProviderIdentity(provider, settings)
}

// IdentityForRuntime identifies vectors a model runtime serves: the model
// package's content digest, the output view, pooling and normalization, the
// input budget and the caller's versioned input policy.
func IdentityForRuntime(descriptor RuntimeDescriptor, inputPolicy string) (ContentIdentity, error) {
	descriptor.Version = 2
	if strings.TrimSpace(descriptor.ModelType) == "" || descriptor.Runtime == "" || descriptor.PoolingContract == "" ||
		descriptor.Layer < 0 || descriptor.Dimension <= 0 || inputPolicy == "" || len(descriptor.Artifacts) == 0 {
		return ContentIdentity{}, fmt.Errorf("incomplete runtime embedding descriptor")
	}
	for _, artifact := range descriptor.Artifacts {
		if artifact.Role == "" || !validDigest(artifact.SHA256) {
			return ContentIdentity{}, fmt.Errorf("runtime embedding descriptor lacks a verified content digest")
		}
	}
	canonical, err := json.Marshal(struct {
		Descriptor  RuntimeDescriptor `json:"descriptor"`
		InputPolicy string            `json:"input_policy"`
	}{descriptor, inputPolicy})
	if err != nil {
		return ContentIdentity{}, err
	}
	digest := sha256.Sum256(canonical)
	return ContentIdentity{Fingerprint: "embedding-v2-" + hex.EncodeToString(digest[:]), Descriptor: descriptor}, nil
}

func validDigest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}
