package embedding

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ErrIdentityUnsupported means this provider has no verified local representation
// descriptor. It must not be treated as evidence that two vector spaces match.
var ErrIdentityUnsupported = errors.New("embedding content identity is unsupported")

// ConsumerSettings describes the actual embedding call, including text preparation
// performed before crossing the native ABI. Zero layer/dimension use model defaults.
type ConsumerSettings struct {
	ModelType   string
	Layer       int
	Dimension   int
	InputPolicy string
}

// ArtifactDigest binds one loaded weight, graph, or external tensor artifact.
type ArtifactDigest struct {
	Role   string `json:"role"`
	SHA256 string `json:"sha256"`
}

// RuntimeDescriptor is captured from a successfully initialized local model.
// Paths and repository revisions are deliberately absent: content defines identity.
type RuntimeDescriptor struct {
	Version               int              `json:"version"`
	ModelType             string           `json:"model_type"`
	Runtime               string           `json:"runtime"`
	EffectiveConfigSHA256 string           `json:"effective_config_sha256"`
	TokenizerSHA256       string           `json:"tokenizer_sha256"`
	Artifacts             []ArtifactDigest `json:"artifacts"`
	Layer                 int              `json:"layer"`
	Dimension             int              `json:"dimension"`
	MaxSequenceLength     int              `json:"max_sequence_length"`
	PoolingContract       string           `json:"pooling_contract"`
	ExecutionPolicy       string           `json:"execution_policy,omitempty"`
}

// ContentIdentity is suitable for isolating vector namespaces. It is not a model
// download revision, a quality certificate, or a remote provider version claim.
type ContentIdentity struct {
	Fingerprint string
	Descriptor  RuntimeDescriptor
}

// RepresentationProvider reports content captured by its owned native instance.
// It never discovers another instance or initializes global model state.
type RepresentationProvider interface {
	RepresentationIdentity(Options, string) (ContentIdentity, error)
}

func ResolveProviderIdentity(provider Provider, settings ConsumerSettings) (ContentIdentity, error) {
	if strings.ToLower(strings.TrimSpace(settings.ModelType)) != "mmbert" {
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

// IdentityFromDescriptor combines the native representation with the caller's
// explicit, versioned input policy. It never reads mutable files after model load.
func IdentityFromDescriptor(raw []byte, inputPolicy string) (ContentIdentity, error) {
	var descriptor RuntimeDescriptor
	if err := json.Unmarshal(raw, &descriptor); err != nil {
		return ContentIdentity{}, fmt.Errorf("decode embedding descriptor: %w", err)
	}
	if descriptor.Version != 1 || descriptor.ModelType != "mmbert" || descriptor.Runtime == "" || descriptor.PoolingContract == "" || descriptor.Layer <= 0 || descriptor.Dimension <= 0 || descriptor.MaxSequenceLength <= 0 || inputPolicy == "" {
		return ContentIdentity{}, fmt.Errorf("incomplete or unsupported embedding runtime descriptor")
	}
	validDigest := func(value string) bool {
		decoded, err := hex.DecodeString(value)
		return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
	}
	if !validDigest(descriptor.EffectiveConfigSHA256) || !validDigest(descriptor.TokenizerSHA256) || len(descriptor.Artifacts) == 0 {
		return ContentIdentity{}, fmt.Errorf("embedding descriptor lacks verified content digests")
	}
	sort.Slice(descriptor.Artifacts, func(i, j int) bool { return descriptor.Artifacts[i].Role < descriptor.Artifacts[j].Role })
	for i, artifact := range descriptor.Artifacts {
		if artifact.Role == "" || !validDigest(artifact.SHA256) || (i > 0 && artifact.Role == descriptor.Artifacts[i-1].Role) {
			return ContentIdentity{}, fmt.Errorf("invalid embedding artifact digest")
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
	return ContentIdentity{Fingerprint: "embedding-v1-" + hex.EncodeToString(digest[:]), Descriptor: descriptor}, nil
}
