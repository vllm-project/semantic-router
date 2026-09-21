package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Representation identities belong to immutable provider views, independently
// of the physical resource pool key. Descriptor calls use the owned instance;
// they never inspect a process-global model or reopen mutable artifact files.
func (p *EmbeddingProvider) CacheIdentity() string { return p.CacheIdentityForOptions(p.options) }

func (p *EmbeddingProvider) CacheIdentityForOptions(options embedding.Options) string {
	if p.contentIdentity {
		identity, err := p.RepresentationIdentity(options, "embedding-request-cache-v1")
		if err == nil {
			return identity.Fingerprint
		}
		// Invalid views cannot be used as evidence of persistent compatibility.
		return fmt.Sprintf("invalid-embedding-view:%p:%d:%d", p, options.Layer, options.Dimension)
	}
	return fmt.Sprintf("%s:layer=%d:dimension=%d", p.identity, options.Layer, options.Dimension)
}

func (p *EmbeddingProvider) RepresentationIdentity(options embedding.Options, inputPolicy string) (embedding.ContentIdentity, error) {
	if !p.contentIdentity {
		return embedding.ContentIdentity{}, embedding.ErrIdentityUnsupported
	}
	p.descriptorMu.Lock()
	defer p.descriptorMu.Unlock()
	raw, ok := p.descriptors[options]
	if !ok {
		err := p.resource.Use(context.Background(), func(value io.Closer) error {
			engine, ok := value.(descriptorEmbeddingEngine)
			if !ok {
				return embedding.ErrIdentityUnsupported
			}
			var err error
			raw, err = engine.RuntimeDescriptor(options.Layer, options.Dimension)
			return err
		})
		if err != nil {
			return embedding.ContentIdentity{}, err
		}
		var descriptor embedding.RuntimeDescriptor
		if decodeErr := json.Unmarshal([]byte(raw), &descriptor); decodeErr != nil {
			return embedding.ContentIdentity{}, decodeErr
		}
		descriptor.ExecutionPolicy = p.executionPolicy
		encoded, err := json.Marshal(descriptor)
		if err != nil {
			return embedding.ContentIdentity{}, err
		}
		raw = string(encoded)
		p.descriptors[options] = raw
	}
	return embedding.IdentityFromDescriptor([]byte(raw), inputPolicy)
}
