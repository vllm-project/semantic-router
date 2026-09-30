package embedding

import (
	"context"
	"encoding/json"
	"testing"
)

type identityViewProvider struct {
	*FuncProvider
	descriptor RuntimeDescriptor
	calls      int
}

func (p *identityViewProvider) RepresentationIdentity(options Options, policy string) (ContentIdentity, error) {
	p.calls++
	d := p.descriptor
	if options.Layer > 0 {
		d.Layer = options.Layer
	}
	if options.Dimension > 0 {
		d.Dimension = options.Dimension
	}
	raw, err := json.Marshal(d)
	if err != nil {
		return ContentIdentity{}, err
	}
	return IdentityFromDescriptor(raw, policy)
}

func (p *identityViewProvider) CacheIdentityForOptions(options Options) string {
	identity, _ := p.RepresentationIdentity(options, "embedding-request-cache-v1")
	return identity.Fingerprint
}

func (p *identityViewProvider) CacheIdentity() string { return p.CacheIdentityForOptions(Options{}) }

func TestOwnedProviderViewsKeepResourceAndSeparateRepresentations(t *testing.T) {
	fn, err := NewFuncProvider("candle", 768, func(context.Context, string) ([]float32, error) {
		t.Fatal("identity invoked inference")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	base := &identityViewProvider{FuncProvider: fn, descriptor: descriptorFixture()}
	set := NewSet(map[string]Provider{"mmbert": base}, "mmbert")
	full, _ := set.Get("mmbert", 0, 0)
	explicit, _ := set.Get("mmbert", base.descriptor.Dimension, base.descriptor.Layer)
	early, _ := set.Get("mmbert", 256, 6)
	if Identity(full) != Identity(explicit) {
		t.Fatal("equivalent zero/default options changed representation identity")
	}
	if Identity(full) == Identity(early) {
		t.Fatal("different depth/dimension shared representation identity")
	}
	for _, provider := range []Provider{full, explicit, early} {
		if provider.(*providerView).Provider != base {
			t.Fatal("view copied the physical provider")
		}
	}
	settings := ConsumerSettings{ModelType: "mmbert", Layer: 6, Dimension: 256, InputPolicy: "memory-v1"}
	identity, err := set.ResolveIdentity(settings)
	if err != nil {
		t.Fatal(err)
	}
	if identity.Descriptor.Layer != 6 || identity.Descriptor.Dimension != 256 {
		t.Fatal("consumer options were lost")
	}
	settings.InputPolicy = "cache-v1"
	another, err := set.ResolveIdentity(settings)
	if err != nil || another.Fingerprint == identity.Fingerprint {
		t.Fatal("consumer input policies shared persisted identity")
	}
	if base.calls == 0 {
		t.Fatal("owned descriptor was not consulted")
	}
}
