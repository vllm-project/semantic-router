package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func descriptorFixture() RuntimeDescriptor {
	return RuntimeDescriptor{Version: 1, ModelType: "mmbert", Runtime: "candle-f32-cpu-v2", EffectiveConfigSHA256: strings.Repeat("1", 64), TokenizerSHA256: strings.Repeat("2", 64), Artifacts: []ArtifactDigest{{Role: "weights", SHA256: strings.Repeat("3", 64)}}, Layer: 22, Dimension: 768, MaxSequenceLength: 32768, PoolingContract: "mean-f32-truncate-l2-v1"}
}

func identityForTest(t *testing.T, d RuntimeDescriptor, policy string) ContentIdentity {
	t.Helper()
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	identity, err := IdentityFromDescriptor(raw, policy)
	if err != nil {
		t.Fatal(err)
	}
	return identity
}

func TestContentIdentityTracksRepresentationAndConsumerPolicy(t *testing.T) {
	baseline := identityForTest(t, descriptorFixture(), "query:v1")
	tests := map[string]func(*RuntimeDescriptor){
		"weights":                 func(d *RuntimeDescriptor) { d.Artifacts[0].SHA256 = strings.Repeat("4", 64) },
		"tokenizer":               func(d *RuntimeDescriptor) { d.TokenizerSHA256 = strings.Repeat("4", 64) },
		"effective config":        func(d *RuntimeDescriptor) { d.EffectiveConfigSHA256 = strings.Repeat("4", 64) },
		"actual runtime fallback": func(d *RuntimeDescriptor) { d.Runtime = "onnx-cpu-fp32-v1" },
		"layer":                   func(d *RuntimeDescriptor) { d.Layer = 6 },
		"dimension":               func(d *RuntimeDescriptor) { d.Dimension = 256 },
		"execution budget": func(d *RuntimeDescriptor) {
			d.ExecutionPolicy = `{"MaxTokens":512,"Overflow":"reject","Precision":"float32"}`
		},
		"context": func(d *RuntimeDescriptor) { d.MaxSequenceLength = 512 },
		"pooling": func(d *RuntimeDescriptor) { d.PoolingContract = "another-contract" },
	}
	for name, modify := range tests {
		t.Run(name, func(t *testing.T) {
			d := descriptorFixture()
			modify(&d)
			if identityForTest(t, d, "query:v1").Fingerprint == baseline.Fingerprint {
				t.Fatal("changed representation retained identity")
			}
		})
	}
	if identityForTest(t, descriptorFixture(), "query:v2").Fingerprint == baseline.Fingerprint {
		t.Fatal("changed input policy retained identity")
	}
}

func TestContentIdentityCanonicalArtifactsAndRejectsUnverified(t *testing.T) {
	d := descriptorFixture()
	d.Artifacts = append(d.Artifacts, ArtifactDigest{Role: "graph", SHA256: strings.Repeat("5", 64)})
	first := identityForTest(t, d, "query:v1")
	d.Artifacts[0], d.Artifacts[1] = d.Artifacts[1], d.Artifacts[0]
	if identityForTest(t, d, "query:v1").Fingerprint != first.Fingerprint {
		t.Fatal("artifact order changed identity")
	}
	d.Artifacts[0].SHA256 = "invalid"
	raw, _ := json.Marshal(d)
	if _, err := IdentityFromDescriptor(raw, "query:v1"); err == nil {
		t.Fatal("accepted missing digest")
	}
	raw, _ = json.Marshal(descriptorFixture())
	if _, err := IdentityFromDescriptor(raw, ""); err == nil {
		t.Fatal("accepted unspecified input policy")
	}
	for _, provider := range []string{"qwen3", "openai", "remote", "bert"} {
		if _, err := ResolveProviderIdentity(nil, ConsumerSettings{ModelType: provider}); !errors.Is(err, ErrIdentityUnsupported) {
			t.Fatalf("%s: %v", provider, err)
		}
	}
}

func TestNamespaceIdentityKeysCandleBERTByEncoderVersion(t *testing.T) {
	embed := func(context.Context, string) ([]float32, error) {
		return nil, errors.New("identity resolution ran inference")
	}
	candle, err := NewFuncProvider("candle", 384, embed)
	if err != nil {
		t.Fatal(err)
	}
	ort, err := NewFuncProvider("ort", 384, embed)
	if err != nil {
		t.Fatal(err)
	}
	memory := ConsumerSettings{ModelType: "bert", InputPolicy: "memory-content-v1"}
	identity, err := ResolveNamespaceIdentity(candle, memory)
	if err != nil || identity.Descriptor.Dimension != 384 || !strings.HasPrefix(identity.Fingerprint, candleBERTNamespace+":") {
		t.Fatalf("Candle BERT namespace: %+v %v", identity, err)
	}
	cache := memory
	cache.InputPolicy = "response-cache-v1"
	if other, _ := ResolveNamespaceIdentity(candle, cache); other.Fingerprint == identity.Fingerprint {
		t.Fatal("memory and cache inputs shared a namespace")
	}
	if _, err = ResolveNamespaceIdentity(candle, ConsumerSettings{ModelType: "bert", Dimension: 256, InputPolicy: "memory-content-v1"}); err == nil {
		t.Fatal("namespace accepted a width the provider does not produce")
	}
	for _, provider := range []Provider{nil, ort} {
		if _, err = ResolveNamespaceIdentity(provider, memory); !errors.Is(err, ErrIdentityUnsupported) {
			t.Fatalf("unchanged BERT runtime acquired a namespace: %v", err)
		}
	}
	if _, err = ResolveNamespaceIdentity(candle, ConsumerSettings{ModelType: "mmbert", InputPolicy: "memory-content-v1"}); !errors.Is(err, ErrIdentityUnsupported) {
		t.Fatalf("other models must keep requiring a content descriptor: %v", err)
	}
}

func TestOmniContentIdentityIncludesModelProcessors(t *testing.T) {
	descriptor := descriptorFixture()
	descriptor.ModelType = "vela_omni"
	descriptor.Layer = 0
	descriptor.Dimension = 384
	descriptor.PoolingContract = "cls-l2"
	descriptor.Artifacts = append(descriptor.Artifacts, ArtifactDigest{Role: "processor_manifest", SHA256: strings.Repeat("5", 64)})
	baseline := identityForTest(t, descriptor, "omni-input-v1")
	descriptor.Artifacts[1].SHA256 = strings.Repeat("6", 64)
	if identityForTest(t, descriptor, "omni-input-v1").Fingerprint == baseline.Fingerprint {
		t.Fatal("processor change retained the old vector namespace")
	}
}
