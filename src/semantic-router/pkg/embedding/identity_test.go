package embedding

import (
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
