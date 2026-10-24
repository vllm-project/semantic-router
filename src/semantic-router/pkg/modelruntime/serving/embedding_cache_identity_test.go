package serving

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func resetEmbeddingVectorCache(t *testing.T) {
	t.Helper()
	previous := sharedVectors
	sharedVectors = embedding.NewVectorCache(1 << 20)
	t.Cleanup(func() { sharedVectors = previous })
}

func TestEmbeddingCacheSeparatesUnhashedModels(t *testing.T) {
	resetEmbeddingVectorCache(t)
	a, b := embeddingCard("model-a", "text"), embeddingCard("model-b", "text")
	a.Ready, b.Ready = true, true
	// Attached cards may omit the optional content digest and revision.
	a.ModelSHA256, b.ModelSHA256, a.Revision, b.Revision = "", "", "", ""
	a.Embedding.Dimensions, b.Embedding.Dimensions = []int{8}, []int{4}
	f := &embedServices{cards: map[string]modelservice.ModelCard{"model-a": a, "model-b": b}}
	r := New(f, nil)
	ctx := context.Background()
	pa, err := r.Embedding(ctx, embeddingSpec("model-a"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer pa.Close()
	pb, err := r.Embedding(ctx, embeddingSpec("model-b"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer pb.Close()
	text := "unhashed-model-cache"
	va, err := pa.Embed(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	before := len(f.embedCalls())
	vb, err := pb.Embed(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model-a dimension=%d returned=%d; model-b dimension=%d returned=%d; model-b additional calls=%d",
		pa.Dimension(), len(va), pb.Dimension(), len(vb), len(f.embedCalls())-before)
	if len(vb) != pb.Dimension() {
		t.Fatalf("model-b received model-a's cached vector: got %d dimensions, want %d", len(vb), pb.Dimension())
	}
	if calls := len(f.embedCalls()) - before; calls != 1 {
		t.Fatalf("model-b made %d backend calls, want 1", calls)
	}
}

func TestEmbeddingCacheIdentityAcrossPreparations(t *testing.T) {
	for _, tc := range []struct {
		name     string
		unhashed bool
		revision string
	}{
		{name: "hashed"},
		{name: "unhashed", unhashed: true},
		{name: "unhashed_with_revision", unhashed: true, revision: "same-revision"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetEmbeddingVectorCache(t)
			card := embeddingCard("same-model", "text")
			card.Ready, card.Revision = true, tc.revision
			if tc.unhashed {
				card.ModelSHA256 = ""
			}
			services := &embedServices{cards: map[string]modelservice.ModelCard{card.ID: card}}
			ctx := context.Background()
			a, err := New(services, nil).Embedding(ctx, embeddingSpec(card.ID), 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			b, err := New(services, nil).Embedding(ctx, embeddingSpec(card.ID), 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Close()
			for _, options := range []embedding.Options{{}, {Dimension: 4, Layer: 6}} {
				identity := a.CacheIdentityForOptions(options)
				if identity == "" || identity != a.CacheIdentityForOptions(options) {
					t.Fatal("cache identity is empty or unstable")
				}
				if same := identity == b.CacheIdentityForOptions(options); same == tc.unhashed {
					t.Fatalf("unhashed=%v: providers have matching cache identities=%v for %+v", tc.unhashed, same, options)
				}
			}
			if a.CacheIdentity() != a.CacheIdentityForOptions(embedding.Options{}) || b.CacheIdentity() != b.CacheIdentityForOptions(embedding.Options{}) {
				t.Fatal("default cache identity differs from the default view's identity")
			}
			text := t.Name()
			if _, err := a.Embed(ctx, text); err != nil {
				t.Fatal(err)
			}
			before := len(services.embedCalls())
			if _, err := b.Embed(ctx, text); err != nil {
				t.Fatal(err)
			}
			want := 0
			if tc.unhashed {
				want = 1
			}
			calls := len(services.embedCalls()) - before
			t.Logf("second provider additional calls=%d", calls)
			if calls != want {
				t.Fatalf("second provider made %d backend calls, want %d", calls, want)
			}
			before = len(services.embedCalls())
			if _, err := b.Embed(ctx, text); err != nil {
				t.Fatal(err)
			}
			if len(services.embedCalls()) != before {
				t.Fatal("provider did not reuse its own cached vector")
			}
		})
	}
}
