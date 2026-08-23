package routingcontext

import (
	"context"
	"testing"
)

func TestGenerationContextRequiresCompleteImmutablePin(t *testing.T) {
	generation := Generation{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 2, SnapshotRevision: 7,
		RoutingDigest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	}
	bound, err := WithGeneration(context.Background(), generation)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := GenerationFrom(bound)
	if !ok || got != generation {
		t.Fatalf("generation = %#v, %v", got, ok)
	}

	generation.PublicationID = ""
	if _, err := WithGeneration(context.Background(), generation); err == nil {
		t.Fatal("partial generation was accepted")
	}
	if _, ok := GenerationFrom(context.Background()); ok {
		t.Fatal("empty context returned a generation")
	}
}
