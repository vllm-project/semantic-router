package serving

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type noCoverageEmbeddingServices struct{ *embedServices }

func (f noCoverageEmbeddingServices) Embed(ctx context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	response, err := f.embedServices.Embed(ctx, deployment, request)
	response.Inputs = nil
	return response, err
}

func TestEmbeddingMissingCoverageRemainsUnknownInCache(t *testing.T) {
	services := noCoverageEmbeddingServices{&embedServices{cards: map[string]modelservice.ModelCard{"emb-unknown": embeddingCard("emb-unknown", "text")}}}
	ctx := context.Background()
	provider, err := New(services, nil).Embedding(ctx, embeddingSpec("emb-unknown"), 4, 6)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	before := len(services.embedCalls())
	for range 2 {
		if fits, checkErr := provider.FitsInput(ctx, "complete query"); fits || !errors.Is(checkErr, binding.ErrCapability) {
			t.Fatalf("missing input usage: fits=%v err=%v", fits, checkErr)
		}
	}
	vector, err := provider.Embed(ctx, "complete query")
	if err != nil || len(vector) != 4 {
		t.Fatalf("ordinary embedding: %v %v", vector, err)
	}
	if got := len(services.embedCalls()) - before; got != 1 {
		t.Fatalf("cached coverage made %d calls, want 1", got)
	}
}
