package embedding

import (
	"context"
	"errors"
	"testing"
)

type windowedEmbedder struct {
	windows []Window
	seen    []string
}

func (e *windowedEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	return []float32{float32(len(text))}, nil
}

func (e *windowedEmbedder) Windows(_ context.Context, _ string, _ int) ([]Window, error) {
	return e.windows, nil
}

// plainEmbedder implements Provider without Windows, the way a remote embedding
// service does.
type plainEmbedder struct{ seen []string }

func (e *plainEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	return []float32{1}, nil
}

func (e *plainEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func (e *plainEmbedder) Dimension() int { return 1 }

func (e *plainEmbedder) Backend() string { return "remote" }

type failingEmbedder struct{}

func (failingEmbedder) Embed(context.Context, string) ([]float32, error) {
	return nil, errors.New("embed failed")
}

func TestQueryVectorsEmbedsEveryWindow(t *testing.T) {
	query := "question at the front and the answer clue at the very end of it"
	embedder := &windowedEmbedder{windows: []Window{{Start: 0, End: 20}, {Start: 20, End: 40}, {Start: 40, End: len(query)}}}

	vectors, err := QueryVectors(context.Background(), embedder, query, DefaultQueryWindowLimit)
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 3 {
		t.Fatalf("embedded %d windows, want 3", len(vectors))
	}
	want := []string{query[0:20], query[20:40], query[40:]}
	for i, text := range want {
		if embedder.seen[i] != text {
			t.Errorf("window %d embedded %q, want %q", i, embedder.seen[i], text)
		}
	}
	if embedder.seen[len(embedder.seen)-1] == query {
		t.Error("a windowed query was also embedded whole")
	}
}

func TestQueryVectorsEmbedsAShortQueryOnce(t *testing.T) {
	embedder := &windowedEmbedder{windows: []Window{{Start: 0, End: 5}}}

	vectors, err := QueryVectors(context.Background(), embedder, "short", DefaultQueryWindowLimit)
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 1 || len(embedder.seen) != 1 || embedder.seen[0] != "short" {
		t.Fatalf("one window embedded %v as %v", embedder.seen, vectors)
	}
}

func TestQueryVectorsKeepsOneEmbeddingWithoutWindows(t *testing.T) {
	embedder := &plainEmbedder{}

	vectors, err := QueryVectors(context.Background(), embedder, "remote model owns its truncation", DefaultQueryWindowLimit)
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 1 || len(embedder.seen) != 1 {
		t.Fatalf("embedder without windows produced %d vectors from %d calls", len(vectors), len(embedder.seen))
	}
}

func TestQueryVectorsReportsEmbedFailure(t *testing.T) {
	if _, err := QueryVectors(context.Background(), failingEmbedder{}, "text", DefaultQueryWindowLimit); err == nil {
		t.Fatal("a failing embedder returned no error")
	}
	if _, err := QueryVectors(context.Background(), nil, "text", DefaultQueryWindowLimit); err == nil {
		t.Fatal("a missing embedder returned no error")
	}
}

func TestSampleWindowsKeepsBothEnds(t *testing.T) {
	windows := make([]Window, 47)
	for i := range windows {
		windows[i] = Window{Start: i * 10, End: i*10 + 10}
	}

	kept := SampleWindows(windows, 8)
	if len(kept) != 8 {
		t.Fatalf("kept %d windows, want 8", len(kept))
	}
	if kept[0] != windows[0] || kept[len(kept)-1] != windows[len(windows)-1] {
		t.Fatalf("sampled windows drop an end: %v", kept)
	}
	for i := 1; i < len(kept); i++ {
		if kept[i].Start <= kept[i-1].Start {
			t.Fatalf("windows are not in order at %d: %v then %v", i, kept[i-1], kept[i])
		}
	}
}

// failingWindowEmbedder has a tokenizer that breaks for a reason other than a
// missing capability, so the fallback must not hide it.
type failingWindowEmbedder struct{ calls int }

func (e *failingWindowEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.calls++
	return []float32{1}, nil
}

func (e *failingWindowEmbedder) Dimension() int { return 1 }

func (e *failingWindowEmbedder) Windows(context.Context, string, int) ([]Window, error) {
	return nil, errors.New("tokenizer exploded")
}

// A prepared provider always satisfies WindowProvider because Set.Get wraps it,
// so a provider without token windows only reveals that when Windows is called.
// This is the shape a remote embedding service takes at runtime.
func TestQueryVectorsKeepsOneEmbeddingForAPreparedRemoteProvider(t *testing.T) {
	remote := &plainEmbedder{}
	set := NewSet(map[string]Provider{"remote": remote}, "remote")
	prepared, err := set.Get("remote", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := prepared.(WindowProvider); !ok {
		t.Fatal("the prepared provider no longer satisfies WindowProvider, so this regression no longer covers the wrapper")
	}

	vectors, err := QueryVectors(context.Background(), prepared, "a query a remote model would truncate on its own", DefaultQueryWindowLimit)
	if err != nil {
		t.Fatalf("a prepared remote provider failed instead of embedding once: %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("prepared remote provider produced %d vectors, want 1", len(vectors))
	}
	if len(remote.seen) != 1 {
		t.Fatalf("prepared remote provider was asked to embed %d times, want 1", len(remote.seen))
	}
}

func TestQueryVectorsReportsARealTokenizerFailure(t *testing.T) {
	broken := &failingWindowEmbedder{}

	if _, err := QueryVectors(context.Background(), broken, "text", DefaultQueryWindowLimit); err == nil {
		t.Fatal("a broken tokenizer was treated as a missing capability")
	}
	if broken.calls != 0 {
		t.Fatalf("a broken tokenizer still fell back to embedding %d times", broken.calls)
	}
}
