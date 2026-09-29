package vectorstore

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestEmbeddingIdentityProtectsIngestionAndAllowsReembedding(t *testing.T) {
	ctx := context.Background()
	backend := NewMemoryBackend(MemoryBackendConfig{})
	registry := NewMemoryMetadataRegistry()
	old := NewManager(backend, registry, 3, BackendTypeMemory, WithEmbeddingIdentity("model-a"))
	historical, err := old.CreateStore(ctx, CreateStoreRequest{Name: "historical"})
	if err != nil {
		t.Fatal(err)
	}
	current := NewManager(backend, registry, 3, BackendTypeMemory, WithEmbeddingIdentity("model-b"))
	if err = current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	files, err := NewFileStore(t.TempDir(), NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatal(err)
	}
	file, err := files.Save("guide.txt", []byte("A retained source document for reembedding."), "assistants")
	if err != nil {
		t.Fatal(err)
	}
	embedder := &mockEmbedder{dim: 3}
	pipeline := NewIngestionPipeline(backend, files, current, embedder, PipelineConfig{Workers: 1, QueueSize: 10})
	pipeline.Start()
	t.Cleanup(func() { _ = pipeline.Stop(context.Background()) })
	if _, err = pipeline.AttachFile(historical.ID, file.ID, nil); !errors.Is(err, ErrEmbeddingIncompatible) {
		t.Fatalf("attachment used incompatible store: %v", err)
	}
	// A job accepted by an earlier runtime cannot bypass the worker-side guard.
	pipeline.fileStatuses["old-job"] = &VectorStoreFile{ID: "old-job", Status: "in_progress"}
	pipeline.processJob(ctx, IngestionJob{VectorStoreFileID: "old-job", VectorStoreID: historical.ID, FileID: file.ID})
	status, err := pipeline.GetFileStatus("old-job")
	if err != nil || status.Status != "failed" || status.LastError.Code != "embedding_incompatible" || embedder.called != 0 {
		t.Fatalf("worker attempted incompatible embedding: %+v, %v", status, err)
	}
	fresh, err := current.CreateStore(ctx, CreateStoreRequest{Name: "current"})
	if err != nil {
		t.Fatal(err)
	}
	attached, err := pipeline.AttachFile(fresh.ID, file.ID, nil)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		status, err = pipeline.GetFileStatus(attached.ID)
		if err != nil {
			t.Fatal(err)
		}
		if status.Status != "in_progress" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if status.Status != "completed" {
		t.Fatalf("reembedding failed: %+v", status)
	}
	if results, err := current.Search(ctx, fresh.ID, []float32{1, 1, 1}, 1, 0, nil); err != nil || len(results) != 1 || results[0].FileID != file.ID {
		t.Fatalf("reembedded source cannot be retrieved: %v, %v", results, err)
	}
	if _, err := files.Get(file.ID); err != nil {
		t.Fatalf("source file disappeared: %v", err)
	}
}
