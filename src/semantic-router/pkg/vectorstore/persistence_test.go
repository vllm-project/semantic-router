package vectorstore

import (
	"context"
	"testing"
	"time"
)

func TestLocalMetadataRegistryRecoversVectorStoreMetadataAcrossRestart(t *testing.T) {
	tempDir := t.TempDir()
	registry, err := NewLocalMetadataRegistry(tempDir)
	if err != nil {
		t.Fatalf("NewLocalMetadataRegistry() error = %v", err)
	}

	backend := NewMemoryBackend(MemoryBackendConfig{})
	fileStore, err := NewFileStoreWithRegistry(tempDir, registry)
	if err != nil {
		t.Fatalf("NewFileStoreWithRegistry() error = %v", err)
	}
	manager := NewManagerWithRegistry(backend, 3, BackendTypeMemory, registry)
	embedder := &mockEmbedder{dim: 3}
	pipeline := NewIngestionPipelineWithRegistry(backend, fileStore, manager, embedder, registry, nil, PipelineConfig{
		Workers:   1,
		QueueSize: 10,
	})
	pipeline.Start()
	t.Cleanup(pipeline.Stop)

	ctx := context.Background()
	store, err := manager.CreateStore(ctx, CreateStoreRequest{Name: "persisted-store"})
	if err != nil {
		t.Fatalf("CreateStore() error = %v", err)
	}

	record, err := fileStore.Save("persisted.txt", []byte("Hello.\n\nWorld."), "assistants")
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	status, err := pipeline.AttachFile(store.ID, record.ID, nil)
	if err != nil {
		t.Fatalf("AttachFile() error = %v", err)
	}

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		current, getErr := pipeline.GetFileStatus(status.ID)
		if getErr == nil && current.Status == "completed" {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}

	restartedRegistry, err := NewLocalMetadataRegistry(tempDir)
	if err != nil {
		t.Fatalf("NewLocalMetadataRegistry(restart) error = %v", err)
	}
	restartedManager := NewManagerWithRegistry(backend, 3, BackendTypeMemory, restartedRegistry)
	if reconcileErr := restartedManager.ReconcilePersistedStores(ctx); reconcileErr != nil {
		t.Fatalf("ReconcilePersistedStores() error = %v", reconcileErr)
	}
	recoveredStatuses, err := restartedRegistry.RecoverInterruptedStatuses("router restart")
	if err != nil {
		t.Fatalf("RecoverInterruptedStatuses() error = %v", err)
	}
	if restoreErr := restartedManager.RestoreFileCounts(recoveredStatuses); restoreErr != nil {
		t.Fatalf("RestoreFileCounts() error = %v", restoreErr)
	}
	restartedStore, err := NewFileStoreWithRegistry(tempDir, restartedRegistry)
	if err != nil {
		t.Fatalf("NewFileStoreWithRegistry(restart) error = %v", err)
	}
	restartedPipeline := NewIngestionPipelineWithRegistry(
		backend,
		restartedStore,
		restartedManager,
		embedder,
		restartedRegistry,
		recoveredStatuses,
		PipelineConfig{},
	)

	gotStore, err := restartedManager.GetStore(store.ID)
	if err != nil {
		t.Fatalf("GetStore(restart) error = %v", err)
	}
	if gotStore.Name != "persisted-store" {
		t.Fatalf("store.Name = %q, want persisted-store", gotStore.Name)
	}
	if gotStore.FileCounts.Completed != 1 || gotStore.FileCounts.Total != 1 {
		t.Fatalf("file counts = %+v, want completed=1 total=1", gotStore.FileCounts)
	}

	gotFile, err := restartedStore.Get(record.ID)
	if err != nil {
		t.Fatalf("Get(file) after restart error = %v", err)
	}
	if gotFile.Filename != "persisted.txt" {
		t.Fatalf("file.Filename = %q, want persisted.txt", gotFile.Filename)
	}

	gotStatus, err := restartedPipeline.GetFileStatus(status.ID)
	if err != nil {
		t.Fatalf("GetFileStatus(restart) error = %v", err)
	}
	if gotStatus.Status != "completed" {
		t.Fatalf("status.Status = %q, want completed", gotStatus.Status)
	}
}

func TestLocalMetadataRegistryMarksInterruptedStatusesFailedOnRecovery(t *testing.T) {
	tempDir := t.TempDir()
	registry, err := NewLocalMetadataRegistry(tempDir)
	if err != nil {
		t.Fatalf("NewLocalMetadataRegistry() error = %v", err)
	}

	if upsertErr := registry.UpsertStatus(&VectorStoreFile{
		ID:            "vsf_interrupted",
		Object:        "vector_store.file",
		VectorStoreID: "vs_recovered",
		FileID:        "file_recovered",
		Status:        "in_progress",
		CreatedAt:     time.Now().Unix(),
	}); upsertErr != nil {
		t.Fatalf("UpsertStatus() error = %v", upsertErr)
	}

	statuses, err := registry.RecoverInterruptedStatuses("router restart interrupted ingestion")
	if err != nil {
		t.Fatalf("RecoverInterruptedStatuses() error = %v", err)
	}

	status := statuses["vsf_interrupted"]
	if status == nil {
		t.Fatal("expected recovered status entry")
	}
	if status.Status != "failed" {
		t.Fatalf("status.Status = %q, want failed", status.Status)
	}
	if status.LastError == nil || status.LastError.Code != restartInterruptedCode {
		t.Fatalf("LastError = %+v, want interrupted marker", status.LastError)
	}
}
