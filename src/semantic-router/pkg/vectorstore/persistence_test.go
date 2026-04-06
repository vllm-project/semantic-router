package vectorstore

import (
	"context"
	"testing"
	"time"
)

type localMetadataRuntimeFixture struct {
	registry  *LocalMetadataRegistry
	backend   VectorStoreBackend
	fileStore *FileStore
	manager   *Manager
	embedder  *mockEmbedder
	pipeline  *IngestionPipeline
}

func TestLocalMetadataRegistryRecoversVectorStoreMetadataAcrossRestart(t *testing.T) {
	tempDir := t.TempDir()
	ctx := context.Background()
	runtime := newLocalMetadataRuntimeFixture(t, tempDir)
	store := mustCreateVectorStore(t, ctx, runtime.manager, "persisted-store")
	record := mustSaveVectorStoreFile(t, runtime.fileStore, "persisted.txt", "Hello.\n\nWorld.")
	status := mustAttachVectorStoreFile(t, runtime.pipeline, store.ID, record.ID)
	waitForVectorStoreFileStatus(t, runtime.pipeline, status.ID, "completed")

	restarted := restartLocalMetadataRuntimeFixture(
		t,
		tempDir,
		ctx,
		runtime.backend,
		runtime.embedder,
	)
	assertRecoveredVectorStoreMetadata(
		t,
		restarted,
		store.ID,
		record.ID,
		status.ID,
	)
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

func newLocalMetadataRuntimeFixture(
	t *testing.T, baseDir string,
) localMetadataRuntimeFixture {
	t.Helper()

	registry := mustNewLocalMetadataRegistry(t, baseDir)
	backend := NewMemoryBackend(MemoryBackendConfig{})
	fileStore := mustNewFileStoreWithRegistry(t, baseDir, registry)
	manager := NewManagerWithRegistry(backend, 3, BackendTypeMemory, registry)
	embedder := &mockEmbedder{dim: 3}
	pipeline := NewIngestionPipelineWithRegistry(
		backend,
		fileStore,
		manager,
		embedder,
		registry,
		nil,
		PipelineConfig{
			Workers:   1,
			QueueSize: 10,
		},
	)
	pipeline.Start()
	t.Cleanup(pipeline.Stop)

	return localMetadataRuntimeFixture{
		registry:  registry,
		backend:   backend,
		fileStore: fileStore,
		manager:   manager,
		embedder:  embedder,
		pipeline:  pipeline,
	}
}

func restartLocalMetadataRuntimeFixture(
	t *testing.T,
	baseDir string,
	ctx context.Context,
	backend VectorStoreBackend,
	embedder *mockEmbedder,
) localMetadataRuntimeFixture {
	t.Helper()

	registry := mustNewLocalMetadataRegistry(t, baseDir)
	manager := NewManagerWithRegistry(backend, 3, BackendTypeMemory, registry)
	if err := manager.ReconcilePersistedStores(ctx); err != nil {
		t.Fatalf("ReconcilePersistedStores() error = %v", err)
	}

	recoveredStatuses, err := registry.RecoverInterruptedStatuses("router restart")
	if err != nil {
		t.Fatalf("RecoverInterruptedStatuses() error = %v", err)
	}
	if err := manager.RestoreFileCounts(recoveredStatuses); err != nil {
		t.Fatalf("RestoreFileCounts() error = %v", err)
	}

	fileStore := mustNewFileStoreWithRegistry(t, baseDir, registry)
	pipeline := NewIngestionPipelineWithRegistry(
		backend,
		fileStore,
		manager,
		embedder,
		registry,
		recoveredStatuses,
		PipelineConfig{},
	)

	return localMetadataRuntimeFixture{
		registry:  registry,
		backend:   backend,
		fileStore: fileStore,
		manager:   manager,
		embedder:  embedder,
		pipeline:  pipeline,
	}
}

func waitForVectorStoreFileStatus(
	t *testing.T,
	pipeline *IngestionPipeline,
	statusID string,
	wantStatus string,
) {
	t.Helper()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		current, err := pipeline.GetFileStatus(statusID)
		if err == nil && current.Status == wantStatus {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}

	current, err := pipeline.GetFileStatus(statusID)
	if err != nil {
		t.Fatalf("GetFileStatus(%s) error = %v", statusID, err)
	}
	t.Fatalf("status.Status = %q, want %q", current.Status, wantStatus)
}

func assertRecoveredVectorStoreMetadata(
	t *testing.T,
	runtime localMetadataRuntimeFixture,
	storeID string,
	recordID string,
	statusID string,
) {
	t.Helper()

	gotStore, err := runtime.manager.GetStore(storeID)
	if err != nil {
		t.Fatalf("GetStore(restart) error = %v", err)
	}
	if gotStore.Name != "persisted-store" {
		t.Fatalf("store.Name = %q, want persisted-store", gotStore.Name)
	}
	if gotStore.FileCounts.Completed != 1 || gotStore.FileCounts.Total != 1 {
		t.Fatalf("file counts = %+v, want completed=1 total=1", gotStore.FileCounts)
	}

	gotFile, err := runtime.fileStore.Get(recordID)
	if err != nil {
		t.Fatalf("Get(file) after restart error = %v", err)
	}
	if gotFile.Filename != "persisted.txt" {
		t.Fatalf("file.Filename = %q, want persisted.txt", gotFile.Filename)
	}

	gotStatus, err := runtime.pipeline.GetFileStatus(statusID)
	if err != nil {
		t.Fatalf("GetFileStatus(restart) error = %v", err)
	}
	if gotStatus.Status != "completed" {
		t.Fatalf("status.Status = %q, want completed", gotStatus.Status)
	}
}

func mustNewLocalMetadataRegistry(
	t *testing.T, baseDir string,
) *LocalMetadataRegistry {
	t.Helper()

	registry, err := NewLocalMetadataRegistry(baseDir)
	if err != nil {
		t.Fatalf("NewLocalMetadataRegistry() error = %v", err)
	}
	return registry
}

func mustNewFileStoreWithRegistry(
	t *testing.T,
	baseDir string,
	registry *LocalMetadataRegistry,
) *FileStore {
	t.Helper()

	fileStore, err := NewFileStoreWithRegistry(baseDir, registry)
	if err != nil {
		t.Fatalf("NewFileStoreWithRegistry() error = %v", err)
	}
	return fileStore
}

func mustCreateVectorStore(
	t *testing.T,
	ctx context.Context,
	manager *Manager,
	name string,
) *VectorStore {
	t.Helper()

	store, err := manager.CreateStore(ctx, CreateStoreRequest{Name: name})
	if err != nil {
		t.Fatalf("CreateStore() error = %v", err)
	}
	return store
}

func mustSaveVectorStoreFile(
	t *testing.T,
	fileStore *FileStore,
	filename string,
	content string,
) *FileRecord {
	t.Helper()

	record, err := fileStore.Save(filename, []byte(content), "assistants")
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	return record
}

func mustAttachVectorStoreFile(
	t *testing.T,
	pipeline *IngestionPipeline,
	storeID string,
	recordID string,
) *VectorStoreFile {
	t.Helper()

	status, err := pipeline.AttachFile(storeID, recordID, nil)
	if err != nil {
		t.Fatalf("AttachFile() error = %v", err)
	}
	return status
}
