/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Embedder generates vector embeddings from text. Implementations
// wrap the actual embedding model (e.g. Candle FFI). Embed takes a
// context so a cancelled lifecycle or request aborts embedding work at
// the next checkpoint instead of running to completion.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dimension() int
}

// IngestionJob represents a file attachment job to be processed.
type IngestionJob struct {
	VectorStoreFileID string
	VectorStoreID     string
	FileID            string
	ChunkingStrategy  *ChunkingStrategy
}

// defaultStopTimeout bounds a Stop call from a shutdown path that does not
// supply its own deadline. It keeps process shutdown responsive even when a
// backend or embedder is wedged.
const defaultStopTimeout = 30 * time.Second

// IngestionPipeline processes file attachment jobs asynchronously.
// It reads files, extracts text, chunks, embeds, and stores the
// resulting vectors in the backend.
type IngestionPipeline struct {
	backend      VectorStoreBackend
	fileStore    *FileStore
	manager      *Manager
	embedder     Embedder
	jobQueue     chan IngestionJob
	workers      int
	lifecycleMu  sync.Mutex
	mu           sync.RWMutex
	fileStatuses map[string]*VectorStoreFile // vsf_id -> status
	wg           sync.WaitGroup
	stopCh       chan struct{}
	// rootCtx is the pipeline lifecycle context. All per-job work derives from
	// it, so cancelling rootCancel signals every in-flight job to abort at its
	// next checkpoint. It is (re)created on each Start.
	rootCtx    context.Context
	rootCancel context.CancelFunc
	running    bool
}

// PipelineConfig holds configuration for the ingestion pipeline.
type PipelineConfig struct {
	Workers   int // number of concurrent workers (default 2)
	QueueSize int // job queue buffer size (default 100)
}

// NewIngestionPipeline creates a new ingestion pipeline.
func NewIngestionPipeline(backend VectorStoreBackend, fileStore *FileStore, manager *Manager, embedder Embedder, cfg PipelineConfig) *IngestionPipeline {
	workers := cfg.Workers
	if workers <= 0 {
		workers = 2
	}
	queueSize := cfg.QueueSize
	if queueSize <= 0 {
		queueSize = 100
	}

	return &IngestionPipeline{
		backend:      backend,
		fileStore:    fileStore,
		manager:      manager,
		embedder:     embedder,
		jobQueue:     make(chan IngestionJob, queueSize),
		workers:      workers,
		fileStatuses: make(map[string]*VectorStoreFile),
		stopCh:       make(chan struct{}),
	}
}

// Start launches the worker goroutines.
func (p *IngestionPipeline) Start() {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	p.mu.Lock()
	if p.running {
		p.mu.Unlock()
		return
	}
	p.stopCh = make(chan struct{})
	stopCh := p.stopCh
	p.rootCtx, p.rootCancel = context.WithCancel(context.Background())
	rootCtx := p.rootCtx
	p.running = true
	p.mu.Unlock()

	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(rootCtx, stopCh)
	}
}

// Stop gracefully shuts down the pipeline, bounded by ctx.
//
// It stops accepting new jobs, fails any still queued, and waits for in-flight
// jobs to drain. If ctx is cancelled or its deadline elapses before draining
// completes, the pipeline root context is cancelled so in-flight jobs abort at
// their next checkpoint, and Stop returns ctx.Err() without blocking further.
// A nil ctx is treated as a bounded shutdown with defaultStopTimeout.
func (p *IngestionPipeline) Stop(ctx context.Context) error {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	p.mu.Lock()
	if !p.running {
		p.mu.Unlock()
		return nil
	}
	stopCh := p.stopCh
	rootCancel := p.rootCancel
	p.running = false
	p.mu.Unlock()

	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultStopTimeout)
		defer cancel()
	}

	p.failQueuedJobs("pipeline_stopped", "ingestion pipeline stopped before processing job")
	close(stopCh)

	// Wait for workers to drain, bounded by ctx. On timeout, cancel the root
	// context so in-flight jobs abort at their next stage checkpoint, then
	// return without waiting further — Stop must not block past ctx.
	//
	// Note: a job wedged *inside* a single stage (e.g. an embedder or backend
	// call that ignores context) cannot be interrupted here; that worker unwinds
	// only once the stage returns. Making individual stages ctx-aware is handled
	// by the follow-up embedder/backend context work. Stop's own contract —
	// returning within ctx — holds regardless.
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if rootCancel != nil {
			rootCancel()
		}
		return nil
	case <-ctx.Done():
		if rootCancel != nil {
			rootCancel()
		}
		return ctx.Err()
	}
}

// AttachFile queues a file for processing and returns the VectorStoreFile status.
func (p *IngestionPipeline) AttachFile(vectorStoreID, fileID string, strategy *ChunkingStrategy) (*VectorStoreFile, error) {
	// Verify the file exists.
	_, err := p.fileStore.Get(fileID)
	if err != nil {
		return nil, fmt.Errorf("file not found: %w", err)
	}

	// Verify the vector store exists.
	_, err = p.manager.GetStore(vectorStoreID)
	if err != nil {
		return nil, fmt.Errorf("vector store not found: %w", err)
	}

	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	if !p.isRunningLocked() {
		return nil, fmt.Errorf("ingestion pipeline is not running")
	}

	vsfID := GenerateVectorStoreFileID()
	vsf := &VectorStoreFile{
		ID:               vsfID,
		Object:           "vector_store.file",
		VectorStoreID:    vectorStoreID,
		FileID:           fileID,
		Status:           "in_progress",
		ChunkingStrategy: strategy,
		CreatedAt:        time.Now().Unix(),
	}

	p.mu.Lock()
	p.fileStatuses[vsfID] = cloneVectorStoreFile(vsf)
	p.mu.Unlock()

	// Update file counts.
	_ = p.manager.UpdateFileCounts(context.Background(), vectorStoreID, func(fc *FileCounts) {
		fc.InProgress++
		fc.Total++
	})

	job := IngestionJob{
		VectorStoreFileID: vsfID,
		VectorStoreID:     vectorStoreID,
		FileID:            fileID,
		ChunkingStrategy:  strategy,
	}

	// Snapshot before enqueuing so the caller always sees "in_progress",
	// even if the worker completes the job before we return.
	snapshot := cloneVectorStoreFile(vsf)

	select {
	case p.jobQueue <- job:
		return snapshot, nil
	default:
		// Queue is full.
		p.setFileStatus(vsfID, "failed", &FileError{
			Code:    "queue_full",
			Message: "ingestion queue is full, try again later",
		})
		_ = p.manager.UpdateFileCounts(context.Background(), vectorStoreID, func(fc *FileCounts) {
			fc.InProgress--
			fc.Failed++
		})
		status, err := p.GetFileStatus(vsfID)
		if err != nil {
			return cloneVectorStoreFile(vsf), nil
		}
		return status, nil
	}
}

func (p *IngestionPipeline) isRunningLocked() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.running
}

// GetFileStatus returns the current status of a vector store file.
func (p *IngestionPipeline) GetFileStatus(vsfID string) (*VectorStoreFile, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	vsf, ok := p.fileStatuses[vsfID]
	if !ok {
		return nil, fmt.Errorf("vector store file not found: %s", vsfID)
	}
	return cloneVectorStoreFile(vsf), nil
}

// ListFileStatuses returns all vector store files for a given vector store.
func (p *IngestionPipeline) ListFileStatuses(vectorStoreID string) []*VectorStoreFile {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var result []*VectorStoreFile
	for _, vsf := range p.fileStatuses {
		if vsf.VectorStoreID == vectorStoreID {
			result = append(result, cloneVectorStoreFile(vsf))
		}
	}
	return result
}

// DetachFile removes a file's chunks from the backend and updates status.
func (p *IngestionPipeline) DetachFile(ctx context.Context, vectorStoreID, vsfID string) error {
	p.mu.Lock()
	vsf, ok := p.fileStatuses[vsfID]
	if !ok {
		p.mu.Unlock()
		return fmt.Errorf("vector store file not found: %s", vsfID)
	}
	if vsf.VectorStoreID != vectorStoreID {
		p.mu.Unlock()
		return fmt.Errorf("vector store file %s does not belong to store %s", vsfID, vectorStoreID)
	}
	fileID := vsf.FileID
	status := vsf.Status
	delete(p.fileStatuses, vsfID)
	p.mu.Unlock()

	if err := p.backend.DeleteByFileID(ctx, vectorStoreID, fileID); err != nil {
		return fmt.Errorf("failed to delete chunks: %w", err)
	}

	_ = p.manager.UpdateFileCounts(context.Background(), vectorStoreID, func(fc *FileCounts) {
		switch status {
		case "completed":
			fc.Completed--
		case "in_progress":
			fc.InProgress--
		case "failed":
			fc.Failed--
		}
		fc.Total--
	})

	return nil
}

// worker is the background goroutine that processes ingestion jobs.
func (p *IngestionPipeline) worker(ctx context.Context, stopCh <-chan struct{}) {
	defer p.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-stopCh:
			return
		case job, ok := <-p.jobQueue:
			if !ok {
				return
			}
			p.processJob(ctx, job)
		}
	}
}

// processJob executes the full ingestion pipeline for a single file. It derives
// all backend work from ctx, and checks ctx between stages so a Stop that
// cancels the lifecycle context aborts the job promptly instead of running to
// completion.
func (p *IngestionPipeline) processJob(ctx context.Context, job IngestionJob) {
	if err := ctx.Err(); err != nil {
		p.failJob(ctx, job, "cancelled", "ingestion cancelled before start")
		return
	}

	// Step 1: Read file content.
	content, err := p.fileStore.Read(job.FileID)
	if err != nil {
		p.failJob(ctx, job, "read_error", fmt.Sprintf("failed to read file: %v", err))
		return
	}

	// Step 2: Get filename for parser.
	record, err := p.fileStore.Get(job.FileID)
	if err != nil {
		p.failJob(ctx, job, "metadata_error", fmt.Sprintf("failed to get file metadata: %v", err))
		return
	}

	// Step 3: Extract text.
	text, err := ExtractText(content, record.Filename)
	if err != nil {
		p.failJob(ctx, job, "parse_error", fmt.Sprintf("failed to extract text: %v", err))
		return
	}

	if err := ctx.Err(); err != nil {
		p.failJob(ctx, job, "cancelled", "ingestion cancelled before chunking")
		return
	}

	// Step 4: Chunk text.
	chunks := ChunkText(text, job.ChunkingStrategy)
	if len(chunks) == 0 {
		p.failJob(ctx, job, "empty_content", "file produced no text chunks")
		return
	}

	// Step 5: Embed each chunk.
	embeddedChunks, ok := p.embedChunks(ctx, job, record.Filename, chunks)
	if !ok {
		return
	}

	if err := ctx.Err(); err != nil {
		p.failJob(ctx, job, "cancelled", "ingestion cancelled before storage")
		return
	}

	// Step 6: Insert into backend.
	if err := p.backend.InsertChunks(ctx, job.VectorStoreID, embeddedChunks); err != nil {
		p.failJob(ctx, job, "storage_error", fmt.Sprintf("failed to store chunks: %v", err))
		return
	}

	// Mark as completed.
	p.setFileStatus(job.VectorStoreFileID, "completed", nil)
	_ = p.manager.UpdateFileCounts(ctx, job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Completed++
	})
}

// embedChunks embeds each chunk, checking ctx before each embedding call so a
// cancelled lifecycle context aborts promptly. On any error it fails the job
// and returns ok=false; the caller should stop processing.
func (p *IngestionPipeline) embedChunks(ctx context.Context, job IngestionJob, filename string, chunks []TextChunk) ([]EmbeddedChunk, bool) {
	embeddedChunks := make([]EmbeddedChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			p.failJob(ctx, job, "cancelled", fmt.Sprintf("ingestion cancelled before embedding chunk %d", chunk.ChunkIndex))
			return nil, false
		}

		embedding, err := p.embedder.Embed(ctx, chunk.Content)
		if err != nil {
			p.failJob(ctx, job, "embedding_error", fmt.Sprintf("failed to embed chunk %d: %v", chunk.ChunkIndex, err))
			return nil, false
		}

		embeddedChunks = append(embeddedChunks, EmbeddedChunk{
			ID:            fmt.Sprintf("%s_chunk_%d", job.FileID, chunk.ChunkIndex),
			FileID:        job.FileID,
			Filename:      filename,
			Content:       chunk.Content,
			Embedding:     embedding,
			ChunkIndex:    chunk.ChunkIndex,
			VectorStoreID: job.VectorStoreID,
		})
	}
	return embeddedChunks, true
}

// failJob marks a job as failed and updates file counts.
func (p *IngestionPipeline) failJob(ctx context.Context, job IngestionJob, code, message string) {
	p.setFileStatus(job.VectorStoreFileID, "failed", &FileError{
		Code:    code,
		Message: message,
	})
	// Count updates use a background context: even when the job context is
	// cancelled, the in-memory counts and durable metadata must stay consistent
	// with the file status we just wrote.
	_ = p.manager.UpdateFileCounts(context.WithoutCancel(ctx), job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Failed++
	})
}

func (p *IngestionPipeline) failQueuedJobs(code, message string) {
	for {
		select {
		case job := <-p.jobQueue:
			p.failJob(context.Background(), job, code, message)
		default:
			return
		}
	}
}

// setFileStatus updates the status and error of a vector store file.
func (p *IngestionPipeline) setFileStatus(vsfID, status string, lastError *FileError) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if vsf, ok := p.fileStatuses[vsfID]; ok {
		vsf.Status = status
		vsf.LastError = cloneFileError(lastError)
	}
}
