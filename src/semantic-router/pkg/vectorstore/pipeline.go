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
// wrap the actual embedding model (e.g. Candle FFI).
type Embedder interface {
	Embed(text string) ([]float32, error)
	Dimension() int
}

// IngestionJob represents a file attachment job to be processed.
type IngestionJob struct {
	VectorStoreFileID string
	VectorStoreID     string
	FileID            string
	ChunkingStrategy  *ChunkingStrategy
}

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
	mu           sync.RWMutex
	fileStatuses map[string]*VectorStoreFile // vsf_id -> status
	wg           sync.WaitGroup
	stopCh       chan struct{}
	running      bool
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
	p.mu.Lock()
	if p.running {
		p.mu.Unlock()
		return
	}
	p.running = true
	p.mu.Unlock()

	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker()
	}
}

// Stop gracefully shuts down the pipeline.
func (p *IngestionPipeline) Stop() {
	p.mu.Lock()
	if !p.running {
		p.mu.Unlock()
		return
	}
	p.running = false
	p.mu.Unlock()

	close(p.stopCh)
	p.wg.Wait()
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
	p.fileStatuses[vsfID] = vsf
	p.mu.Unlock()

	// Update file counts.
	_ = p.manager.UpdateFileCounts(vectorStoreID, func(fc *FileCounts) {
		fc.InProgress++
		fc.Total++
	})

	job := IngestionJob{
		VectorStoreFileID: vsfID,
		VectorStoreID:     vectorStoreID,
		FileID:            fileID,
		ChunkingStrategy:  strategy,
	}

	select {
	case p.jobQueue <- job:
		return vsf, nil
	default:
		// Queue is full.
		p.setFileStatus(vsfID, "failed", &FileError{
			Code:    "queue_full",
			Message: "ingestion queue is full, try again later",
		})
		_ = p.manager.UpdateFileCounts(vectorStoreID, func(fc *FileCounts) {
			fc.InProgress--
			fc.Failed++
		})
		return vsf, nil
	}
}

// GetFileStatus returns the current status of a vector store file.
func (p *IngestionPipeline) GetFileStatus(vsfID string) (*VectorStoreFile, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	vsf, ok := p.fileStatuses[vsfID]
	if !ok {
		return nil, fmt.Errorf("vector store file not found: %s", vsfID)
	}
	return vsf, nil
}

// ListFileStatuses returns all vector store files for a given vector store.
func (p *IngestionPipeline) ListFileStatuses(vectorStoreID string) []*VectorStoreFile {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var result []*VectorStoreFile
	for _, vsf := range p.fileStatuses {
		if vsf.VectorStoreID == vectorStoreID {
			result = append(result, vsf)
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
	delete(p.fileStatuses, vsfID)
	p.mu.Unlock()

	if err := p.backend.DeleteByFileID(ctx, vectorStoreID, fileID); err != nil {
		return fmt.Errorf("failed to delete chunks: %w", err)
	}

	_ = p.manager.UpdateFileCounts(vectorStoreID, func(fc *FileCounts) {
		switch vsf.Status {
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
func (p *IngestionPipeline) worker() {
	defer p.wg.Done()

	for {
		select {
		case <-p.stopCh:
			return
		case job, ok := <-p.jobQueue:
			if !ok {
				return
			}
			p.processJob(job)
		}
	}
}

// processJob executes the full ingestion pipeline for a single file.
func (p *IngestionPipeline) processJob(job IngestionJob) {
	ctx := context.Background()

	// Step 1: Read file content.
	content, err := p.fileStore.Read(job.FileID)
	if err != nil {
		p.failJob(job, "read_error", fmt.Sprintf("failed to read file: %v", err))
		return
	}

	// Step 2: Get filename for parser.
	record, err := p.fileStore.Get(job.FileID)
	if err != nil {
		p.failJob(job, "metadata_error", fmt.Sprintf("failed to get file metadata: %v", err))
		return
	}

	// Step 3: Extract text.
	text, err := ExtractText(content, record.Filename)
	if err != nil {
		p.failJob(job, "parse_error", fmt.Sprintf("failed to extract text: %v", err))
		return
	}

	// Step 4: Chunk text.
	chunks := ChunkText(text, job.ChunkingStrategy)
	if len(chunks) == 0 {
		p.failJob(job, "empty_content", "file produced no text chunks")
		return
	}

	// Step 5: Embed each chunk.
	embeddedChunks := make([]EmbeddedChunk, 0, len(chunks))
	for _, chunk := range chunks {
		embedding, err := p.embedder.Embed(chunk.Content)
		if err != nil {
			p.failJob(job, "embedding_error", fmt.Sprintf("failed to embed chunk %d: %v", chunk.ChunkIndex, err))
			return
		}

		ec := EmbeddedChunk{
			ID:            fmt.Sprintf("%s_chunk_%d", job.FileID, chunk.ChunkIndex),
			FileID:        job.FileID,
			Filename:      record.Filename,
			Content:       chunk.Content,
			Embedding:     embedding,
			ChunkIndex:    chunk.ChunkIndex,
			VectorStoreID: job.VectorStoreID,
		}
		embeddedChunks = append(embeddedChunks, ec)
	}

	// Step 6: Insert into backend.
	if err := p.backend.InsertChunks(ctx, job.VectorStoreID, embeddedChunks); err != nil {
		p.failJob(job, "storage_error", fmt.Sprintf("failed to store chunks: %v", err))
		return
	}

	// Mark as completed.
	p.setFileStatus(job.VectorStoreFileID, "completed", nil)
	_ = p.manager.UpdateFileCounts(job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Completed++
	})
}

// failJob marks a job as failed and updates file counts.
func (p *IngestionPipeline) failJob(job IngestionJob, code, message string) {
	p.setFileStatus(job.VectorStoreFileID, "failed", &FileError{
		Code:    code,
		Message: message,
	})
	_ = p.manager.UpdateFileCounts(job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Failed++
	})
}

// setFileStatus updates the status and error of a vector store file.
func (p *IngestionPipeline) setFileStatus(vsfID, status string, lastError *FileError) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if vsf, ok := p.fileStatuses[vsfID]; ok {
		vsf.Status = status
		vsf.LastError = lastError
	}
}
