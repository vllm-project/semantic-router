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
	"errors"
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

// cleanupTimeout bounds the detached context used to persist status and count
// updates for a job that is being torn down. Cleanup must outlive the job's own
// cancelled context so durable counts stay coherent, but it must not be
// unbounded: a stalled metadata registry would otherwise push Stop past its
// deadline.
const cleanupTimeout = 5 * time.Second

// pipelineState is the tri-state lifecycle of an IngestionPipeline.
//
// A Stop that exhausts its deadline leaves the pipeline in pipelineStopping
// rather than pipelineStopped, because its workers have not been reclaimed yet.
// Keeping that distinction is what lets a second Stop join the same generation
// instead of reporting a success that never happened.
type pipelineState int

const (
	pipelineStopped pipelineState = iota
	pipelineRunning
	pipelineStopping
)

// generation is the worker-owned state of a single Start/Stop cycle.
//
// Each Start creates a new generation, so a timed-out Stop cannot leak an old
// worker into a new one: the old generation keeps its own queue, WaitGroup and
// root context until its workers actually exit, and a job enqueued after the
// restart is only ever visible on the new generation's queue.
type generation struct {
	jobQueue   chan IngestionJob
	stopCh     chan struct{}
	rootCtx    context.Context
	rootCancel context.CancelFunc
	wg         sync.WaitGroup
	// drained is closed by a single watcher goroutine once every worker in this
	// generation has returned. Closing a channel rather than exposing wg.Wait
	// lets several concurrent Stop calls wait on the same generation under
	// their own deadlines.
	drained   chan struct{}
	stopOnce  sync.Once
	closeOnce sync.Once
}

// signalStop closes the generation's stop channel exactly once, so repeated
// Stop calls on the same generation do not panic.
func (g *generation) signalStop() {
	g.stopOnce.Do(func() { close(g.stopCh) })
}

// markDrained records that every worker in this generation has returned.
func (g *generation) markDrained() {
	g.closeOnce.Do(func() { close(g.drained) })
}

// hasDrained reports whether this generation's workers have all returned,
// without blocking.
func (g *generation) hasDrained() bool {
	select {
	case <-g.drained:
		return true
	default:
		return false
	}
}

// newGeneration allocates the per-Start worker state: its own queue, stop
// channel and lifecycle root context.
func newGeneration(queueSize int) *generation {
	ctx, cancel := context.WithCancel(context.Background())
	return &generation{
		jobQueue:   make(chan IngestionJob, queueSize),
		stopCh:     make(chan struct{}),
		rootCtx:    ctx,
		rootCancel: cancel,
		drained:    make(chan struct{}),
	}
}

// IngestionPipeline processes file attachment jobs asynchronously.
// It reads files, extracts text, chunks, embeds, and stores the
// resulting vectors in the backend.
type IngestionPipeline struct {
	backend      VectorStoreBackend
	fileStore    *FileStore
	manager      *Manager
	embedder     Embedder
	workers      int
	queueSize    int
	lifecycleMu  sync.Mutex
	mu           sync.RWMutex
	fileStatuses map[string]*VectorStoreFile // vsf_id -> status

	// state, current and draining are guarded by mu.
	state pipelineState
	// current is the generation installed by the most recent Start. It stays
	// set while that generation is stopping, so a second Stop after a timed-out
	// one joins the same generation instead of returning nil.
	current *generation
	// draining holds generations that a timed-out Stop left behind and that a
	// later Start displaced. They own their queue, WaitGroup and root context,
	// so their workers cannot observe a newer generation's jobs; a subsequent
	// Stop still joins them so shutdown accounts for every worker.
	draining []*generation
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
		workers:      workers,
		queueSize:    queueSize,
		fileStatuses: make(map[string]*VectorStoreFile),
		state:        pipelineStopped,
	}
}

// Start launches the worker goroutines for a new generation.
//
// A generation left behind by a timed-out Stop is carried forward rather than
// reused: it keeps its own queue, WaitGroup and root context, so one of its
// workers can never dequeue a job enqueued after this Start.
func (p *IngestionPipeline) Start() {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	p.mu.Lock()
	if p.state == pipelineRunning {
		p.mu.Unlock()
		return
	}

	// Reap generations that finished draining since the last transition and
	// carry the rest forward so a later Stop still joins them.
	carried := make([]*generation, 0, len(p.draining)+1)
	for _, g := range p.draining {
		if !g.hasDrained() {
			carried = append(carried, g)
		}
	}
	if p.current != nil && !p.current.hasDrained() {
		carried = append(carried, p.current)
	}
	p.draining = carried

	gen := newGeneration(p.queueSize)
	p.current = gen
	p.state = pipelineRunning
	p.mu.Unlock()

	for i := 0; i < p.workers; i++ {
		gen.wg.Add(1)
		go p.worker(gen)
	}
	// One watcher per generation publishes "all workers returned" as a closed
	// channel, so concurrent Stop calls can each wait under their own deadline.
	go func() {
		gen.wg.Wait()
		gen.markDrained()
	}()
}

// Stop gracefully shuts down the pipeline, bounded by ctx.
//
// It stops accepting new jobs, fails any still queued, and waits for in-flight
// jobs to drain. If ctx is cancelled or its deadline elapses before draining
// completes, every outstanding generation's root context is cancelled so
// in-flight jobs abort at their next checkpoint, and Stop returns ctx.Err()
// without blocking further. A nil ctx is treated as a bounded shutdown with
// defaultStopTimeout.
//
// Stop is idempotent, but only a Stop that actually observed every worker
// return reports success. A timed-out Stop leaves the pipeline in
// pipelineStopping, so a later Stop joins the same generation and reports its
// own outcome rather than a nil that the first call never earned.
func (p *IngestionPipeline) Stop(ctx context.Context) error {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	p.mu.Lock()
	if p.state == pipelineStopped {
		p.mu.Unlock()
		return nil
	}
	gen := p.current
	pending := make([]*generation, 0, len(p.draining)+1)
	if gen != nil {
		pending = append(pending, gen)
	}
	pending = append(pending, p.draining...)
	p.state = pipelineStopping
	p.mu.Unlock()

	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultStopTimeout)
		defer cancel()
	}

	// Only the current generation still accepts jobs, so it is the only one
	// with a queue left to fail. Older generations were sealed when they were
	// displaced.
	if gen != nil {
		p.failQueuedJobs(gen, "pipeline_stopped", "ingestion pipeline stopped before processing job")
	}
	for _, g := range pending {
		g.signalStop()
	}

	// Wait for every outstanding generation, bounded by ctx.
	//
	// Note: a job wedged *inside* a single stage (e.g. an embedder or backend
	// call that ignores context) cannot be interrupted here; that worker unwinds
	// only once the stage returns. Making individual stages ctx-aware is handled
	// by the follow-up embedder/backend context work. Stop's own contract —
	// returning within ctx — holds regardless.
	waitErr := joinGenerations(ctx, pending)

	// Cancel the roots either way: on a clean drain this releases the context
	// resources, on a timeout it tells wedged workers to abort at their next
	// checkpoint.
	for _, g := range pending {
		if g.rootCancel != nil {
			g.rootCancel()
		}
	}

	p.mu.Lock()
	var stillDraining []*generation
	for _, g := range pending {
		if !g.hasDrained() {
			stillDraining = append(stillDraining, g)
		}
	}
	if len(stillDraining) == 0 {
		p.state = pipelineStopped
		p.current = nil
		p.draining = nil
	} else {
		// Keep the generation a second Stop should join addressable as current;
		// anything older stays in draining.
		p.state = pipelineStopping
		p.current = nil
		p.draining = nil
		for _, g := range stillDraining {
			if g == gen {
				p.current = g
				continue
			}
			p.draining = append(p.draining, g)
		}
	}
	p.mu.Unlock()

	return waitErr
}

// joinGenerations waits for each generation's workers to return, bounded by
// ctx. It returns ctx.Err() as soon as the deadline elapses so Stop never
// blocks past its caller's bound.
func joinGenerations(ctx context.Context, gens []*generation) error {
	for _, g := range gens {
		select {
		case <-g.drained:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
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

	gen := p.runningGeneration()
	if gen == nil {
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
	case gen.jobQueue <- job:
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

// runningGeneration returns the generation currently accepting jobs, or nil if
// the pipeline is not running.
func (p *IngestionPipeline) runningGeneration() *generation {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.state != pipelineRunning {
		return nil
	}
	return p.current
}

// cleanupContext derives a detached, bounded context for persistence that must
// still run after ctx was cancelled. Detaching keeps durable counts coherent
// with the file status just written; the bound keeps a stalled metadata
// registry from pushing Stop past its own deadline.
func cleanupContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
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

// worker is the background goroutine that processes ingestion jobs. It is
// bound to one generation and only ever reads that generation's queue.
func (p *IngestionPipeline) worker(gen *generation) {
	defer gen.wg.Done()

	for {
		select {
		case <-gen.rootCtx.Done():
			return
		case <-gen.stopCh:
			return
		case job, ok := <-gen.jobQueue:
			if !ok {
				return
			}
			p.processJob(gen.rootCtx, job)
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

	// Mark as completed. The count update runs under a detached, bounded
	// context: a concurrent Stop may already have cancelled the job context, but
	// the durable counts must still match the status just written, and a stalled
	// registry must not hold Stop past its deadline.
	p.setFileStatus(job.VectorStoreFileID, "completed", nil)
	completedCtx, cancelCompleted := cleanupContext(ctx)
	defer cancelCompleted()
	_ = p.manager.UpdateFileCounts(completedCtx, job.VectorStoreID, func(fc *FileCounts) {
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
			// A cooperative embedder returns context.Canceled/DeadlineExceeded
			// when the lifecycle root is cancelled during shutdown. That is an
			// intentional termination, not a backend fault, so record it with
			// the "cancelled" code rather than misclassifying it as an
			// embedding error in status/metrics.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				p.failJob(ctx, job, "cancelled", fmt.Sprintf("ingestion cancelled while embedding chunk %d: %v", chunk.ChunkIndex, err))
				return nil, false
			}
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
	// Count updates use a detached, bounded context: even when the job context
	// is cancelled, the in-memory counts and durable metadata must stay
	// consistent with the file status we just wrote, without letting a stalled
	// registry hold shutdown open.
	cleanupCtx, cancel := cleanupContext(ctx)
	defer cancel()
	_ = p.manager.UpdateFileCounts(cleanupCtx, job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Failed++
	})
}

func (p *IngestionPipeline) failQueuedJobs(gen *generation, code, message string) {
	for {
		select {
		case job := <-gen.jobQueue:
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
