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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// defaultStopTimeout bounds a Stop call from a shutdown path that does not
// supply its own deadline. It keeps process shutdown responsive even when a
// backend or embedder is wedged.
const defaultStopTimeout = 30 * time.Second

// cleanupTimeout bounds the best-effort metadata/persistence work Stop performs
// outside the drain wait (failing queued jobs). It is derived from the caller's
// Stop deadline but capped so a wedged metadata registry cannot keep Stop
// running past its own contract.
const cleanupTimeout = 5 * time.Second

// lifecycleState tracks where a pipeline generation is in its lifecycle so Stop
// can report honestly and Start can refuse to reuse a generation that has not
// finished joining.
type lifecycleState int

const (
	// stateStopped means no generation is live: Start may create a new one.
	stateStopped lifecycleState = iota
	// stateRunning means the current generation is accepting and processing jobs.
	stateRunning
	// stateStopping means Stop was called but the current generation's workers
	// have not finished joining (e.g. a bounded Stop timed out with a wedged
	// worker still live). The generation's resources must not be reused until it
	// reaches stateStopped.
	stateStopping
)

// pipelineGeneration owns the per-Start resources whose lifetimes must not
// outlive a single running/stopping cycle. Giving each generation its own
// WaitGroup, queue, stop channel, and root context means a timed-out Stop that
// leaves old workers live can never race a subsequent Start: the new generation
// gets fresh resources, and the old one is joined independently.
type pipelineGeneration struct {
	id         uint64
	jobQueue   chan IngestionJob
	stopCh     chan struct{}
	wg         sync.WaitGroup
	rootCtx    context.Context
	rootCancel context.CancelFunc
	// done is closed once every worker in this generation has returned. A Stop
	// that timed out can be retried by waiting on this channel, and Start blocks
	// on it before creating the next generation.
	done chan struct{}
}

// IngestionPipeline processes file attachment jobs asynchronously.
// It reads files, extracts text, chunks, embeds, and stores the
// resulting vectors in the backend.
type IngestionPipeline struct {
	backend   VectorStoreBackend
	fileStore *FileStore
	manager   *Manager
	embedder  Embedder
	workers   int
	queueSize int

	// lifecycleMu serializes Start/Stop so generations are created and joined
	// one at a time. It is never held across unbounded I/O.
	lifecycleMu sync.Mutex

	mu           sync.RWMutex
	fileStatuses map[string]*VectorStoreFile // vsf_id -> status

	// state and gen are guarded by mu. gen is the current generation; callers
	// that need to enqueue or inspect the live queue read it under mu.
	state lifecycleState
	gen   *pipelineGeneration
	// genSeq monotonically numbers generations for diagnostics.
	genSeq uint64
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
		state:        stateStopped,
	}
}

// Start launches the worker goroutines for a fresh generation.
//
// If a previous generation is still stopping (a bounded Stop timed out with
// workers not yet joined), Start blocks until that generation has fully joined
// before creating the next one. This guarantees a new generation never shares a
// WaitGroup, queue, or root context with a still-live old generation.
//
// The wait for a stopping generation happens WITHOUT holding lifecycleMu, so a
// concurrent AttachFile or retry Stop is never blocked behind a Start that is
// itself parked on a wedged old generation. lifecycleMu is only held for the
// brief state check and generation creation.
func (p *IngestionPipeline) Start() {
	for {
		p.lifecycleMu.Lock()

		p.mu.Lock()
		state := p.state
		var prevDone chan struct{}
		if state == stateStopping && p.gen != nil {
			prevDone = p.gen.done
		}
		p.mu.Unlock()

		switch state {
		case stateRunning:
			// Already running: idempotent no-op.
			p.lifecycleMu.Unlock()
			return
		case stateStopping:
			// A previous generation is still unwinding. Release lifecycleMu before
			// waiting so attaches/Stops can proceed, then retry the transition.
			p.lifecycleMu.Unlock()
			if prevDone != nil {
				<-prevDone
			}
			continue
		}

		// stateStopped: create the next generation while holding lifecycleMu.
		p.mu.Lock()
		p.genSeq++
		gen := &pipelineGeneration{
			id:       p.genSeq,
			jobQueue: make(chan IngestionJob, p.queueSize),
			stopCh:   make(chan struct{}),
			done:     make(chan struct{}),
		}
		gen.rootCtx, gen.rootCancel = context.WithCancel(context.Background())
		p.gen = gen
		p.state = stateRunning
		p.mu.Unlock()

		for i := 0; i < p.workers; i++ {
			gen.wg.Add(1)
			go p.worker(gen)
		}
		// Reap this generation's workers so a later Start/Stop can observe join
		// completion via gen.done without holding any lock across the wait. Once
		// joined, self-heal the state: if this generation is still current and
		// stopping (a bounded Stop timed out and later released), move it to
		// stopped so the pipeline reflects reality without needing another call.
		go func() {
			gen.wg.Wait()
			close(gen.done)
			p.markGenerationStopped(gen)
		}()
		p.lifecycleMu.Unlock()
		return
	}
}

// Stop gracefully shuts down the current generation, bounded by ctx.
//
// It stops accepting new jobs, fails any still queued (bounded by a cleanup
// deadline so a wedged metadata registry cannot stall shutdown), and waits for
// in-flight jobs to drain. If ctx is cancelled or its deadline elapses before
// draining completes, the generation's root context is cancelled so in-flight
// jobs abort at their next checkpoint, and Stop returns ctx.Err() without
// blocking further. A nil ctx is treated as a bounded shutdown with
// defaultStopTimeout.
//
// Honest lifecycle reporting: if the drain times out, the generation stays in
// stateStopping (its workers are still live). A subsequent Stop does not return
// nil — it waits on the same generation's completion (bounded by its own ctx)
// and only reports success once the workers have actually joined. This lets a
// caller retry the join with a longer deadline instead of being told shutdown
// succeeded when it did not.
func (p *IngestionPipeline) Stop(ctx context.Context) error {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultStopTimeout)
		defer cancel()
	}

	p.mu.Lock()
	switch p.state {
	case stateStopped:
		// Nothing live to join.
		p.mu.Unlock()
		return nil
	case stateStopping:
		// A prior Stop timed out; this generation's workers are still unwinding.
		// Wait on its completion (bounded by ctx) rather than falsely reporting a
		// clean shutdown.
		gen := p.gen
		p.mu.Unlock()
		return p.awaitGeneration(ctx, gen)
	}

	// stateRunning: begin shutting this generation down.
	gen := p.gen
	p.state = stateStopping
	p.mu.Unlock()

	// Fail queued jobs under a bounded cleanup context so a stalled registry
	// cannot keep Stop running past its deadline. This runs before the drain
	// wait but is itself bounded, and lifecycleMu is not held across the
	// per-job persistence I/O inside failQueuedJobs (it uses the cleanup ctx).
	p.failQueuedJobs(ctx, gen, "pipeline_stopped", "ingestion pipeline stopped before processing job")
	close(gen.stopCh)

	// Wait for workers to drain, bounded by ctx. On timeout, cancel the root
	// context so in-flight jobs abort at their next stage checkpoint, then
	// return without waiting further — Stop must not block past ctx. The
	// generation remains in stateStopping until its workers join (observed via
	// gen.done by awaitGeneration/Start).
	//
	// Note: a job wedged *inside* a single stage (e.g. an embedder or backend
	// call that ignores context) cannot be interrupted here; that worker unwinds
	// only once the stage returns. Making individual stages ctx-aware is handled
	// by the follow-up embedder/backend context work. Stop's own contract —
	// returning within ctx — holds regardless.
	return p.awaitGeneration(ctx, gen)
}

// awaitGeneration waits for gen's workers to finish joining, bounded by ctx.
// On graceful completion it transitions to stateStopped and returns nil,
// letting any in-flight job run to completion. On ctx expiry it cancels the
// generation root context so in-flight jobs abort at their next checkpoint,
// then returns ctx.Err() while leaving the generation in stateStopping so a
// later call can retry the join.
func (p *IngestionPipeline) awaitGeneration(ctx context.Context, gen *pipelineGeneration) error {
	if gen == nil {
		return nil
	}

	select {
	case <-gen.done:
		// Workers drained gracefully; cancel the (now-unused) root to release its
		// resources. Cancellation here cannot abort any work — every worker has
		// already returned.
		if gen.rootCancel != nil {
			gen.rootCancel()
		}
		p.markGenerationStopped(gen)
		return nil
	case <-ctx.Done():
		// Deadline elapsed with workers still live. Signal in-flight jobs to abort
		// at their next checkpoint, but do not block further — Stop must return
		// within ctx. Stay in stateStopping so the caller can retry the join with
		// a longer deadline. Cancelling is idempotent, so repeated timed-out Stop
		// calls on the same generation are safe.
		if gen.rootCancel != nil {
			gen.rootCancel()
		}
		return ctx.Err()
	}
}

// markGenerationStopped transitions the pipeline to stateStopped if gen is still
// the current generation and its workers have joined. It is safe to call more
// than once for the same generation.
func (p *IngestionPipeline) markGenerationStopped(gen *pipelineGeneration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.gen == gen && p.state == stateStopping {
		p.state = stateStopped
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
	job := IngestionJob{
		VectorStoreFileID: vsfID,
		VectorStoreID:     vectorStoreID,
		FileID:            fileID,
		ChunkingStrategy:  strategy,
	}

	// Fast pre-check (no I/O): reject before reserving any count if the pipeline
	// is not currently running. This avoids an increment/compensate round-trip in
	// the common not-running case and, together with Start not holding
	// lifecycleMu while it waits out a stopping generation, ensures a reserved
	// count is always either consumed by a worker or compensated below — never
	// stranded behind a Start parked on a wedged old generation.
	if !p.isRunning() {
		return nil, fmt.Errorf("ingestion pipeline is not running")
	}

	// Increment the durable count BEFORE enqueueing and OUTSIDE lifecycleMu,
	// using a bounded context. Doing the increment first guarantees a worker can
	// never observe (and decrement for) this job before its in_progress count is
	// recorded, so the count never transiently goes negative. Doing it outside
	// lifecycleMu with a bounded context guarantees a slow/wedged metadata
	// registry can never keep AttachFile holding lifecycleMu and thereby block
	// Stop past its deadline (the P1-class hang, via the attach path).
	countCtx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
	defer cancel()
	_ = p.manager.UpdateFileCounts(countCtx, vectorStoreID, func(fc *FileCounts) {
		fc.InProgress++
		fc.Total++
	})

	snapshot := cloneVectorStoreFile(vsf)
	switch p.enqueueJob(vsfID, vsf, job) {
	case enqueueQueued:
		return snapshot, nil
	case enqueueNotRunning:
		// Incremented above but the pipeline is not running, so no worker will
		// process this job. Compensate the reservation (bounded, outside lock).
		p.compensateAttachCount(vectorStoreID)
		return nil, fmt.Errorf("ingestion pipeline is not running")
	default: // enqueueQueueFull
		// Move the count from in_progress to failed (bounded, outside lock). Net
		// effect over both updates: Total+1, Failed+1, InProgress 0.
		failCtx, failCancel := context.WithTimeout(context.Background(), cleanupTimeout)
		defer failCancel()
		_ = p.manager.UpdateFileCounts(failCtx, vectorStoreID, func(fc *FileCounts) {
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

// enqueueResult reports the outcome of the AttachFile enqueue critical section.
type enqueueResult int

const (
	enqueueQueued enqueueResult = iota
	enqueueNotRunning
	enqueueQueueFull
)

// enqueueJob runs the enqueue critical section under lifecycleMu: it captures
// the running generation, registers the in-memory status, and enqueues the job.
// Holding lifecycleMu makes the enqueue atomic with respect to Stop's queue
// drain, so a job can never land in a queue whose workers have already exited.
// No unbounded I/O happens under the lock.
func (p *IngestionPipeline) enqueueJob(vsfID string, vsf *VectorStoreFile, job IngestionJob) enqueueResult {
	p.lifecycleMu.Lock()
	defer p.lifecycleMu.Unlock()

	gen := p.runningGeneration()
	if gen == nil {
		return enqueueNotRunning
	}

	p.mu.Lock()
	p.fileStatuses[vsfID] = cloneVectorStoreFile(vsf)
	p.mu.Unlock()

	select {
	case gen.jobQueue <- job:
		return enqueueQueued
	default:
		p.setFileStatus(vsfID, "failed", &FileError{
			Code:    "queue_full",
			Message: "ingestion queue is full, try again later",
		})
		return enqueueQueueFull
	}
}

// compensateAttachCount reverses the in_progress/total increment made by
// AttachFile when the job could not be enqueued because the pipeline is not
// running. It uses a bounded context and is never called under lifecycleMu.
func (p *IngestionPipeline) compensateAttachCount(vectorStoreID string) {
	ctx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
	defer cancel()
	_ = p.manager.UpdateFileCounts(ctx, vectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Total--
	})
}

// runningGeneration returns the current generation only when the pipeline is in
// stateRunning, else nil. It takes p.mu itself; callers hold lifecycleMu so the
// returned generation cannot be swapped out before the caller enqueues onto it.
func (p *IngestionPipeline) runningGeneration() *pipelineGeneration {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.state != stateRunning {
		return nil
	}
	return p.gen
}

// isRunning reports whether the pipeline is currently in stateRunning. It is a
// lock-free-of-lifecycleMu snapshot used as a fast pre-check; the authoritative
// check happens under lifecycleMu in the enqueue critical section.
func (p *IngestionPipeline) isRunning() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.state == stateRunning
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

// worker is the background goroutine that processes ingestion jobs for a single
// generation. It reads only from that generation's queue and derives per-job
// work from that generation's root context, so it can never service a job that
// belongs to a newer generation.
func (p *IngestionPipeline) worker(gen *pipelineGeneration) {
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

	// Mark as completed. Commit the status and the durable count coherently:
	// once InsertChunks has succeeded the chunks are persisted, so the count
	// update must not be dropped just because shutdown cancelled the job ctx
	// between insert and persist. Use a detached, bounded cleanup context
	// (WithoutCancel + timeout) so the completed transition is recorded even
	// under a racing Stop, and surface a persistence failure instead of leaving
	// the durable count silently stale.
	p.setFileStatus(job.VectorStoreFileID, "completed", nil)
	persistCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
	defer cancel()
	if err := p.manager.UpdateFileCounts(persistCtx, job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Completed++
	}); err != nil {
		// The chunks are durably stored and the in-memory status is completed,
		// but the durable count could not be persisted. Surface it so the
		// inconsistency is observable rather than silently swallowed; the
		// in-memory count still reflects the completion for the running process.
		logging.Warnf(
			"vectorstore: chunks stored for file %s but completed-count persist failed: %v",
			job.VectorStoreFileID, err,
		)
	}
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

		embedding, err := p.embedder.Embed(chunk.Content)
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

// failJob marks a job as failed and updates file counts. The count update is
// detached from the (possibly cancelled) job context so the in-memory counts
// and durable metadata stay consistent with the failed status we just wrote,
// but bounded by cleanupTimeout so a wedged metadata registry cannot make a
// worker's failure path hang.
func (p *IngestionPipeline) failJob(ctx context.Context, job IngestionJob, code, message string) {
	p.setFileStatus(job.VectorStoreFileID, "failed", &FileError{
		Code:    code,
		Message: message,
	})
	persistCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
	defer cancel()
	_ = p.manager.UpdateFileCounts(persistCtx, job.VectorStoreID, func(fc *FileCounts) {
		fc.InProgress--
		fc.Failed++
	})
}

// failQueuedJobs drains the given generation's queue and marks each still-queued
// job failed, bounded by ctx. Every drained job's in-memory status is set to
// failed unconditionally — that is cheap (no I/O) and must happen so no queued
// job is left stuck in_progress after shutdown. Only the durable count
// persistence is bounded by ctx: queued jobs never produced durable chunks, so
// when the registry is wedged we honor the Stop deadline and skip the persist
// rather than blocking shutdown. The full queue is always drained (the loop
// exits only when the queue is empty), so no job is lost regardless of ctx.
// lifecycleMu is held by the caller, but the per-job persistence uses ctx (not
// an unbounded background context), so the lock is never held across unbounded
// I/O.
func (p *IngestionPipeline) failQueuedJobs(ctx context.Context, gen *pipelineGeneration, code, message string) {
	for {
		select {
		case job := <-gen.jobQueue:
			// Always record the in-memory failed status (no I/O) so a drained job
			// is never left stuck in_progress, even past the Stop deadline.
			p.setFileStatus(job.VectorStoreFileID, "failed", &FileError{
				Code:    code,
				Message: message,
			})
			// Bound the durable count persist by the Stop deadline (ctx) as well as
			// a per-write cap. When ctx is already expired, WithTimeout yields an
			// already-canceled context and UpdateFileCounts returns promptly; the
			// in-memory count is still decremented, only the durable write is
			// skipped. Honoring ctx takes priority — Stop must not block past it.
			persistCtx, cancel := context.WithTimeout(ctx, cleanupTimeout)
			_ = p.manager.UpdateFileCounts(persistCtx, job.VectorStoreID, func(fc *FileCounts) {
				fc.InProgress--
				fc.Failed++
			})
			cancel()
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
