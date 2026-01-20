package store

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// WriteOpType represents the type of write operation.
type WriteOpType int

const (
	// OpStore stores a new record.
	OpStore WriteOpType = iota
	// OpUpdate updates an existing record.
	OpUpdate
	// OpDelete deletes a record.
	OpDelete
)

// WriteOperation represents an async write operation.
type WriteOperation struct {
	Type   WriteOpType
	Record *RoutingRecord
}

// AsyncWriter handles asynchronous writes to the replay store.
// It buffers write operations and processes them in batches
// to avoid blocking request processing.
type AsyncWriter struct {
	store         ReplayStore
	writeChan     chan WriteOperation
	batchSize     int
	flushInterval time.Duration
	workers       int

	wg      sync.WaitGroup
	done    chan struct{}
	mu      sync.Mutex
	running bool
}

// AsyncWriterConfig configures the async writer.
type AsyncWriterConfig struct {
	// BufferSize is the channel buffer size for write operations.
	// Default: 1000
	BufferSize int

	// BatchSize is the number of operations to batch before writing.
	// Default: 10
	BatchSize int

	// FlushIntervalMs is the maximum time in milliseconds to wait before flushing.
	// Default: 100
	FlushIntervalMs int

	// Workers is the number of worker goroutines.
	// Default: 2
	Workers int
}

// DefaultAsyncWriterConfig returns the default async writer configuration.
func DefaultAsyncWriterConfig() AsyncWriterConfig {
	return AsyncWriterConfig{
		BufferSize:      1000,
		BatchSize:       10,
		FlushIntervalMs: 100,
		Workers:         2,
	}
}

// NewAsyncWriter creates a new async writer.
func NewAsyncWriter(store ReplayStore, config AsyncWriterConfig) *AsyncWriter {
	if config.BufferSize <= 0 {
		config.BufferSize = 1000
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 10
	}
	if config.FlushIntervalMs <= 0 {
		config.FlushIntervalMs = 100
	}
	if config.Workers <= 0 {
		config.Workers = 2
	}

	return &AsyncWriter{
		store:         store,
		writeChan:     make(chan WriteOperation, config.BufferSize),
		batchSize:     config.BatchSize,
		flushInterval: time.Duration(config.FlushIntervalMs) * time.Millisecond,
		workers:       config.Workers,
		done:          make(chan struct{}),
	}
}

// Start begins the async writer workers.
func (w *AsyncWriter) Start() {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.running {
		return
	}
	w.running = true

	for i := 0; i < w.workers; i++ {
		w.wg.Add(1)
		go w.worker(i)
	}

	logging.Infof("AsyncWriter started with %d workers, buffer size %d, batch size %d",
		w.workers, cap(w.writeChan), w.batchSize)
}

// worker processes write operations from the channel.
func (w *AsyncWriter) worker(id int) {
	defer w.wg.Done()

	batch := make([]WriteOperation, 0, w.batchSize)
	ticker := time.NewTicker(w.flushInterval)
	defer ticker.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}

		ctx := context.Background()
		for _, op := range batch {
			var err error
			switch op.Type {
			case OpStore:
				err = w.store.StoreRecord(ctx, op.Record)
			case OpUpdate:
				err = w.store.UpdateRecord(ctx, op.Record)
			case OpDelete:
				if op.Record != nil {
					err = w.store.DeleteRecord(ctx, op.Record.ID)
				}
			}
			if err != nil {
				logging.Warnf("AsyncWriter[%d]: failed to execute %s for record %s: %v",
					id, opTypeName(op.Type), getRecordID(op.Record), err)
			}
		}
		batch = batch[:0]
	}

	for {
		select {
		case op, ok := <-w.writeChan:
			if !ok {
				flush()
				return
			}
			batch = append(batch, op)
			if len(batch) >= w.batchSize {
				flush()
			}
		case <-ticker.C:
			flush()
		case <-w.done:
			// Drain remaining operations
			for {
				select {
				case op, ok := <-w.writeChan:
					if !ok {
						flush()
						return
					}
					batch = append(batch, op)
				default:
					flush()
					return
				}
			}
		}
	}
}

// Enqueue adds a write operation to the async queue.
// Returns true if the operation was enqueued, false if the buffer is full.
func (w *AsyncWriter) Enqueue(op WriteOperation) bool {
	select {
	case w.writeChan <- op:
		return true
	default:
		logging.Warnf("AsyncWriter: write buffer full, dropping %s operation for record %s",
			opTypeName(op.Type), getRecordID(op.Record))
		return false
	}
}

// EnqueueStore enqueues a store operation.
func (w *AsyncWriter) EnqueueStore(record *RoutingRecord) bool {
	return w.Enqueue(WriteOperation{Type: OpStore, Record: record})
}

// EnqueueUpdate enqueues an update operation.
func (w *AsyncWriter) EnqueueUpdate(record *RoutingRecord) bool {
	return w.Enqueue(WriteOperation{Type: OpUpdate, Record: record})
}

// EnqueueDelete enqueues a delete operation.
func (w *AsyncWriter) EnqueueDelete(recordID string) bool {
	return w.Enqueue(WriteOperation{Type: OpDelete, Record: &RoutingRecord{ID: recordID}})
}

// Stop gracefully shuts down the async writer.
// It waits for all pending operations to complete.
func (w *AsyncWriter) Stop() {
	w.mu.Lock()
	if !w.running {
		w.mu.Unlock()
		return
	}
	w.running = false
	w.mu.Unlock()

	close(w.done)
	close(w.writeChan)
	w.wg.Wait()

	logging.Infof("AsyncWriter stopped")
}

// IsRunning returns whether the async writer is running.
func (w *AsyncWriter) IsRunning() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.running
}

// PendingCount returns the number of pending operations in the buffer.
func (w *AsyncWriter) PendingCount() int {
	return len(w.writeChan)
}

// opTypeName returns a human-readable name for the operation type.
func opTypeName(t WriteOpType) string {
	switch t {
	case OpStore:
		return "store"
	case OpUpdate:
		return "update"
	case OpDelete:
		return "delete"
	default:
		return "unknown"
	}
}

// getRecordID safely extracts the record ID.
func getRecordID(record *RoutingRecord) string {
	if record == nil {
		return "<nil>"
	}
	return record.ID
}
