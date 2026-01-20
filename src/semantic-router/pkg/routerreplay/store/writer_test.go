//go:build !windows && cgo

package store_test

import (
	"context"
	"fmt"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Silence unused import warning
var _ = fmt.Sprintf

var _ = Describe("AsyncWriter", func() {
	var (
		memStore    *store.MemoryStore
		asyncWriter *store.AsyncWriter
		ctx         context.Context
	)

	BeforeEach(func() {
		ctx = context.Background()
		memStore = store.NewMemoryStore(store.MemoryStoreConfig{
			MaxRecords: 100,
		}, 3600, true)
	})

	AfterEach(func() {
		if asyncWriter != nil {
			asyncWriter.Stop()
		}
		if memStore != nil {
			memStore.Close()
		}
	})

	Describe("Basic Operations", func() {
		BeforeEach(func() {
			asyncWriter = store.NewAsyncWriter(memStore, store.AsyncWriterConfig{
				BufferSize:      100,
				BatchSize:       5,
				FlushIntervalMs: 50,
				Workers:         2,
			})
			asyncWriter.Start()
		})

		It("should enqueue and process store operations", func() {
			record := &store.RoutingRecord{
				ID:       "async-test-1",
				Decision: "test-decision",
			}

			success := asyncWriter.EnqueueStore(record)
			Expect(success).To(BeTrue())

			// Wait for async processing
			Eventually(func() int {
				return memStore.RecordCount()
			}, 1*time.Second, 10*time.Millisecond).Should(Equal(1))
		})

		It("should process multiple records in batch", func() {
			for i := 0; i < 10; i++ {
				record := &store.RoutingRecord{
					ID:       fmt.Sprintf("batch-test-%d", i),
					Decision: "test-decision",
				}
				asyncWriter.EnqueueStore(record)
			}

			// Wait for all records to be processed
			Eventually(func() int {
				return memStore.RecordCount()
			}, 2*time.Second, 50*time.Millisecond).Should(Equal(10))
		})

		It("should process update operations", func() {
			// First store a record
			record := &store.RoutingRecord{
				ID:             "update-test-1",
				Decision:       "test-decision",
				ResponseStatus: 0,
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			// Update via async writer
			record.ResponseStatus = 200
			asyncWriter.EnqueueUpdate(record)

			// Wait for update to be processed
			Eventually(func() int {
				rec, _ := memStore.GetRecord(ctx, "update-test-1")
				if rec != nil {
					return rec.ResponseStatus
				}
				return 0
			}, 1*time.Second, 10*time.Millisecond).Should(Equal(200))
		})
	})

	Describe("Graceful Shutdown", func() {
		It("should process pending operations on stop", func() {
			asyncWriter = store.NewAsyncWriter(memStore, store.AsyncWriterConfig{
				BufferSize:      100,
				BatchSize:       50, // Large batch to prevent immediate flush
				FlushIntervalMs: 10000,
				Workers:         1,
			})
			asyncWriter.Start()

			// Enqueue some records
			for i := 0; i < 5; i++ {
				record := &store.RoutingRecord{
					ID:       fmt.Sprintf("shutdown-test-%d", i),
					Decision: "test-decision",
				}
				asyncWriter.EnqueueStore(record)
			}

			// Stop should process pending records
			asyncWriter.Stop()

			// All records should be processed
			Expect(memStore.RecordCount()).To(Equal(5))
		})
	})

	Describe("Buffer Full Handling", func() {
		It("should return false when buffer is full", func() {
			asyncWriter = store.NewAsyncWriter(memStore, store.AsyncWriterConfig{
				BufferSize:      2,
				BatchSize:       100, // Large batch to prevent processing
				FlushIntervalMs: 10000,
				Workers:         0, // No workers, nothing gets processed
			})
			// Don't start - buffer will fill up

			// Fill the buffer
			for i := 0; i < 3; i++ {
				record := &store.RoutingRecord{
					ID: fmt.Sprintf("buffer-test-%d", i),
				}
				success := asyncWriter.EnqueueStore(record)
				if i < 2 {
					Expect(success).To(BeTrue())
				} else {
					Expect(success).To(BeFalse())
				}
			}
		})
	})

	Describe("Concurrent Access", func() {
		It("should handle concurrent enqueues safely", func() {
			// Create a larger store for this test
			largeStore := store.NewMemoryStore(store.MemoryStoreConfig{
				MaxRecords: 1000,
			}, 3600, true)
			defer largeStore.Close()

			asyncWriter = store.NewAsyncWriter(largeStore, store.AsyncWriterConfig{
				BufferSize:      1000,
				BatchSize:       10,
				FlushIntervalMs: 10,
				Workers:         4,
			})
			asyncWriter.Start()

			var wg sync.WaitGroup
			numGoroutines := 10
			recordsPerGoroutine := 20

			for g := 0; g < numGoroutines; g++ {
				wg.Add(1)
				go func(goroutineID int) {
					defer wg.Done()
					for i := 0; i < recordsPerGoroutine; i++ {
						record := &store.RoutingRecord{
							ID:       fmt.Sprintf("concurrent-g%d-r%d", goroutineID, i),
							Decision: "test-decision",
						}
						asyncWriter.EnqueueStore(record)
					}
				}(g)
			}

			wg.Wait()

			// Wait for all records to be processed
			Eventually(func() int {
				return largeStore.RecordCount()
			}, 5*time.Second, 50*time.Millisecond).Should(Equal(numGoroutines * recordsPerGoroutine))
		})
	})

	Describe("Status Methods", func() {
		It("should report running status correctly", func() {
			asyncWriter = store.NewAsyncWriter(memStore, store.DefaultAsyncWriterConfig())

			Expect(asyncWriter.IsRunning()).To(BeFalse())

			asyncWriter.Start()
			Expect(asyncWriter.IsRunning()).To(BeTrue())

			asyncWriter.Stop()
			Expect(asyncWriter.IsRunning()).To(BeFalse())
		})

		It("should report pending count", func() {
			asyncWriter = store.NewAsyncWriter(memStore, store.AsyncWriterConfig{
				BufferSize:      100,
				BatchSize:       100,
				FlushIntervalMs: 10000,
				Workers:         0, // No workers
			})

			Expect(asyncWriter.PendingCount()).To(Equal(0))

			record := &store.RoutingRecord{ID: "pending-test"}
			asyncWriter.EnqueueStore(record)

			Expect(asyncWriter.PendingCount()).To(Equal(1))
		})
	})
})
