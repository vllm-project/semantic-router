//go:build !windows && cgo

package store_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestStore(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Router Replay Store Suite")
}

// Silence unused import warning
var _ = fmt.Sprintf

var _ = Describe("MemoryStore", func() {
	var (
		memStore *store.MemoryStore
		ctx      context.Context
	)

	BeforeEach(func() {
		ctx = context.Background()
		memStore = store.NewMemoryStore(store.MemoryStoreConfig{
			MaxRecords: 10,
		}, 3600, true)
	})

	AfterEach(func() {
		if memStore != nil {
			memStore.Close()
		}
	})

	Describe("StoreRecord", func() {
		It("should store a record successfully", func() {
			record := &store.RoutingRecord{
				ID:            "test-id-1",
				Timestamp:     time.Now(),
				Decision:      "test-decision",
				Category:      "test-category",
				SelectedModel: "gpt-4",
			}

			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())
			Expect(memStore.RecordCount()).To(Equal(1))
		})

		It("should return error for nil record", func() {
			err := memStore.StoreRecord(ctx, nil)
			Expect(err).To(Equal(store.ErrInvalidInput))
		})

		It("should return error for empty ID", func() {
			record := &store.RoutingRecord{
				ID:       "",
				Decision: "test-decision",
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).To(Equal(store.ErrInvalidInput))
		})

		It("should return error for duplicate ID", func() {
			record := &store.RoutingRecord{
				ID:       "test-id-1",
				Decision: "test-decision",
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			err = memStore.StoreRecord(ctx, record)
			Expect(err).To(Equal(store.ErrAlreadyExists))
		})

		It("should evict oldest record when at capacity", func() {
			// Fill the store
			for i := 0; i < 10; i++ {
				record := &store.RoutingRecord{
					ID:        fmt.Sprintf("record-%d", i),
					Timestamp: time.Now().Add(time.Duration(i) * time.Second),
				}
				err := memStore.StoreRecord(ctx, record)
				Expect(err).NotTo(HaveOccurred())
			}
			Expect(memStore.RecordCount()).To(Equal(10))

			// Add one more, should evict the oldest
			record := &store.RoutingRecord{
				ID:        "record-new",
				Timestamp: time.Now(),
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())
			Expect(memStore.RecordCount()).To(Equal(10))

			// First record should be evicted
			_, err = memStore.GetRecord(ctx, "record-0")
			Expect(err).To(Equal(store.ErrNotFound))

			// New record should exist
			_, err = memStore.GetRecord(ctx, "record-new")
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("GetRecord", func() {
		It("should retrieve a stored record", func() {
			record := &store.RoutingRecord{
				ID:            "test-id-1",
				Timestamp:     time.Now(),
				Decision:      "test-decision",
				Category:      "test-category",
				SelectedModel: "gpt-4",
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			retrieved, err := memStore.GetRecord(ctx, "test-id-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(retrieved.ID).To(Equal("test-id-1"))
			Expect(retrieved.Decision).To(Equal("test-decision"))
		})

		It("should return error for non-existent record", func() {
			_, err := memStore.GetRecord(ctx, "non-existent")
			Expect(err).To(Equal(store.ErrNotFound))
		})

		It("should return error for empty ID", func() {
			_, err := memStore.GetRecord(ctx, "")
			Expect(err).To(Equal(store.ErrInvalidID))
		})
	})

	Describe("UpdateRecord", func() {
		It("should update an existing record", func() {
			record := &store.RoutingRecord{
				ID:             "test-id-1",
				Timestamp:      time.Now(),
				ResponseStatus: 0,
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			record.ResponseStatus = 200
			err = memStore.UpdateRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			retrieved, err := memStore.GetRecord(ctx, "test-id-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(retrieved.ResponseStatus).To(Equal(200))
		})

		It("should return error for non-existent record", func() {
			record := &store.RoutingRecord{
				ID: "non-existent",
			}
			err := memStore.UpdateRecord(ctx, record)
			Expect(err).To(Equal(store.ErrNotFound))
		})
	})

	Describe("DeleteRecord", func() {
		It("should delete an existing record", func() {
			record := &store.RoutingRecord{
				ID:       "test-id-1",
				Decision: "test-decision",
			}
			err := memStore.StoreRecord(ctx, record)
			Expect(err).NotTo(HaveOccurred())

			err = memStore.DeleteRecord(ctx, "test-id-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(memStore.RecordCount()).To(Equal(0))
		})

		It("should return error for non-existent record", func() {
			err := memStore.DeleteRecord(ctx, "non-existent")
			Expect(err).To(Equal(store.ErrNotFound))
		})
	})

	Describe("ListRecords", func() {
		BeforeEach(func() {
			// Add some test records
			for i := 0; i < 5; i++ {
				record := &store.RoutingRecord{
					ID:            fmt.Sprintf("record-%d", i),
					Timestamp:     time.Now().Add(time.Duration(i) * time.Second),
					Decision:      fmt.Sprintf("decision-%d", i%2),
					Category:      "test-category",
					SelectedModel: "gpt-4",
					FromCache:     i%2 == 0,
				}
				err := memStore.StoreRecord(ctx, record)
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should list all records with default pagination", func() {
			result, err := memStore.ListRecords(ctx, store.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result.Records)).To(Equal(5))
		})

		It("should respect limit parameter", func() {
			result, err := memStore.ListRecords(ctx, store.ListOptions{Limit: 2})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result.Records)).To(Equal(2))
			Expect(result.HasMore).To(BeTrue())
		})

		It("should support cursor pagination", func() {
			// Get first page
			result1, err := memStore.ListRecords(ctx, store.ListOptions{Limit: 2, Order: "desc"})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result1.Records)).To(Equal(2))

			// Get second page using after cursor
			result2, err := memStore.ListRecords(ctx, store.ListOptions{
				Limit: 2,
				After: result1.LastID,
				Order: "desc",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result2.Records)).To(BeNumerically(">", 0))
		})

		It("should filter by decision name", func() {
			result, err := memStore.ListRecords(ctx, store.ListOptions{
				DecisionName: "decision-0",
			})
			Expect(err).NotTo(HaveOccurred())
			for _, rec := range result.Records {
				Expect(rec.Decision).To(Equal("decision-0"))
			}
		})

		It("should filter by from_cache", func() {
			fromCache := true
			result, err := memStore.ListRecords(ctx, store.ListOptions{
				FromCache: &fromCache,
			})
			Expect(err).NotTo(HaveOccurred())
			for _, rec := range result.Records {
				Expect(rec.FromCache).To(BeTrue())
			}
		})

		It("should sort by timestamp descending by default", func() {
			result, err := memStore.ListRecords(ctx, store.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result.Records)).To(BeNumerically(">", 1))

			// Verify descending order
			for i := 1; i < len(result.Records); i++ {
				Expect(result.Records[i-1].Timestamp.After(result.Records[i].Timestamp) ||
					result.Records[i-1].Timestamp.Equal(result.Records[i].Timestamp)).To(BeTrue())
			}
		})

		It("should sort ascending when specified", func() {
			result, err := memStore.ListRecords(ctx, store.ListOptions{Order: "asc"})
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result.Records)).To(BeNumerically(">", 1))

			// Verify ascending order
			for i := 1; i < len(result.Records); i++ {
				Expect(result.Records[i].Timestamp.After(result.Records[i-1].Timestamp) ||
					result.Records[i].Timestamp.Equal(result.Records[i-1].Timestamp)).To(BeTrue())
			}
		})
	})

	Describe("Disabled Store", func() {
		var disabledStore *store.MemoryStore

		BeforeEach(func() {
			disabledStore = store.NewMemoryStore(store.MemoryStoreConfig{}, 0, false)
		})

		AfterEach(func() {
			disabledStore.Close()
		})

		It("should return disabled error on StoreRecord", func() {
			record := &store.RoutingRecord{ID: "test"}
			err := disabledStore.StoreRecord(ctx, record)
			Expect(err).To(Equal(store.ErrStoreDisabled))
		})

		It("should return disabled error on GetRecord", func() {
			_, err := disabledStore.GetRecord(ctx, "test")
			Expect(err).To(Equal(store.ErrStoreDisabled))
		})

		It("should return disabled error on ListRecords", func() {
			_, err := disabledStore.ListRecords(ctx, store.ListOptions{})
			Expect(err).To(Equal(store.ErrStoreDisabled))
		})

		It("should report as not enabled", func() {
			Expect(disabledStore.IsEnabled()).To(BeFalse())
		})
	})
})
