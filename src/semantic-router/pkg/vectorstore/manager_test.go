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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func newManagerTestFixture() (*Manager, context.Context) {
	GinkgoHelper()

	backend := NewMemoryBackend(MemoryBackendConfig{})
	registry := NewMemoryMetadataRegistry()
	return NewManager(backend, registry, 768, BackendTypeMemory), context.Background()
}

var _ = Describe("Manager store creation and retrieval", func() {
	var (
		mgr *Manager
		ctx context.Context
	)

	BeforeEach(func() {
		mgr, ctx = newManagerTestFixture()
	})

	Context("CreateStore", func() {
		It("should create a vector store", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{
				Name: "test-store",
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(vs).NotTo(BeNil())
			Expect(vs.ID).To(HavePrefix("vs_"))
			Expect(vs.Object).To(Equal("vector_store"))
			Expect(vs.Name).To(Equal("test-store"))
			Expect(vs.Status).To(Equal("active"))
			Expect(vs.BackendType).To(Equal("memory"))
			Expect(vs.CreatedAt).To(BeNumerically(">", 0))
		})

		It("should create store with metadata", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{
				Name:     "meta-store",
				Metadata: map[string]interface{}{"env": "test"},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(vs.Metadata["env"]).To(Equal("test"))
		})

		It("should create store with expiration", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{
				Name:         "expiring",
				ExpiresAfter: &ExpirationPolicy{Anchor: "last_active_at", Days: 7},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(vs.ExpiresAfter).NotTo(BeNil())
			Expect(vs.ExpiresAfter.Days).To(Equal(7))
		})

		It("should create backing collection in backend", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "backed"})
			Expect(err).NotTo(HaveOccurred())

			exists, err := mgr.Backend().CollectionExists(ctx, vs.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue())
		})
	})

	Context("GetStore", func() {
		It("should return an existing store", func() {
			created, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "get-test"})
			Expect(err).NotTo(HaveOccurred())

			vs, err := mgr.GetStore(created.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(vs.ID).To(Equal(created.ID))
			Expect(vs.Name).To(Equal("get-test"))
		})

		It("should return error for non-existent store", func() {
			_, err := mgr.GetStore("vs_nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not found"))
		})

		It("should return defensive store copies", func() {
			created, err := mgr.CreateStore(ctx, CreateStoreRequest{
				Name:     "copy-test",
				Metadata: map[string]interface{}{"env": "test"},
			})
			Expect(err).NotTo(HaveOccurred())

			got, err := mgr.GetStore(created.ID)
			Expect(err).NotTo(HaveOccurred())
			got.Name = "mutated"
			got.Metadata["env"] = "mutated"

			reloaded, err := mgr.GetStore(created.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(reloaded.Name).To(Equal("copy-test"))
			Expect(reloaded.Metadata["env"]).To(Equal("test"))
		})
	})
})

var _ = Describe("Manager store listing", func() {
	var (
		mgr *Manager
		ctx context.Context
	)

	BeforeEach(func() {
		mgr, ctx = newManagerTestFixture()
	})

	Context("ListStores", func() {
		var created []*VectorStore

		BeforeEach(func() {
			created = nil
			for i := 0; i < 5; i++ {
				vs, err := mgr.CreateStore(ctx, CreateStoreRequest{
					Name: "list-test",
				})
				Expect(err).NotTo(HaveOccurred())
				created = append(created, vs)
			}
			setStoreCreatedAt(mgr, created)
		})

		It("should return all stores", func() {
			stores := mgr.ListStores(ListStoresParams{})
			Expect(stores).To(HaveLen(5))
		})

		It("should respect limit", func() {
			stores := mgr.ListStores(ListStoresParams{Limit: 2})
			Expect(stores).To(HaveLen(2))
		})

		It("should cap limit at 100", func() {
			stores := mgr.ListStores(ListStoresParams{Limit: 200})
			Expect(stores).To(HaveLen(5)) // only 5 exist
		})

		It("should respect after cursor in sorted order", func() {
			stores := mgr.ListStores(ListStoresParams{After: created[2].ID})
			Expect(storeIDs(stores)).To(Equal([]string{created[1].ID, created[0].ID}))
		})

		It("should respect before cursor in sorted order", func() {
			stores := mgr.ListStores(ListStoresParams{Before: created[2].ID})
			Expect(storeIDs(stores)).To(Equal([]string{created[4].ID, created[3].ID}))
		})

		It("should respect asc order", func() {
			stores := mgr.ListStores(ListStoresParams{Order: "asc", Limit: 3})
			Expect(storeIDs(stores)).To(Equal([]string{created[0].ID, created[1].ID, created[2].ID}))
		})

		It("should handle empty result", func() {
			emptyMgr := NewManager(NewMemoryBackend(MemoryBackendConfig{}), NewMemoryMetadataRegistry(), 768, BackendTypeMemory)
			stores := emptyMgr.ListStores(ListStoresParams{})
			Expect(stores).To(BeEmpty())
		})
	})
})

func setStoreCreatedAt(mgr *Manager, stores []*VectorStore) {
	GinkgoHelper()

	mgr.mu.Lock()
	defer mgr.mu.Unlock()
	for i, store := range stores {
		mgr.stores[store.ID].CreatedAt = int64(i + 1)
	}
}

func storeIDs(stores []*VectorStore) []string {
	ids := make([]string, 0, len(stores))
	for _, store := range stores {
		ids = append(ids, store.ID)
	}
	return ids
}

var _ = Describe("Manager store mutation", func() {
	var (
		mgr *Manager
		ctx context.Context
	)

	BeforeEach(func() {
		mgr, ctx = newManagerTestFixture()
	})

	Context("UpdateStore", func() {
		It("should update name", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "original"})
			Expect(err).NotTo(HaveOccurred())

			newName := "updated"
			updated, err := mgr.UpdateStore(ctx, vs.ID, UpdateStoreRequest{Name: &newName})
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.Name).To(Equal("updated"))
		})

		It("should update metadata", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "meta"})
			Expect(err).NotTo(HaveOccurred())

			updated, err := mgr.UpdateStore(ctx, vs.ID, UpdateStoreRequest{
				Metadata: map[string]interface{}{"key": "val"},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.Metadata["key"]).To(Equal("val"))
		})

		It("should return error for non-existent store", func() {
			_, err := mgr.UpdateStore(ctx, "vs_nonexistent", UpdateStoreRequest{})
			Expect(err).To(HaveOccurred())
		})
	})

	Context("DeleteStore", func() {
		It("should delete a store", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "delete-me"})
			Expect(err).NotTo(HaveOccurred())

			err = mgr.DeleteStore(ctx, vs.ID)
			Expect(err).NotTo(HaveOccurred())

			_, err = mgr.GetStore(vs.ID)
			Expect(err).To(HaveOccurred())
		})

		It("should delete backing collection", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "del-backend"})
			Expect(err).NotTo(HaveOccurred())

			err = mgr.DeleteStore(ctx, vs.ID)
			Expect(err).NotTo(HaveOccurred())

			exists, err := mgr.Backend().CollectionExists(ctx, vs.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeFalse())
		})

		It("should return error for non-existent store", func() {
			err := mgr.DeleteStore(ctx, "vs_nonexistent")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("UpdateFileCounts", func() {
		It("should update file counts", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "counts"})
			Expect(err).NotTo(HaveOccurred())

			err = mgr.UpdateFileCounts(ctx, vs.ID, func(fc *FileCounts) {
				fc.Completed++
				fc.Total++
			})
			Expect(err).NotTo(HaveOccurred())

			updated, err := mgr.GetStore(vs.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.FileCounts.Completed).To(Equal(1))
			Expect(updated.FileCounts.Total).To(Equal(1))
		})

		It("should return error for non-existent store", func() {
			err := mgr.UpdateFileCounts(ctx, "vs_nonexistent", func(fc *FileCounts) {})
			Expect(err).To(HaveOccurred())
		})
	})
})
