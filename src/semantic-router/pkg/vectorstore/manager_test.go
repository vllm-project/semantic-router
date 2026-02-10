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

var _ = Describe("Manager", func() {
	var (
		mgr *Manager
		ctx context.Context
	)

	BeforeEach(func() {
		backend := NewMemoryBackend(MemoryBackendConfig{})
		mgr = NewManager(backend, 768, BackendTypeMemory)
		ctx = context.Background()
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
	})

	Context("ListStores", func() {
		BeforeEach(func() {
			for i := 0; i < 5; i++ {
				_, err := mgr.CreateStore(ctx, CreateStoreRequest{
					Name: "list-test",
				})
				Expect(err).NotTo(HaveOccurred())
			}
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

		It("should handle empty result", func() {
			emptyMgr := NewManager(NewMemoryBackend(MemoryBackendConfig{}), 768, BackendTypeMemory)
			stores := emptyMgr.ListStores(ListStoresParams{})
			Expect(stores).To(BeEmpty())
		})
	})

	Context("UpdateStore", func() {
		It("should update name", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "original"})
			Expect(err).NotTo(HaveOccurred())

			newName := "updated"
			updated, err := mgr.UpdateStore(vs.ID, UpdateStoreRequest{Name: &newName})
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.Name).To(Equal("updated"))
		})

		It("should update metadata", func() {
			vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "meta"})
			Expect(err).NotTo(HaveOccurred())

			updated, err := mgr.UpdateStore(vs.ID, UpdateStoreRequest{
				Metadata: map[string]interface{}{"key": "val"},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(updated.Metadata["key"]).To(Equal("val"))
		})

		It("should return error for non-existent store", func() {
			_, err := mgr.UpdateStore("vs_nonexistent", UpdateStoreRequest{})
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

			err = mgr.UpdateFileCounts(vs.ID, func(fc *FileCounts) {
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
			err := mgr.UpdateFileCounts("vs_nonexistent", func(fc *FileCounts) {})
			Expect(err).To(HaveOccurred())
		})
	})
})
