package vectorstore

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MemoryMetadataRegistry", func() {
	var (
		reg *MemoryMetadataRegistry
		ctx context.Context
	)

	BeforeEach(func() {
		reg = NewMemoryMetadataRegistry()
		ctx = context.Background()
	})

	Context("Store CRUD", func() {
		It("saves and retrieves a store", func() {
			vs := &VectorStore{ID: "vs_1", Name: "test", Status: "active", CreatedAt: 1000}
			Expect(reg.SaveStore(ctx, vs)).To(Succeed())

			got, err := reg.GetStore(ctx, "vs_1")
			Expect(err).NotTo(HaveOccurred())
			Expect(got.Name).To(Equal("test"))
		})

		It("lists all stores", func() {
			Expect(reg.SaveStore(ctx, &VectorStore{ID: "vs_a"})).To(Succeed())
			Expect(reg.SaveStore(ctx, &VectorStore{ID: "vs_b"})).To(Succeed())

			stores, err := reg.ListStores(ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(stores).To(HaveLen(2))
		})

		It("deletes a store", func() {
			Expect(reg.SaveStore(ctx, &VectorStore{ID: "vs_del"})).To(Succeed())
			Expect(reg.DeleteStore(ctx, "vs_del")).To(Succeed())

			_, err := reg.GetStore(ctx, "vs_del")
			Expect(err).To(HaveOccurred())
		})

		It("upserts on duplicate ID", func() {
			Expect(reg.SaveStore(ctx, &VectorStore{ID: "vs_dup", Name: "v1"})).To(Succeed())
			Expect(reg.SaveStore(ctx, &VectorStore{ID: "vs_dup", Name: "v2"})).To(Succeed())

			got, err := reg.GetStore(ctx, "vs_dup")
			Expect(err).NotTo(HaveOccurred())
			Expect(got.Name).To(Equal("v2"))
		})

		It("returns error for missing store", func() {
			_, err := reg.GetStore(ctx, "vs_missing")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("File CRUD", func() {
		It("saves and retrieves a file record", func() {
			fr := &FileRecord{ID: "file_1", Filename: "a.txt", Status: "uploaded", CreatedAt: 2000}
			Expect(reg.SaveFile(ctx, fr)).To(Succeed())

			got, err := reg.GetFile(ctx, "file_1")
			Expect(err).NotTo(HaveOccurred())
			Expect(got.Filename).To(Equal("a.txt"))
		})

		It("lists all file records", func() {
			Expect(reg.SaveFile(ctx, &FileRecord{ID: "f_a"})).To(Succeed())
			Expect(reg.SaveFile(ctx, &FileRecord{ID: "f_b"})).To(Succeed())
			Expect(reg.SaveFile(ctx, &FileRecord{ID: "f_c"})).To(Succeed())

			files, err := reg.ListFiles(ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(files).To(HaveLen(3))
		})

		It("deletes a file record", func() {
			Expect(reg.SaveFile(ctx, &FileRecord{ID: "f_del"})).To(Succeed())
			Expect(reg.DeleteFile(ctx, "f_del")).To(Succeed())

			_, err := reg.GetFile(ctx, "f_del")
			Expect(err).To(HaveOccurred())
		})

		It("returns error for missing file", func() {
			_, err := reg.GetFile(ctx, "f_missing")
			Expect(err).To(HaveOccurred())
		})
	})
})
