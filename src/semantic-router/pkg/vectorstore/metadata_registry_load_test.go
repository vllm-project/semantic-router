package vectorstore

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LoadFromRegistry", func() {
	var ctx context.Context

	BeforeEach(func() {
		ctx = context.Background()
	})

	Context("Manager", func() {
		It("populates stores from the registry on startup", func() {
			reg := NewMemoryMetadataRegistry()
			Expect(reg.SaveStore(ctx, &VectorStore{
				ID: "vs_pre", Name: "preloaded", Status: "active", CreatedAt: 100,
			})).To(Succeed())

			backend := NewMemoryBackend(MemoryBackendConfig{})
			mgr := NewManager(backend, reg, 768, BackendTypeMemory)

			Expect(mgr.LoadFromRegistry(ctx)).To(Succeed())

			vs, err := mgr.GetStore("vs_pre")
			Expect(err).NotTo(HaveOccurred())
			Expect(vs.Name).To(Equal("preloaded"))
		})

		It("returns empty when registry has no stores", func() {
			reg := NewMemoryMetadataRegistry()
			backend := NewMemoryBackend(MemoryBackendConfig{})
			mgr := NewManager(backend, reg, 768, BackendTypeMemory)

			Expect(mgr.LoadFromRegistry(ctx)).To(Succeed())

			stores := mgr.ListStores(ListStoresParams{})
			Expect(stores).To(BeEmpty())
		})
	})

	Context("FileStore", func() {
		It("populates files from the registry on startup", func() {
			reg := NewMemoryMetadataRegistry()
			Expect(reg.SaveFile(ctx, &FileRecord{
				ID: "file_pre", Filename: "doc.txt", Status: "uploaded", CreatedAt: 200,
			})).To(Succeed())

			tempDir := GinkgoT().TempDir()
			fs, err := NewFileStore(tempDir, reg)
			Expect(err).NotTo(HaveOccurred())

			Expect(fs.LoadFromRegistry(ctx)).To(Succeed())

			fr, err := fs.Get("file_pre")
			Expect(err).NotTo(HaveOccurred())
			Expect(fr.Filename).To(Equal("doc.txt"))
		})
	})
})
