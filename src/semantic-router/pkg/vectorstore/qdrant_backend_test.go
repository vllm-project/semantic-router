package vectorstore

import (
	"context"
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("QdrantBackend", func() {
	skipQdrant := os.Getenv("SKIP_QDRANT_TESTS") != "false"

	Context("integration tests", func() {
		BeforeEach(func() {
			if skipQdrant {
				Skip("Skipping Qdrant tests (set SKIP_QDRANT_TESTS=false to enable)")
			}
		})

		It("should insert and search chunks", func() {
			backend, err := NewQdrantBackend(QdrantBackendConfig{
				Host: "localhost",
				Port: 6334,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()
			vsID := "test_search"

			_ = backend.DeleteCollection(ctx, vsID)

			err = backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.DeleteCollection(ctx, vsID) }()

			chunks := []EmbeddedChunk{
				{ID: "c1", FileID: "f1", Filename: "doc.txt", Content: "hello", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "c2", FileID: "f1", Filename: "doc.txt", Content: "world", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 1},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0.5, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically(">=", 1))
			Expect(results[0].Content).To(Equal("hello"))
		})

		It("should delete chunks by file ID", func() {
			backend, err := NewQdrantBackend(QdrantBackendConfig{
				Host: "localhost",
				Port: 6334,
			})
			Expect(err).NotTo(HaveOccurred())
			defer backend.Close()

			ctx := context.Background()
			vsID := "test_delete_by_file"

			_ = backend.DeleteCollection(ctx, vsID)
			err = backend.CreateCollection(ctx, vsID, 3)
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.DeleteCollection(ctx, vsID) }()

			chunks := []EmbeddedChunk{
				{ID: "d1", FileID: "file-a", Filename: "a.txt", Content: "alpha", Embedding: normalizeVec([]float32{1, 0, 0}), ChunkIndex: 0},
				{ID: "d2", FileID: "file-b", Filename: "b.txt", Content: "beta", Embedding: normalizeVec([]float32{0, 1, 0}), ChunkIndex: 0},
			}
			err = backend.InsertChunks(ctx, vsID, chunks)
			Expect(err).NotTo(HaveOccurred())

			err = backend.DeleteByFileID(ctx, vsID, "file-a")
			Expect(err).NotTo(HaveOccurred())

			results, err := backend.Search(ctx, vsID, normalizeVec([]float32{1, 0, 0}), 5, 0.0, nil)
			Expect(err).NotTo(HaveOccurred())
			for _, r := range results {
				Expect(r.FileID).NotTo(Equal("file-a"))
			}
		})
	})
})
