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
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// deterministicEmbedder returns embeddings based on simple word matching
// to produce predictable, repeatable similarity scores.
type deterministicEmbedder struct {
	dim      int
	keywords map[string][]float32
}

func newDeterministicEmbedder(dim int) *deterministicEmbedder {
	e := &deterministicEmbedder{
		dim:      dim,
		keywords: make(map[string][]float32),
	}
	// Pre-define embeddings for known terms so search results are deterministic.
	// Order matters: more specific keywords are checked first via the ordered rules.
	e.keywords["pto"] = normalizedBasis(dim, 0)
	e.keywords["vacation"] = normalizedBasis(dim, 0)
	e.keywords["password"] = normalizedBasis(dim, 1)
	e.keywords["security"] = normalizedBasis(dim, 1)
	e.keywords["login"] = normalizedBasis(dim, 1)
	// "policy" is intentionally omitted — it's ambiguous across topics
	return e
}

func (e *deterministicEmbedder) Embed(text string) ([]float32, error) {
	// Check for keyword matches and return the corresponding basis vector.
	for keyword, emb := range e.keywords {
		if containsWord(text, keyword) {
			return emb, nil
		}
	}
	// Default: return a vector pointing in a neutral direction.
	v := make([]float32, e.dim)
	v[e.dim-1] = 1.0
	return v, nil
}

func (e *deterministicEmbedder) Dimension() int {
	return e.dim
}

// normalizedBasis returns a unit vector with 1.0 at the given index.
func normalizedBasis(dim, idx int) []float32 {
	v := make([]float32, dim)
	if idx < dim {
		v[idx] = 1.0
	}
	return v
}

// containsWord checks if text contains a word (case-insensitive substring).
func containsWord(text, word string) bool {
	lower := make([]byte, len(text))
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		lower[i] = c
	}
	wLower := make([]byte, len(word))
	for i := 0; i < len(word); i++ {
		c := word[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		wLower[i] = c
	}
	return bytesContains(lower, wLower)
}

func bytesContains(haystack, needle []byte) bool {
	if len(needle) > len(haystack) {
		return false
	}
	for i := 0; i <= len(haystack)-len(needle); i++ {
		match := true
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

var _ = Describe("Integration: Full Ingest-to-Search Pipeline", func() {
	var (
		backend  *MemoryBackend
		store    *FileStore
		mgr      *Manager
		embedder *deterministicEmbedder
		pipeline *IngestionPipeline
		tempDir  string
		ctx      context.Context
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "integration-test-*")
		Expect(err).NotTo(HaveOccurred())

		backend = NewMemoryBackend(MemoryBackendConfig{})
		store, err = NewFileStore(tempDir)
		Expect(err).NotTo(HaveOccurred())

		embedder = newDeterministicEmbedder(8)
		mgr = NewManager(backend, embedder.Dimension(), BackendTypeMemory)
		pipeline = NewIngestionPipeline(backend, store, mgr, embedder, PipelineConfig{
			Workers:   2,
			QueueSize: 10,
		})
		pipeline.Start()
		ctx = context.Background()
	})

	AfterEach(func() {
		pipeline.Stop()
		os.RemoveAll(tempDir)
	})

	It("should ingest a document and find it via semantic search", func() {
		// 1. Create vector store
		vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "integration-test"})
		Expect(err).NotTo(HaveOccurred())

		// 2. Upload two files with different topics
		ptoDoc := `PTO Policy

All employees receive 20 days of paid time off per year.
Vacation requests must be submitted 2 weeks in advance.`

		securityDoc := `Password Policy

All passwords must be at least 12 characters.
Login attempts are locked after 5 failures.`

		ptoRecord, err := store.Save("pto-policy.txt", []byte(ptoDoc), "assistants")
		Expect(err).NotTo(HaveOccurred())

		secRecord, err := store.Save("security-policy.txt", []byte(securityDoc), "assistants")
		Expect(err).NotTo(HaveOccurred())

		// 3. Attach both files
		vsfPTO, err := pipeline.AttachFile(vs.ID, ptoRecord.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		vsfSec, err := pipeline.AttachFile(vs.ID, secRecord.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		// 4. Wait for both to complete
		Eventually(func() string {
			s, _ := pipeline.GetFileStatus(vsfPTO.ID)
			return s.Status
		}, 10*time.Second, 100*time.Millisecond).Should(Equal("completed"))

		Eventually(func() string {
			s, _ := pipeline.GetFileStatus(vsfSec.ID)
			return s.Status
		}, 10*time.Second, 100*time.Millisecond).Should(Equal("completed"))

		// 5. Verify file counts
		updated, err := mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Completed).To(Equal(2))
		Expect(updated.FileCounts.Total).To(Equal(2))
		Expect(updated.FileCounts.InProgress).To(Equal(0))
		Expect(updated.FileCounts.Failed).To(Equal(0))

		// 6. Search for PTO-related content
		ptoQuery, err := embedder.Embed("What is our PTO policy?")
		Expect(err).NotTo(HaveOccurred())

		results, err := backend.Search(ctx, vs.ID, ptoQuery, 5, 0.5, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))

		// The top result should be from the PTO document
		Expect(results[0].Filename).To(Equal("pto-policy.txt"))
		Expect(results[0].Score).To(BeNumerically(">", 0.9))

		// 7. Search for security-related content
		secQuery, err := embedder.Embed("How do I reset my password?")
		Expect(err).NotTo(HaveOccurred())

		results, err = backend.Search(ctx, vs.ID, secQuery, 5, 0.5, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))

		// The top result should be from the security document
		Expect(results[0].Filename).To(Equal("security-policy.txt"))
		Expect(results[0].Score).To(BeNumerically(">", 0.9))
	})

	It("should simulate the RAG retrieval flow: embed query → search → format context", func() {
		// This test simulates what retrieveFromVectorStore() does
		// without needing the full OpenAIRouter.

		// Setup: create store and ingest a document
		vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "rag-sim"})
		Expect(err).NotTo(HaveOccurred())

		doc := "Our company vacation policy allows 20 days PTO per year."
		record, err := store.Save("handbook.txt", []byte(doc), "assistants")
		Expect(err).NotTo(HaveOccurred())

		vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() string {
			s, _ := pipeline.GetFileStatus(vsf.ID)
			return s.Status
		}, 10*time.Second, 100*time.Millisecond).Should(Equal("completed"))

		// Simulate RAG retrieval:
		// 1. Embed the user query
		userQuery := "What is the vacation policy?"
		queryEmbedding, err := embedder.Embed(userQuery)
		Expect(err).NotTo(HaveOccurred())

		// 2. Search the vector store
		topK := 5
		threshold := float32(0.5)
		results, err := mgr.Backend().Search(ctx, vs.ID, queryEmbedding, topK, threshold, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))

		// 3. Format as context string (same as retrieveFromVectorStore)
		var contextParts []string
		for _, r := range results {
			contextParts = append(contextParts, r.Content)
		}
		retrievedContext := ""
		for i, part := range contextParts {
			if i > 0 {
				retrievedContext += "\n\n---\n\n"
			}
			retrievedContext += part
		}

		// 4. Verify the context contains relevant content
		Expect(retrievedContext).NotTo(BeEmpty())
		Expect(retrievedContext).To(ContainSubstring("vacation"))

		// 5. Verify similarity score is meaningful
		Expect(results[0].Score).To(BeNumerically(">", 0.5))
	})

	It("should handle multiple file formats through the pipeline", func() {
		vs, fmtErr := mgr.CreateStore(ctx, CreateStoreRequest{Name: "formats-test"})
		Expect(fmtErr).NotTo(HaveOccurred())

		// Upload different formats
		files := map[string][]byte{
			"doc.txt":  []byte("PTO policy details for text format."),
			"doc.md":   []byte("# PTO Policy\n\nMarkdown format vacation rules."),
			"doc.json": []byte(`{"topic":"PTO","content":"JSON format policy data"}`),
			"doc.csv":  []byte("Topic,Content\nPTO,CSV format vacation info"),
		}

		for name, content := range files {
			record, saveErr := store.Save(name, content, "assistants")
			Expect(saveErr).NotTo(HaveOccurred())

			vsf, attachErr := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(attachErr).NotTo(HaveOccurred())

			Eventually(func() string {
				s, _ := pipeline.GetFileStatus(vsf.ID)
				return s.Status
			}, 10*time.Second, 100*time.Millisecond).Should(Equal("completed"),
				fmt.Sprintf("File %s should complete", name))
		}

		// All 4 files should be completed
		updated, fmtErr := mgr.GetStore(vs.ID)
		Expect(fmtErr).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Completed).To(Equal(4))

		// Search should return results from multiple files
		query, _ := embedder.Embed("PTO policy information")
		results, err := backend.Search(ctx, vs.ID, query, 10, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 4))
	})

	It("should handle detach and verify chunks are removed", func() {
		vs, err := mgr.CreateStore(ctx, CreateStoreRequest{Name: "detach-test"})
		Expect(err).NotTo(HaveOccurred())

		record, err := store.Save("temp.txt", []byte("Temporary PTO document."), "assistants")
		Expect(err).NotTo(HaveOccurred())

		vsf, err := pipeline.AttachFile(vs.ID, record.ID, nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() string {
			s, _ := pipeline.GetFileStatus(vsf.ID)
			return s.Status
		}, 10*time.Second, 100*time.Millisecond).Should(Equal("completed"))

		// Verify search finds results
		query, _ := embedder.Embed("PTO")
		results, err := backend.Search(ctx, vs.ID, query, 10, 0.5, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(results)).To(BeNumerically(">=", 1))

		// Detach file
		err = pipeline.DetachFile(ctx, vs.ID, vsf.ID)
		Expect(err).NotTo(HaveOccurred())

		// Search should return no results now
		results, err = backend.Search(ctx, vs.ID, query, 10, 0.5, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(results).To(BeEmpty())

		// File counts should be updated
		updated, err := mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Total).To(Equal(0))
	})

	It("should handle concurrent file ingestion", func() {
		vs, concErr := mgr.CreateStore(ctx, CreateStoreRequest{Name: "concurrent-test"})
		Expect(concErr).NotTo(HaveOccurred())

		// Attach 5 files concurrently
		vsfIDs := make([]string, 5)
		for i := 0; i < 5; i++ {
			content := fmt.Sprintf("Document %d about PTO policy section %d.", i, i)
			record, saveErr := store.Save(fmt.Sprintf("doc-%d.txt", i), []byte(content), "assistants")
			Expect(saveErr).NotTo(HaveOccurred())

			vsf, attachErr := pipeline.AttachFile(vs.ID, record.ID, nil)
			Expect(attachErr).NotTo(HaveOccurred())
			vsfIDs[i] = vsf.ID
		}

		// Wait for all to complete
		for _, id := range vsfIDs {
			Eventually(func() string {
				s, _ := pipeline.GetFileStatus(id)
				return s.Status
			}, 15*time.Second, 100*time.Millisecond).Should(Equal("completed"))
		}

		// Verify all completed
		updated, err := mgr.GetStore(vs.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(updated.FileCounts.Completed).To(Equal(5))
		Expect(updated.FileCounts.Total).To(Equal(5))
	})
})

// Verify deterministicEmbedder satisfies the Embedder interface.
var _ Embedder = (*deterministicEmbedder)(nil)
