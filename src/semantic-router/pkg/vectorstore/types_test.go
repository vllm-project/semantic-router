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
	"encoding/json"
	"strings"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestVectorStore(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "VectorStore Suite")
}

var _ = Describe("ID Generation", func() {
	Context("GenerateVectorStoreID", func() {
		It("should generate IDs with vs_ prefix", func() {
			id := GenerateVectorStoreID()
			Expect(id).To(HavePrefix("vs_"))
			Expect(len(id)).To(BeNumerically(">", len(PrefixVectorStore)))
		})

		It("should generate unique IDs", func() {
			id1 := GenerateVectorStoreID()
			id2 := GenerateVectorStoreID()
			Expect(id1).NotTo(Equal(id2))
		})
	})

	Context("GenerateFileID", func() {
		It("should generate IDs with file_ prefix", func() {
			id := GenerateFileID()
			Expect(id).To(HavePrefix("file_"))
			Expect(len(id)).To(BeNumerically(">", len(PrefixFile)))
		})

		It("should generate unique IDs", func() {
			id1 := GenerateFileID()
			id2 := GenerateFileID()
			Expect(id1).NotTo(Equal(id2))
		})
	})

	Context("GenerateVectorStoreFileID", func() {
		It("should generate IDs with vsf_ prefix", func() {
			id := GenerateVectorStoreFileID()
			Expect(id).To(HavePrefix("vsf_"))
			Expect(len(id)).To(BeNumerically(">", len(PrefixVectorStoreFile)))
		})

		It("should generate unique IDs", func() {
			id1 := GenerateVectorStoreFileID()
			id2 := GenerateVectorStoreFileID()
			Expect(id1).NotTo(Equal(id2))
		})
	})
})

var _ = Describe("JSON Serialization", func() {
	Context("VectorStore", func() {
		It("should serialize with correct JSON tags", func() {
			vs := VectorStore{
				ID:          "vs_123",
				Object:      "vector_store",
				Name:        "test-store",
				CreatedAt:   1700000000,
				Status:      "active",
				FileCounts:  FileCounts{InProgress: 1, Completed: 2, Failed: 0, Total: 3},
				BackendType: "memory",
				Metadata:    map[string]interface{}{"key": "value"},
			}

			data, err := json.Marshal(vs)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["id"]).To(Equal("vs_123"))
			Expect(decoded["object"]).To(Equal("vector_store"))
			Expect(decoded["name"]).To(Equal("test-store"))
			Expect(decoded["created_at"]).To(BeNumerically("==", 1700000000))
			Expect(decoded["status"]).To(Equal("active"))
			Expect(decoded["backend_type"]).To(Equal("memory"))

			fileCounts := decoded["file_counts"].(map[string]interface{})
			Expect(fileCounts["in_progress"]).To(BeNumerically("==", 1))
			Expect(fileCounts["completed"]).To(BeNumerically("==", 2))
			Expect(fileCounts["failed"]).To(BeNumerically("==", 0))
			Expect(fileCounts["total"]).To(BeNumerically("==", 3))
		})

		It("should omit nil optional fields", func() {
			vs := VectorStore{
				ID:     "vs_123",
				Object: "vector_store",
			}

			data, err := json.Marshal(vs)
			Expect(err).NotTo(HaveOccurred())

			str := string(data)
			Expect(str).NotTo(ContainSubstring("expires_after"))
			Expect(str).NotTo(ContainSubstring("metadata"))
		})

		It("should include expires_after when set", func() {
			vs := VectorStore{
				ID:     "vs_123",
				Object: "vector_store",
				ExpiresAfter: &ExpirationPolicy{
					Anchor: "last_active_at",
					Days:   7,
				},
			}

			data, err := json.Marshal(vs)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			expiresAfter := decoded["expires_after"].(map[string]interface{})
			Expect(expiresAfter["anchor"]).To(Equal("last_active_at"))
			Expect(expiresAfter["days"]).To(BeNumerically("==", 7))
		})
	})

	Context("VectorStoreFile", func() {
		It("should serialize with correct JSON tags", func() {
			vsf := VectorStoreFile{
				ID:            "vsf_456",
				Object:        "vector_store.file",
				VectorStoreID: "vs_123",
				FileID:        "file_789",
				Status:        "completed",
				CreatedAt:     1700000000,
			}

			data, err := json.Marshal(vsf)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["id"]).To(Equal("vsf_456"))
			Expect(decoded["object"]).To(Equal("vector_store.file"))
			Expect(decoded["vector_store_id"]).To(Equal("vs_123"))
			Expect(decoded["file_id"]).To(Equal("file_789"))
			Expect(decoded["status"]).To(Equal("completed"))
		})

		It("should include last_error when set", func() {
			vsf := VectorStoreFile{
				ID:     "vsf_456",
				Object: "vector_store.file",
				LastError: &FileError{
					Code:    "parse_error",
					Message: "failed to parse file",
				},
			}

			data, err := json.Marshal(vsf)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			lastError := decoded["last_error"].(map[string]interface{})
			Expect(lastError["code"]).To(Equal("parse_error"))
			Expect(lastError["message"]).To(Equal("failed to parse file"))
		})
	})

	Context("FileRecord", func() {
		It("should serialize with correct JSON tags", func() {
			fr := FileRecord{
				ID:        "file_abc",
				Object:    "file",
				Bytes:     12345,
				CreatedAt: 1700000000,
				Filename:  "document.txt",
				Purpose:   "assistants",
				Status:    "processed",
			}

			data, err := json.Marshal(fr)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["id"]).To(Equal("file_abc"))
			Expect(decoded["object"]).To(Equal("file"))
			Expect(decoded["bytes"]).To(BeNumerically("==", 12345))
			Expect(decoded["filename"]).To(Equal("document.txt"))
			Expect(decoded["purpose"]).To(Equal("assistants"))
		})
	})

	Context("ChunkingStrategy", func() {
		It("should serialize auto strategy", func() {
			cs := ChunkingStrategy{Type: "auto"}

			data, err := json.Marshal(cs)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["type"]).To(Equal("auto"))
			Expect(decoded).NotTo(HaveKey("static"))
		})

		It("should serialize static strategy with config", func() {
			cs := ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 800,
					ChunkOverlapTokens: 400,
				},
			}

			data, err := json.Marshal(cs)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["type"]).To(Equal("static"))
			static := decoded["static"].(map[string]interface{})
			Expect(static["max_chunk_size_tokens"]).To(BeNumerically("==", 800))
			Expect(static["chunk_overlap_tokens"]).To(BeNumerically("==", 400))
		})
	})

	Context("SearchResult", func() {
		It("should serialize with correct JSON tags", func() {
			sr := SearchResult{
				FileID:     "file_abc",
				Filename:   "doc.txt",
				Content:    "some text",
				Score:      0.95,
				ChunkIndex: 3,
			}

			data, err := json.Marshal(sr)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["file_id"]).To(Equal("file_abc"))
			Expect(decoded["filename"]).To(Equal("doc.txt"))
			Expect(decoded["content"]).To(Equal("some text"))
			Expect(decoded["score"]).To(BeNumerically("~", 0.95, 0.001))
			Expect(decoded["chunk_index"]).To(BeNumerically("==", 3))
		})
	})

	Context("EmbeddedChunk", func() {
		It("should serialize with correct JSON tags", func() {
			ec := EmbeddedChunk{
				ID:            "file_abc_chunk_0",
				FileID:        "file_abc",
				Filename:      "doc.txt",
				Content:       "chunk text",
				Embedding:     []float32{0.1, 0.2, 0.3},
				ChunkIndex:    0,
				VectorStoreID: "vs_123",
			}

			data, err := json.Marshal(ec)
			Expect(err).NotTo(HaveOccurred())

			var decoded map[string]interface{}
			err = json.Unmarshal(data, &decoded)
			Expect(err).NotTo(HaveOccurred())

			Expect(decoded["id"]).To(Equal("file_abc_chunk_0"))
			Expect(decoded["file_id"]).To(Equal("file_abc"))
			Expect(decoded["vector_store_id"]).To(Equal("vs_123"))
			Expect(decoded["chunk_index"]).To(BeNumerically("==", 0))
		})
	})
})

var _ = Describe("ID Prefix Constants", func() {
	It("should have correct prefix values", func() {
		Expect(PrefixVectorStore).To(Equal("vs_"))
		Expect(PrefixFile).To(Equal("file_"))
		Expect(PrefixVectorStoreFile).To(Equal("vsf_"))
	})

	It("should produce IDs containing UUID characters", func() {
		vsID := GenerateVectorStoreID()
		uuidPart := strings.TrimPrefix(vsID, PrefixVectorStore)
		Expect(uuidPart).To(MatchRegexp(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`))

		fileID := GenerateFileID()
		uuidPart = strings.TrimPrefix(fileID, PrefixFile)
		Expect(uuidPart).To(MatchRegexp(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`))

		vsfID := GenerateVectorStoreFileID()
		uuidPart = strings.TrimPrefix(vsfID, PrefixVectorStoreFile)
		Expect(uuidPart).To(MatchRegexp(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`))
	})
})
