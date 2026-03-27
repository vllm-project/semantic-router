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
	"encoding/binary"
	"math"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Verify ValkeyBackend satisfies the interface at compile time.
var _ VectorStoreBackend = (*ValkeyBackend)(nil)

var _ = Describe("ValkeyBackend helpers", func() {
	Context("float32SliceToBytes", func() {
		It("should produce correct little-endian bytes", func() {
			input := []float32{1.0, 2.0, 3.0}
			b := float32SliceToBytes(input)
			Expect(len(b)).To(Equal(12))
			for i, expected := range input {
				bits := binary.LittleEndian.Uint32(b[i*4:])
				Expect(math.Float32frombits(bits)).To(Equal(expected))
			}
		})
		It("should roundtrip arbitrary values", func() {
			input := []float32{0.0, -1.5, 3.14, math.MaxFloat32, math.SmallestNonzeroFloat32}
			b := float32SliceToBytes(input)
			Expect(len(b)).To(Equal(len(input) * 4))
			for i, expected := range input {
				Expect(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))).To(Equal(expected))
			}
		})
		It("should return empty slice for empty input", func() {
			Expect(float32SliceToBytes([]float32{})).To(HaveLen(0))
		})
	})

	Context("escapeTagValue", func() {
		It("should escape hyphens", func() { Expect(escapeTagValue("file-123")).To(Equal("file\\-123")) })
		It("should escape dots", func() { Expect(escapeTagValue("doc.txt")).To(Equal("doc\\.txt")) })
		It("should escape colons", func() { Expect(escapeTagValue("ns:val")).To(Equal("ns\\:val")) })
		It("should escape slashes", func() { Expect(escapeTagValue("path/to")).To(Equal("path\\/to")) })
		It("should escape spaces", func() { Expect(escapeTagValue("hello world")).To(Equal("hello\\ world")) })
		It("should escape multiple special characters", func() {
			Expect(escapeTagValue("a-b.c:d/e f")).To(Equal("a\\-b\\.c\\:d\\/e\\ f"))
		})
		It("should leave safe strings unchanged", func() { Expect(escapeTagValue("abc123")).To(Equal("abc123")) })
	})

	Context("distanceToSimilarity", func() {
		It("COSINE", func() { Expect(distanceToSimilarity("COSINE", 0.2)).To(BeNumerically("~", 0.9, 0.001)) })
		It("L2", func() { Expect(distanceToSimilarity("L2", 0.3)).To(BeNumerically("~", 0.769, 0.01)) })
		It("IP", func() { Expect(distanceToSimilarity("IP", 0.95)).To(BeNumerically("~", 0.95, 0.001)) })
		It("zero distance", func() { Expect(distanceToSimilarity("COSINE", 0.0)).To(BeNumerically("~", 1.0, 0.001)) })
		It("case-insensitive", func() { Expect(distanceToSimilarity("cosine", 0.2)).To(BeNumerically("~", 0.9, 0.001)) })
	})

	Context("toInt64", func() {
		It("int64", func() { Expect(toInt64(int64(42))).To(Equal(int64(42))) })
		It("float64", func() { Expect(toInt64(float64(42.9))).To(Equal(int64(42))) })
		It("string", func() { Expect(toInt64("123")).To(Equal(int64(123))) })
		It("nil", func() { Expect(toInt64(nil)).To(Equal(int64(0))) })
		It("unsupported", func() { Expect(toInt64(true)).To(Equal(int64(0))) })
		It("invalid string", func() { Expect(toInt64("abc")).To(Equal(int64(0))) })
	})

	Context("extractKeysFromSearchResult", func() {
		It("flat string result", func() {
			Expect(extractKeysFromSearchResult([]interface{}{int64(2), "key:1", "key:2"})).To(Equal([]string{"key:1", "key:2"}))
		})
		It("map-based result", func() {
			result := []interface{}{int64(2), map[string]interface{}{"key:1": map[string]interface{}{}, "key:2": map[string]interface{}{}}}
			Expect(extractKeysFromSearchResult(result)).To(ConsistOf("key:1", "key:2"))
		})
		It("zero results", func() { Expect(extractKeysFromSearchResult([]interface{}{int64(0)})).To(BeNil()) })
		It("non-array", func() { Expect(extractKeysFromSearchResult("not an array")).To(BeNil()) })
		It("nil", func() { Expect(extractKeysFromSearchResult(nil)).To(BeNil()) })
	})
})

var _ = Describe("ValkeyBackend naming and config", func() {
	Context("naming helpers", func() {
		var vb *ValkeyBackend
		BeforeEach(func() { vb = &ValkeyBackend{collectionPrefix: "vsr_vs_"} })
		It("indexName", func() { Expect(vb.indexName("store123")).To(Equal("vsr_vs_store123_idx")) })
		It("keyPrefix", func() { Expect(vb.keyPrefix("store123")).To(Equal("vsr_vs_store123:")) })
		It("chunkKey", func() { Expect(vb.chunkKey("store123", "c1")).To(Equal("vsr_vs_store123:c1")) })
		It("custom prefix", func() {
			vb.collectionPrefix = "custom_"
			Expect(vb.indexName("abc")).To(Equal("custom_abc_idx"))
			Expect(vb.keyPrefix("abc")).To(Equal("custom_abc:"))
			Expect(vb.chunkKey("abc", "x")).To(Equal("custom_abc:x"))
		})
	})

	Context("config defaults", func() {
		It("should apply defaults for zero-value config", func() {
			host, port, prefix, indexM, indexEf, metricType, timeout, err := valkeyDefaults(ValkeyBackendConfig{})
			Expect(err).NotTo(HaveOccurred())
			Expect(host).To(Equal("localhost"))
			Expect(port).To(Equal(6379))
			Expect(prefix).To(Equal("vsr_vs_"))
			Expect(indexM).To(Equal(16))
			Expect(indexEf).To(Equal(200))
			Expect(metricType).To(Equal("COSINE"))
			Expect(timeout).To(Equal(10))
		})
		It("should reject unsupported metric type", func() {
			_, _, _, _, _, _, _, err := valkeyDefaults(ValkeyBackendConfig{MetricType: "HAMMING"})
			Expect(err).To(HaveOccurred())
		})
	})

	Context("factory registration", func() {
		It("should fail with unreachable host", func() {
			_, err := NewBackend(BackendTypeValkey, BackendConfigs{
				Valkey: ValkeyBackendConfig{Host: "192.0.2.1", Port: 6379, ConnectTimeout: 1},
			})
			Expect(err).To(HaveOccurred())
		})
		It("should include valkey in unsupported type error", func() {
			_, err := NewBackend("nosql", BackendConfigs{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("valkey"))
		})
	})
})

var _ = Describe("ValkeyBackend parseSearchResults", func() {
	var vb *ValkeyBackend
	BeforeEach(func() { vb = &ValkeyBackend{metricType: "COSINE"} })

	It("nil input", func() {
		r, err := vb.parseSearchResults(nil, 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeNil())
	})
	It("non-array input", func() {
		r, err := vb.parseSearchResults("not an array", 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeNil())
	})
	It("zero total count", func() {
		r, err := vb.parseSearchResults([]interface{}{int64(0)}, 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeNil())
	})
	It("valid results", func() {
		result := []interface{}{int64(1), map[string]interface{}{
			"key:1": map[string]interface{}{
				"file_id": "f1", "filename": "doc.txt", "content": "hello",
				"vector_distance": "0.2", "chunk_index": "0",
			},
		}}
		r, err := vb.parseSearchResults(result, 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(1))
		Expect(r[0].FileID).To(Equal("f1"))
		Expect(r[0].Score).To(BeNumerically("~", 0.9, 0.01))
	})
	It("filter below threshold", func() {
		result := []interface{}{int64(1), map[string]interface{}{
			"key:1": map[string]interface{}{"file_id": "f1", "content": "hello", "vector_distance": "1.6"},
		}}
		r, err := vb.parseSearchResults(result, 0.5)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeEmpty())
	})
	It("skip non-map fields", func() {
		r, err := vb.parseSearchResults([]interface{}{int64(1), "not-a-map"}, 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeEmpty())
	})
	It("missing vector_distance", func() {
		result := []interface{}{int64(1), map[string]interface{}{
			"key:1": map[string]interface{}{"file_id": "f1", "content": "hello"},
		}}
		r, err := vb.parseSearchResults(result, 0.1)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(BeEmpty())
	})
	It("multiple results", func() {
		result := []interface{}{int64(2), map[string]interface{}{
			"key:1": map[string]interface{}{"file_id": "f1", "content": "hello", "vector_distance": "0.1", "chunk_index": "0"},
			"key:2": map[string]interface{}{"file_id": "f2", "content": "world", "vector_distance": "0.4", "chunk_index": "1"},
		}}
		r, err := vb.parseSearchResults(result, 0)
		Expect(err).NotTo(HaveOccurred())
		Expect(r).To(HaveLen(2))
	})
})
