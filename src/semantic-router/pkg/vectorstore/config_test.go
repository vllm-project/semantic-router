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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("VectorStoreConfig", func() {
	Context("Validate", func() {
		It("should pass when disabled", func() {
			cfg := &config.VectorStoreConfig{Enabled: false}
			Expect(cfg.Validate()).NotTo(HaveOccurred())
		})

		It("should fail when enabled without backend_type", func() {
			cfg := &config.VectorStoreConfig{Enabled: true}
			Expect(cfg.Validate()).To(HaveOccurred())
		})

		It("should pass with memory backend", func() {
			cfg := &config.VectorStoreConfig{
				Enabled:     true,
				BackendType: "memory",
			}
			Expect(cfg.Validate()).NotTo(HaveOccurred())
		})

		It("should fail with milvus backend but no milvus config", func() {
			cfg := &config.VectorStoreConfig{
				Enabled:     true,
				BackendType: "milvus",
			}
			err := cfg.Validate()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("milvus configuration is required"))
		})

		It("should pass with milvus backend and config", func() {
			cfg := &config.VectorStoreConfig{
				Enabled:     true,
				BackendType: "milvus",
				Milvus:      &config.MilvusConfig{},
			}
			Expect(cfg.Validate()).NotTo(HaveOccurred())
		})

		It("should fail with unknown backend type", func() {
			cfg := &config.VectorStoreConfig{
				Enabled:     true,
				BackendType: "redis",
			}
			err := cfg.Validate()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("must be 'memory' or 'milvus'"))
		})
	})

	Context("ApplyDefaults", func() {
		It("should fill in all defaults", func() {
			cfg := &config.VectorStoreConfig{Enabled: true, BackendType: "memory"}
			cfg.ApplyDefaults()

			Expect(cfg.FileStorageDir).To(Equal("/var/lib/vsr/data"))
			Expect(cfg.MaxFileSizeMB).To(Equal(50))
			Expect(cfg.EmbeddingModel).To(Equal("bert"))
			Expect(cfg.EmbeddingDimension).To(Equal(384)) // BERT default
			Expect(cfg.IngestionWorkers).To(Equal(2))
			Expect(cfg.SupportedFormats).To(ContainElements(".txt", ".md", ".json", ".csv", ".html"))
		})

		It("should not override existing values", func() {
			cfg := &config.VectorStoreConfig{
				Enabled:            true,
				BackendType:        "memory",
				FileStorageDir:     "/custom/path",
				MaxFileSizeMB:      100,
				EmbeddingModel:     "qwen3",
				EmbeddingDimension: 1024,
				IngestionWorkers:   4,
				SupportedFormats:   []string{".txt"},
			}
			cfg.ApplyDefaults()

			Expect(cfg.FileStorageDir).To(Equal("/custom/path"))
			Expect(cfg.MaxFileSizeMB).To(Equal(100))
			Expect(cfg.EmbeddingModel).To(Equal("qwen3"))
			Expect(cfg.EmbeddingDimension).To(Equal(1024))
			Expect(cfg.IngestionWorkers).To(Equal(4))
			Expect(cfg.SupportedFormats).To(Equal([]string{".txt"}))
		})
	})
})
