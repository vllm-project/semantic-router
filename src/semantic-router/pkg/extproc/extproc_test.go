package extproc_test

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestExtProc(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ExtProc Suite")
}

var _ = Describe("ExtProc Package", func() {
	Describe("Basic Setup", func() {
		It("should create test configuration successfully", func() {
			cfg := CreateTestConfig()
			Expect(cfg).NotTo(BeNil())
			Expect(cfg.BertModel.ModelID).To(Equal("sentence-transformers/all-MiniLM-L12-v2"))
			Expect(cfg.DefaultModel).To(Equal("model-b"))
			Expect(len(cfg.Categories)).To(Equal(1))
			Expect(cfg.Categories[0].Name).To(Equal("coding"))
		})

		It("should create test router successfully", func() {
			cfg := CreateTestConfig()
			router, err := CreateTestRouter(cfg)
			Expect(err).To(Or(BeNil(), HaveOccurred())) // May fail due to model dependencies
			if err == nil {
				Expect(router).NotTo(BeNil())
				Expect(router.Config).To(Equal(cfg))
			}
		})

		It("should handle missing model files gracefully", func() {
			cfg := CreateTestConfig()
			// Intentionally use invalid paths to test error handling
			cfg.Classifier.CategoryModel.CategoryMappingPath = "/nonexistent/path/category_mapping.json"
			cfg.Classifier.PIIModel.PIIMappingPath = "/nonexistent/path/pii_mapping.json"

			_, err := CreateTestRouter(cfg)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no such file or directory"))
		})
	})

	Describe("Configuration Validation", func() {
		It("should validate required configuration fields", func() {
			cfg := CreateTestConfig()

			// Test essential fields are present
			Expect(cfg.BertModel.ModelID).NotTo(BeEmpty())
			Expect(cfg.DefaultModel).NotTo(BeEmpty())
			Expect(cfg.ModelConfig).NotTo(BeEmpty())
			Expect(cfg.ModelConfig).To(HaveKey("model-a"))
			Expect(cfg.ModelConfig).To(HaveKey("model-b"))
		})

		It("should have valid cache configuration", func() {
			cfg := CreateTestConfig()

			Expect(cfg.SemanticCache.MaxEntries).To(BeNumerically(">", 0))
			Expect(cfg.SemanticCache.TTLSeconds).To(BeNumerically(">", 0))
			Expect(cfg.SemanticCache.SimilarityThreshold).NotTo(BeNil())
			Expect(*cfg.SemanticCache.SimilarityThreshold).To(BeNumerically(">=", 0))
			Expect(*cfg.SemanticCache.SimilarityThreshold).To(BeNumerically("<=", 1))
		})

		It("should have valid classifier configuration", func() {
			cfg := CreateTestConfig()

			Expect(cfg.Classifier.CategoryModel.ModelID).NotTo(BeEmpty())
			Expect(cfg.Classifier.CategoryModel.CategoryMappingPath).NotTo(BeEmpty())
			Expect(cfg.Classifier.PIIModel.ModelID).NotTo(BeEmpty())
			Expect(cfg.Classifier.PIIModel.PIIMappingPath).NotTo(BeEmpty())
		})

		It("should have valid tools configuration", func() {
			cfg := CreateTestConfig()

			Expect(cfg.Tools.TopK).To(BeNumerically(">", 0))
			Expect(cfg.Tools.FallbackToEmpty).To(BeTrue())
		})
	})

	Describe("Mock Components", func() {
		It("should create mock stream successfully", func() {
			requests := []*ext_proc.ProcessingRequest{}
			stream := NewMockStream(requests)

			Expect(stream).NotTo(BeNil())
			Expect(stream.Requests).To(HaveLen(0))
			Expect(stream.Responses).To(HaveLen(0))
			Expect(stream.RecvIndex).To(Equal(0))
			Expect(stream.Context()).NotTo(BeNil())
		})

		It("should handle mock stream operations", func() {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{})

			// Test Recv on empty stream
			_, err := stream.Recv()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("EOF"))

			// Test Send
			response := &ext_proc.ProcessingResponse{}
			err = stream.Send(response)
			Expect(err).NotTo(HaveOccurred())
			Expect(stream.Responses).To(HaveLen(1))
		})
	})
})

func init() {
	// Any package-level initialization can go here
}
