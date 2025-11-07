package classification

import (
	"errors"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKeywordEmbeddingClassifierSuite(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "KeywordEmbeddingClassifier Suite")
}

var _ = Describe("KeywordEmbeddingClassifier", func() {
	var origCalculate func(string, []string, int, string, int) (*candle_binding.BatchSimilarityOutput, error)

	BeforeEach(func() {
		origCalculate = calculateSimilarityBatch
	})

	AfterEach(func() {
		calculateSimilarityBatch = origCalculate
	})

	It("classifies with mean aggregation", func() {
		calculateSimilarityBatch = func(query string, candidates []string, topK int, modelType string, targetDim int) (*candle_binding.BatchSimilarityOutput, error) {
			return &candle_binding.BatchSimilarityOutput{Matches: []candle_binding.BatchSimilarityMatch{{Index: 0, Similarity: 0.9}, {Index: 1, Similarity: 0.8}, {Index: 2, Similarity: 0.7}}}, nil
		}

		rules := []config.EmbeddingRule{{
			Category:                  "cat1",
			Keywords:                  []string{"science", "math"},
			AggregationMethodConfiged: config.AggregationMethodMean,
			SimilarityThreshold:       0.8,
			Model:                     "auto",
			Dimension:                 768,
		}}

		clf, err := NewKeywordEmbeddingClassifier(rules)
		Expect(err).ToNot(HaveOccurred())

		cat, score, err := clf.Classify("some text")
		Expect(err).ToNot(HaveOccurred())
		Expect(cat).To(Equal("cat1"))
		Expect(score).To(BeNumerically("~", 0.8, 1e-6))
	})

	It("classifies with max aggregation", func() {
		calculateSimilarityBatch = func(query string, candidates []string, topK int, modelType string, targetDim int) (*candle_binding.BatchSimilarityOutput, error) {
			return &candle_binding.BatchSimilarityOutput{Matches: []candle_binding.BatchSimilarityMatch{{Index: 0, Similarity: 0.4}, {Index: 1, Similarity: 0.6}}}, nil
		}

		rules := []config.EmbeddingRule{{
			Category:                  "cat2",
			Keywords:                  []string{"x", "y"},
			AggregationMethodConfiged: config.AggregationMethodMax,
			SimilarityThreshold:       0.5,
			Model:                     "auto",
			Dimension:                 512,
		}}

		clf, err := NewKeywordEmbeddingClassifier(rules)
		Expect(err).ToNot(HaveOccurred())

		cat, score, err := clf.Classify("other text")
		Expect(err).ToNot(HaveOccurred())
		Expect(cat).To(Equal("cat2"))
		Expect(score).To(BeNumerically("~", 0.6, 1e-6))
	})

	It("classifies with any aggregation", func() {
		calculateSimilarityBatch = func(query string, candidates []string, topK int, modelType string, targetDim int) (*candle_binding.BatchSimilarityOutput, error) {
			return &candle_binding.BatchSimilarityOutput{Matches: []candle_binding.BatchSimilarityMatch{{Index: 0, Similarity: 0.2}, {Index: 1, Similarity: 0.95}}}, nil
		}

		rules := []config.EmbeddingRule{{
			Category:                  "cat3",
			Keywords:                  []string{"p", "q"},
			AggregationMethodConfiged: config.AggregationMethodAny,
			SimilarityThreshold:       0.7,
			Model:                     "auto",
			Dimension:                 256,
		}}

		clf, err := NewKeywordEmbeddingClassifier(rules)
		Expect(err).ToNot(HaveOccurred())

		cat, score, err := clf.Classify("third text")
		Expect(err).ToNot(HaveOccurred())
		Expect(cat).To(Equal("cat3"))
		Expect(score).To(BeNumerically("~", 0.7, 1e-6))
	})

	It("returns error when CalculateSimilarityBatch fails", func() {
		calculateSimilarityBatch = func(query string, candidates []string, topK int, modelType string, targetDim int) (*candle_binding.BatchSimilarityOutput, error) {
			return nil, errors.New("external failure")
		}

		rules := []config.EmbeddingRule{{
			Category:                  "cat4",
			Keywords:                  []string{"z"},
			AggregationMethodConfiged: config.AggregationMethodMean,
			SimilarityThreshold:       0.1,
			Model:                     "auto",
			Dimension:                 768,
		}}

		clf, err := NewKeywordEmbeddingClassifier(rules)
		Expect(err).ToNot(HaveOccurred())

		_, _, err = clf.Classify("will error")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to calculate batch similarity"))
	})
})
