package classification

import (
	"errors"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type MockPIIInitializer struct{ InitError error }

func (m *MockPIIInitializer) Init(_ string, useCPU bool, numClasses int) error {
	return m.InitError
}

type MockPIIInferenceResponse struct {
	classifyTokensResult candle_binding.TokenClassificationResult
	classifyTokensError  error
}

type MockPIIInference struct {
	MockPIIInferenceResponse
	responseMap map[string]MockPIIInferenceResponse
}

func (m *MockPIIInference) setMockResponse(text string, entities []candle_binding.TokenEntity, err error) {
	m.responseMap[text] = MockPIIInferenceResponse{
		classifyTokensResult: candle_binding.TokenClassificationResult{Entities: entities},
		classifyTokensError:  err,
	}
}

func (m *MockPIIInference) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	if response, exists := m.responseMap[text]; exists {
		return response.classifyTokensResult, response.classifyTokensError
	}
	return m.classifyTokensResult, m.classifyTokensError
}

func newTestPIIClassifier() (*Classifier, *MockPIIInitializer, *MockPIIInference) {
	mockInitializer := &MockPIIInitializer{}
	mockModel := &MockPIIInference{
		responseMap: make(map[string]MockPIIInferenceResponse),
	}

	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = "test-pii-model"
	cfg.PIIMappingPath = "test-pii-mapping-path"
	cfg.PIIModel.Threshold = 0.7

	classifier, _ := newClassifierWithOptions(cfg,
		withPII(&PIIMapping{
			LabelToIdx: map[string]int{"PERSON": 0, "EMAIL": 1},
			IdxToLabel: map[string]string{"0": "PERSON", "1": "EMAIL"},
		}, mockInitializer, mockModel),
	)

	return classifier, mockInitializer, mockModel
}

func piiEntity(entityType, text string, start, end int, confidence float32) candle_binding.TokenEntity {
	return candle_binding.TokenEntity{
		EntityType: entityType,
		Text:       text,
		Start:      start,
		End:        end,
		Confidence: confidence,
	}
}

type translatePIITypeTestCase struct {
	name    string
	mapping *PIIMapping
	input   string
	want    string
}

func translatePIITypeTestCases() []translatePIITypeTestCase {
	bioTaggedMapping := &PIIMapping{
		LabelToIdx: map[string]int{
			"B-PERSON": 1, "I-PERSON": 2,
			"B-EMAIL_ADDRESS": 3, "I-EMAIL_ADDRESS": 4,
		},
		IdxToLabel: map[string]string{
			"1": "B-PERSON", "2": "I-PERSON",
			"3": "B-EMAIL_ADDRESS", "4": "I-EMAIL_ADDRESS",
		},
	}
	plainMapping := &PIIMapping{
		LabelToIdx: map[string]int{"PERSON": 1, "EMAIL_ADDRESS": 2},
		IdxToLabel: map[string]string{"1": "PERSON", "2": "EMAIL_ADDRESS"},
	}

	return []translatePIITypeTestCase{
		{name: "LABEL_2 with BIO mapping returns bare PERSON", mapping: bioTaggedMapping, input: "LABEL_2", want: "PERSON"},
		{name: "LABEL_1 with BIO mapping returns bare PERSON", mapping: bioTaggedMapping, input: "LABEL_1", want: "PERSON"},
		{name: "LABEL_4 with BIO mapping returns bare EMAIL_ADDRESS", mapping: bioTaggedMapping, input: "LABEL_4", want: "EMAIL_ADDRESS"},
		{name: "class_2 with BIO mapping returns bare PERSON", mapping: bioTaggedMapping, input: "class_2", want: "PERSON"},
		{name: "I-PERSON raw input stripped to PERSON", mapping: plainMapping, input: "I-PERSON", want: "PERSON"},
		{name: "B-EMAIL_ADDRESS raw input stripped to EMAIL_ADDRESS", mapping: plainMapping, input: "B-EMAIL_ADDRESS", want: "EMAIL_ADDRESS"},
		{name: "PERSON passes through plain mapping unchanged", mapping: plainMapping, input: "PERSON", want: "PERSON"},
		{name: "nil mapping strips I-PERSON", mapping: nil, input: "I-PERSON", want: "PERSON"},
	}
}

func TestTranslatePIIType(t *testing.T) {
	for _, tt := range translatePIITypeTestCases() {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.mapping.TranslatePIIType(tt.input)
			if got != tt.want {
				t.Errorf("TranslatePIIType(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

var _ = Describe("PII detection initialization", func() {
	var (
		classifier      *Classifier
		mockInitializer *MockPIIInitializer
	)

	BeforeEach(func() {
		classifier, mockInitializer, _ = newTestPIIClassifier()
	})

	It("should succeed", func() {
		err := classifier.initializePIIClassifier()
		Expect(err).ToNot(HaveOccurred())
	})

	It("should fail when the PII mapping is missing", func() {
		classifier.PIIMapping = nil

		err := classifier.initializePIIClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
	})

	It("should fail when there are not enough PII classes", func() {
		classifier.PIIMapping = &PIIMapping{
			LabelToIdx: map[string]int{"PERSON": 0},
			IdxToLabel: map[string]string{"0": "PERSON"},
		}

		err := classifier.initializePIIClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not enough PII types for classification"))
	})

	It("should surface initializer failures", func() {
		mockInitializer.InitError = errors.New("initialize PII classifier failed")

		err := classifier.initializePIIClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("initialize PII classifier failed"))
	})
})

var _ = Describe("PII detection configuration", func() {
	var classifier *Classifier

	BeforeEach(func() {
		classifier, _, _ = newTestPIIClassifier()
	})

	type configRow struct {
		modelID        string
		piiMappingPath string
		piiMapping     *PIIMapping
	}

	DescribeTable("should reject incomplete configuration",
		func(row configRow) {
			classifier.Config.PIIModel.ModelID = row.modelID
			classifier.Config.PIIMappingPath = row.piiMappingPath
			classifier.PIIMapping = row.piiMapping

			piiTypes, err := classifier.ClassifyPII("Some text")

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
			Expect(piiTypes).To(BeEmpty())
		},
		Entry("when model ID is empty", configRow{modelID: ""}),
		Entry("when mapping path is empty", configRow{piiMappingPath: ""}),
		Entry("when mapping is nil", configRow{piiMapping: nil}),
	)

	It("should ignore empty text", func() {
		piiTypes, err := classifier.ClassifyPII("")

		Expect(err).ToNot(HaveOccurred())
		Expect(piiTypes).To(BeEmpty())
	})
})

var _ = Describe("PII classification", func() {
	var (
		classifier *Classifier
		mockModel  *MockPIIInference
	)

	BeforeEach(func() {
		classifier, _, mockModel = newTestPIIClassifier()
	})

	It("should return detected PII types above the configured threshold", func() {
		mockModel.classifyTokensResult = candle_binding.TokenClassificationResult{
			Entities: []candle_binding.TokenEntity{
				piiEntity("PERSON", "John Doe", 0, 8, 0.9),
				piiEntity("EMAIL", "john@example.com", 9, 25, 0.8),
			},
		}

		piiTypes, err := classifier.ClassifyPII("John Doe john@example.com")

		Expect(err).ToNot(HaveOccurred())
		Expect(piiTypes).To(ConsistOf("PERSON", "EMAIL"))
	})

	It("should filter out entities below the configured threshold", func() {
		mockModel.classifyTokensResult = candle_binding.TokenClassificationResult{
			Entities: []candle_binding.TokenEntity{
				piiEntity("PERSON", "John Doe", 0, 8, 0.9),
				piiEntity("EMAIL", "john@example.com", 9, 25, 0.5),
			},
		}

		piiTypes, err := classifier.ClassifyPII("John Doe john@example.com")

		Expect(err).ToNot(HaveOccurred())
		Expect(piiTypes).To(ConsistOf("PERSON"))
	})

	It("should return an empty result when no PII is detected", func() {
		mockModel.classifyTokensResult = candle_binding.TokenClassificationResult{
			Entities: []candle_binding.TokenEntity{},
		}

		piiTypes, err := classifier.ClassifyPII("Some text")

		Expect(err).ToNot(HaveOccurred())
		Expect(piiTypes).To(BeEmpty())
	})

	It("should surface model inference failures", func() {
		mockModel.classifyTokensError = errors.New("PII model inference failed")

		piiTypes, err := classifier.ClassifyPII("Some text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("PII token classification error"))
		Expect(piiTypes).To(BeNil())
	})
})

var _ = Describe("PII content analysis", func() {
	var (
		classifier *Classifier
		mockModel  *MockPIIInference
	)

	BeforeEach(func() {
		classifier, _, mockModel = newTestPIIClassifier()
	})

	It("should fail when the PII mapping is missing", func() {
		classifier.PIIMapping = nil

		hasPII, _, err := classifier.AnalyzeContentForPII([]string{"Some text"})

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
		Expect(hasPII).To(BeFalse())
	})

	It("should skip empty texts and inference failures while retaining valid results", func() {
		mockModel.setMockResponse("Bob", []candle_binding.TokenEntity{}, errors.New("model inference failed"))
		mockModel.setMockResponse("Lisa Smith", []candle_binding.TokenEntity{
			piiEntity("PERSON", "Lisa", 0, 4, 0.3),
		}, nil)
		mockModel.setMockResponse("Alice Smith", []candle_binding.TokenEntity{
			piiEntity("PERSON", "Alice", 0, 5, 0.9),
		}, nil)
		mockModel.setMockResponse("No PII here", []candle_binding.TokenEntity{}, nil)
		mockModel.setMockResponse("", []candle_binding.TokenEntity{}, nil)

		hasPII, results, err := classifier.AnalyzeContentForPII([]string{"Bob", "Lisa Smith", "Alice Smith", "No PII here", ""})

		Expect(err).ToNot(HaveOccurred())
		Expect(hasPII).To(BeTrue())
		Expect(results).To(HaveLen(3))
		Expect(results[0].HasPII).To(BeFalse())
		Expect(results[0].Entities).To(BeEmpty())
		Expect(results[1].HasPII).To(BeTrue())
		Expect(results[1].Entities).To(HaveLen(1))
		Expect(results[1].Entities[0].EntityType).To(Equal("PERSON"))
		Expect(results[1].Entities[0].Text).To(Equal("Alice"))
		Expect(results[2].HasPII).To(BeFalse())
		Expect(results[2].Entities).To(BeEmpty())
	})
})

var _ = Describe("PII content detection", func() {
	var (
		classifier *Classifier
		mockModel  *MockPIIInference
	)

	BeforeEach(func() {
		classifier, _, mockModel = newTestPIIClassifier()
	})

	It("should return the union of detected PII types", func() {
		mockModel.setMockResponse("Bob", []candle_binding.TokenEntity{}, errors.New("model inference failed"))
		mockModel.setMockResponse("Lisa Smith", []candle_binding.TokenEntity{
			piiEntity("PERSON", "Lisa", 0, 4, 0.8),
		}, nil)
		mockModel.setMockResponse("Alice Smith alice@example.com", []candle_binding.TokenEntity{
			piiEntity("PERSON", "Alice", 0, 5, 0.9),
			piiEntity("EMAIL", "alice@example.com", 12, 29, 0.9),
		}, nil)
		mockModel.setMockResponse("No PII here", []candle_binding.TokenEntity{}, nil)
		mockModel.setMockResponse("", []candle_binding.TokenEntity{}, nil)

		detectedPII := classifier.DetectPIIInContent([]string{"Bob", "Lisa Smith", "Alice Smith alice@example.com", "No PII here", ""})

		Expect(detectedPII).To(ConsistOf("PERSON", "EMAIL"))
	})
})
