package classification

import (
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type MockJailbreakInferenceResponse struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

type MockJailbreakInference struct {
	MockJailbreakInferenceResponse
	responseMap map[string]MockJailbreakInferenceResponse
}

func (m *MockJailbreakInference) setMockResponse(text string, class int, confidence float32, err error) {
	m.responseMap[text] = MockJailbreakInferenceResponse{
		classifyResult: candle_binding.ClassResult{
			Class:      class,
			Confidence: confidence,
		},
		classifyError: err,
	}
}

func (m *MockJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	if response, exists := m.responseMap[text]; exists {
		return response.classifyResult, response.classifyError
	}
	return m.classifyResult, m.classifyError
}

type MockJailbreakInitializer struct {
	InitError error
}

func (m *MockJailbreakInitializer) Init(_ string, useCPU bool, numClasses ...int) error {
	return m.InitError
}

func newTestJailbreakClassifier() (*Classifier, *MockJailbreakInitializer, *MockJailbreakInference) {
	mockInitializer := &MockJailbreakInitializer{}
	mockModel := &MockJailbreakInference{
		responseMap: make(map[string]MockJailbreakInferenceResponse),
	}

	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.ModelID = "test-model"
	cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
	cfg.PromptGuard.Threshold = 0.7

	classifier, _ := newClassifierWithOptions(cfg,
		withJailbreak(&JailbreakMapping{
			LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
			IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
		}, mockInitializer, mockModel),
	)

	return classifier, mockInitializer, mockModel
}

var _ = Describe("jailbreak detection initialization", func() {
	var (
		classifier      *Classifier
		mockInitializer *MockJailbreakInitializer
	)

	BeforeEach(func() {
		classifier, mockInitializer, _ = newTestJailbreakClassifier()
	})

	It("should succeed", func() {
		err := classifier.initializeJailbreakClassifier()
		Expect(err).ToNot(HaveOccurred())
	})

	It("should fail when the jailbreak mapping is missing", func() {
		classifier.JailbreakMapping = nil

		err := classifier.initializeJailbreakClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("jailbreak detection is not properly configured"))
	})

	It("should fail when there are not enough jailbreak classes", func() {
		classifier.JailbreakMapping = &JailbreakMapping{
			LabelToIdx: map[string]int{"jailbreak": 0},
			IdxToLabel: map[string]string{"0": "jailbreak"},
		}

		err := classifier.initializeJailbreakClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not enough jailbreak types for classification"))
	})

	It("should surface initializer failures", func() {
		mockInitializer.InitError = errors.New("initialize jailbreak classifier failed")

		err := classifier.initializeJailbreakClassifier()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("initialize jailbreak classifier failed"))
	})
})

var _ = Describe("jailbreak detection configuration", func() {
	var classifier *Classifier

	BeforeEach(func() {
		classifier, _, _ = newTestJailbreakClassifier()
	})

	type configRow struct {
		enabled              bool
		modelID              string
		jailbreakMappingPath string
		jailbreakMapping     *JailbreakMapping
	}

	DescribeTable("should reject incomplete configuration",
		func(row configRow) {
			classifier.Config.PromptGuard.Enabled = row.enabled
			classifier.Config.PromptGuard.ModelID = row.modelID
			classifier.Config.PromptGuard.JailbreakMappingPath = row.jailbreakMappingPath
			classifier.JailbreakMapping = row.jailbreakMapping

			isJailbreak, _, _, err := classifier.CheckForJailbreak("Some text")

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("jailbreak detection is not enabled or properly configured"))
			Expect(isJailbreak).To(BeFalse())
		},
		Entry("when prompt guard is disabled", configRow{enabled: false}),
		Entry("when model ID is empty", configRow{modelID: ""}),
		Entry("when mapping path is empty", configRow{jailbreakMappingPath: ""}),
		Entry("when mapping is nil", configRow{jailbreakMapping: nil}),
	)

	It("should ignore empty text", func() {
		isJailbreak, _, _, err := classifier.CheckForJailbreak("")

		Expect(err).ToNot(HaveOccurred())
		Expect(isJailbreak).To(BeFalse())
	})
})

var _ = Describe("jailbreak detection classification", func() {
	var (
		classifier *Classifier
		mockModel  *MockJailbreakInference
	)

	BeforeEach(func() {
		classifier, _, mockModel = newTestJailbreakClassifier()
	})

	It("should return jailbreak results above the configured threshold", func() {
		mockModel.classifyResult = candle_binding.ClassResult{Class: 0, Confidence: 0.9}

		isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a jailbreak attempt")

		Expect(err).ToNot(HaveOccurred())
		Expect(isJailbreak).To(BeTrue())
		Expect(jailbreakType).To(Equal("jailbreak"))
		Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
	})

	It("should return benign results above the threshold", func() {
		mockModel.classifyResult = candle_binding.ClassResult{Class: 1, Confidence: 0.9}

		isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a normal question")

		Expect(err).ToNot(HaveOccurred())
		Expect(isJailbreak).To(BeFalse())
		Expect(jailbreakType).To(Equal("benign"))
		Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
	})

	It("should return false when the jailbreak confidence is below the threshold", func() {
		mockModel.classifyResult = candle_binding.ClassResult{Class: 0, Confidence: 0.5}

		isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Ambiguous text")

		Expect(err).ToNot(HaveOccurred())
		Expect(isJailbreak).To(BeFalse())
		Expect(jailbreakType).To(Equal("jailbreak"))
		Expect(confidence).To(BeNumerically("~", 0.5, 0.001))
	})

	It("should surface inference failures", func() {
		mockModel.classifyError = errors.New("model inference failed")

		isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("jailbreak classification failed"))
		Expect(isJailbreak).To(BeFalse())
		Expect(jailbreakType).To(Equal(""))
		Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
	})

	It("should fail when the predicted class is unknown", func() {
		mockModel.classifyResult = candle_binding.ClassResult{Class: 9, Confidence: 0.9}

		isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unknown jailbreak class index"))
		Expect(isJailbreak).To(BeFalse())
		Expect(jailbreakType).To(Equal(""))
		Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
	})
})

var _ = Describe("jailbreak detection content analysis", func() {
	var (
		classifier *Classifier
		mockModel  *MockJailbreakInference
	)

	BeforeEach(func() {
		classifier, _, mockModel = newTestJailbreakClassifier()
	})

	It("should fail when the jailbreak mapping is missing", func() {
		classifier.JailbreakMapping = nil

		hasJailbreak, _, err := classifier.AnalyzeContentForJailbreak([]string{"Some text"})

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("jailbreak detection is not enabled or properly configured"))
		Expect(hasJailbreak).To(BeFalse())
	})

	It("should skip empty texts and inference failures while retaining valid results", func() {
		mockModel.setMockResponse("text0", 0, 0.9, errors.New("model inference failed"))
		mockModel.setMockResponse("text1", 0, 0.3, nil)
		mockModel.setMockResponse("text2", 1, 0.9, nil)
		mockModel.setMockResponse("text3", 0, 0.9, nil)
		mockModel.setMockResponse("", 0, 0.9, nil)

		hasJailbreak, results, err := classifier.AnalyzeContentForJailbreak([]string{"text0", "text1", "text2", "text3", ""})

		Expect(err).ToNot(HaveOccurred())
		Expect(hasJailbreak).To(BeTrue())
		Expect(results).To(HaveLen(3))
		Expect(results[0].Content).To(Equal("text1"))
		Expect(results[0].IsJailbreak).To(BeFalse())
		Expect(results[0].JailbreakType).To(Equal("jailbreak"))
		Expect(results[0].Confidence).To(BeNumerically("~", 0.3, 0.001))
		Expect(results[1].Content).To(Equal("text2"))
		Expect(results[1].IsJailbreak).To(BeFalse())
		Expect(results[1].JailbreakType).To(Equal("benign"))
		Expect(results[1].Confidence).To(BeNumerically("~", 0.9, 0.001))
		Expect(results[2].Content).To(Equal("text3"))
		Expect(results[2].IsJailbreak).To(BeTrue())
		Expect(results[2].JailbreakType).To(Equal("jailbreak"))
		Expect(results[2].Confidence).To(BeNumerically("~", 0.9, 0.001))
	})
})
