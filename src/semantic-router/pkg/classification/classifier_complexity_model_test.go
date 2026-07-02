package classification

import (
	"errors"
	"os"
	"path/filepath"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type mockComplexityInferenceResponse struct {
	result candle_binding.ClassResult
	err    error
}

// MockComplexityInference is a test double for the trained complexity classifier.
type MockComplexityInference struct {
	mockComplexityInferenceResponse
	responseMap map[string]mockComplexityInferenceResponse
}

func (m *MockComplexityInference) setResponse(text string, class int, confidence float32, err error) {
	m.responseMap[text] = mockComplexityInferenceResponse{
		result: candle_binding.ClassResult{Class: class, Confidence: confidence},
		err:    err,
	}
}

func (m *MockComplexityInference) Classify(text string) (candle_binding.ClassResult, error) {
	if r, ok := m.responseMap[text]; ok {
		return r.result, r.err
	}
	return m.result, m.err
}

// MockComplexityInitializer records Init calls for assertions.
type MockComplexityInitializer struct {
	InitError error
	called    bool
	modelID   string
}

func (m *MockComplexityInitializer) Init(modelID string, _ bool) error {
	m.called = true
	m.modelID = modelID
	return m.InitError
}

func difficultyMapping() *ComplexityMapping {
	return &ComplexityMapping{
		LabelToIdx: map[string]int{"easy": 0, "medium": 1, "hard": 2},
		IdxToLabel: map[string]string{"0": "easy", "1": "medium", "2": "hard"},
	}
}

func newTestComplexityModelClassifier(rules []config.ComplexityRule, mapping *ComplexityMapping, mock *MockComplexityInference, initializer *MockComplexityInitializer) *Classifier {
	cfg := &config.RouterConfig{}
	cfg.ComplexityRules = rules
	cfg.ComplexityModel.Classifier.ModelID = "test-complexity-model"
	classifier, _ := newClassifierWithOptions(cfg, withComplexityModel(mapping, initializer, mock))
	return classifier
}

func newComplexitySignalResults() (*SignalResults, *sync.Mutex) {
	return &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}, &sync.Mutex{}
}

func modelRule(name string, threshold float32) config.ComplexityRule {
	return config.ComplexityRule{Name: name, Method: config.ComplexityMethodModel, Threshold: threshold}
}

var _ = Describe("complexity model classification", func() {
	const text = "design a distributed consensus protocol"

	var (
		mock        *MockComplexityInference
		initializer *MockComplexityInitializer
	)

	BeforeEach(func() {
		mock = &MockComplexityInference{responseMap: make(map[string]mockComplexityInferenceResponse)}
		initializer = &MockComplexityInitializer{}
	})

	It("reports the predicted difficulty label for a confident prediction", func() {
		mock.setResponse(text, 2, 0.91, nil) // class 2 -> "hard"
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.5)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(ConsistOf("needs_reasoning:hard"))
		Expect(results.SignalConfidences["complexity:needs_reasoning:hard"]).To(BeNumerically("~", 0.91, 0.0001))
		Expect(results.SignalValues["complexity:needs_reasoning:margin"]).To(BeNumerically("~", 0.91, 0.0001))
	})

	It("uses a negative margin for an easy prediction", func() {
		mock.setResponse(text, 0, 0.8, nil) // class 0 -> "easy"
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.5)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(ConsistOf("needs_reasoning:easy"))
		Expect(results.SignalValues["complexity:needs_reasoning:margin"]).To(BeNumerically("~", -0.8, 0.0001))
	})

	It("falls back to the neutral medium band below the confidence floor", func() {
		mock.setResponse(text, 2, 0.6, nil) // hard, but confidence < threshold
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.9)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(ConsistOf("needs_reasoning:medium"))
		Expect(results.SignalValues["complexity:needs_reasoning:margin"]).To(BeNumerically("~", 0.0, 0.0001))
	})

	It("skips the rule when the predicted class index is not mapped", func() {
		mock.setResponse(text, 7, 0.95, nil) // out of range for the 3-class mapping
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.5)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(BeEmpty())
	})

	It("skips the rule when inference fails", func() {
		mock.setResponse(text, 0, 0, errors.New("boom"))
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.5)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(BeEmpty())
	})

	It("evaluates every model-mode rule from a single inference", func() {
		mock.setResponse(text, 2, 0.88, nil)
		classifier := newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("rule_a", 0.5), modelRule("rule_b", 0.5)}, difficultyMapping(), mock, initializer,
		)

		results, mu := newComplexitySignalResults()
		classifier.evaluateComplexitySignal(results, mu, text, "", nil)

		Expect(results.MatchedComplexityRules).To(ConsistOf("rule_a:hard", "rule_b:hard"))
	})
})

var _ = Describe("complexity model runtime initialization", func() {
	newClassifier := func(mapping *ComplexityMapping) (*Classifier, *MockComplexityInitializer) {
		initializer := &MockComplexityInitializer{}
		mock := &MockComplexityInference{responseMap: make(map[string]mockComplexityInferenceResponse)}
		return newTestComplexityModelClassifier(
			[]config.ComplexityRule{modelRule("needs_reasoning", 0.5)}, mapping, mock, initializer,
		), initializer
	}

	It("is enabled and initializes the model when configured", func() {
		classifier, initializer := newClassifier(difficultyMapping())
		Expect(classifier.IsComplexityModelEnabled()).To(BeTrue())
		Expect(classifier.initializeComplexityModelClassifier()).To(Succeed())
		Expect(initializer.called).To(BeTrue())
		Expect(initializer.modelID).To(Equal("test-complexity-model"))
	})

	It("is disabled and errors on init when the mapping is missing", func() {
		classifier, _ := newClassifier(nil)
		Expect(classifier.IsComplexityModelEnabled()).To(BeFalse())
		Expect(classifier.initializeComplexityModelClassifier()).To(HaveOccurred())
	})
})

var _ = Describe("complexity mapping", func() {
	It("maps class indices to difficulty labels", func() {
		mapping := difficultyMapping()
		label, ok := mapping.GetDifficultyFromIndex(2)
		Expect(ok).To(BeTrue())
		Expect(label).To(Equal("hard"))

		_, ok = mapping.GetDifficultyFromIndex(9)
		Expect(ok).To(BeFalse())
	})

	It("loads HuggingFace id2label naming", func() {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "complexity_mapping.json")
		Expect(os.WriteFile(path, []byte(`{"id_to_label": {"0": "easy", "1": "medium", "2": "hard"}}`), 0o600)).To(Succeed())

		mapping, err := LoadComplexityMapping(path)
		Expect(err).ToNot(HaveOccurred())

		label, ok := mapping.GetDifficultyFromIndex(1)
		Expect(ok).To(BeTrue())
		Expect(label).To(Equal("medium"))
	})

	It("loads the category-classifier mapping convention (idx_to_category)", func() {
		// Matches category_mapping.json shipped alongside merged classifier checkpoints,
		// including the 0=easy,1=hard,2=medium index ordering.
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "category_mapping.json")
		Expect(os.WriteFile(path, []byte(`{"category_to_idx":{"easy":0,"hard":1,"medium":2},"idx_to_category":{"0":"easy","1":"hard","2":"medium"}}`), 0o600)).To(Succeed())

		mapping, err := LoadComplexityMapping(path)
		Expect(err).ToNot(HaveOccurred())

		label, ok := mapping.GetDifficultyFromIndex(1)
		Expect(ok).To(BeTrue())
		Expect(label).To(Equal("hard"))
	})
})

var _ = Describe("complexity rule method helpers", func() {
	It("detects model-mode rules", func() {
		Expect(config.ComplexityRule{Method: "model"}.UsesModel()).To(BeTrue())
		Expect(config.ComplexityRule{}.UsesModel()).To(BeFalse())
		Expect(config.ComplexityRule{Method: "embedding"}.UsesModel()).To(BeFalse())

		Expect(config.HasModelComplexityRule([]config.ComplexityRule{{Method: "model"}})).To(BeTrue())
		Expect(config.HasModelComplexityRule([]config.ComplexityRule{{}})).To(BeFalse())
	})
})

var _ = Describe("complexity model construction gating", func() {
	var mappingPath string

	BeforeEach(func() {
		dir := GinkgoT().TempDir()
		mappingPath = filepath.Join(dir, "complexity_mapping.json")
		Expect(os.WriteFile(mappingPath, []byte(`{"idx_to_label":{"0":"easy","1":"medium","2":"hard"}}`), 0o600)).To(Succeed())
	})

	buildWith := func(rules []config.ComplexityRule, modelID, mapPath string) (*Classifier, error) {
		cfg := &config.RouterConfig{}
		cfg.ComplexityRules = rules
		cfg.ComplexityModel.Classifier.ModelID = modelID
		cfg.ComplexityModel.Classifier.ComplexityMappingPath = mapPath
		return BuildClassifier(cfg, nil, nil, nil)
	}

	It("wires the model classifier when a rule opts in and a model is configured", func() {
		c, err := buildWith([]config.ComplexityRule{modelRule("r", 0.5)}, "test-model", mappingPath)
		Expect(err).ToNot(HaveOccurred())
		Expect(c.IsComplexityModelEnabled()).To(BeTrue())
	})

	It("stays inert when no rule opts into model mode", func() {
		c, err := buildWith([]config.ComplexityRule{{Name: "r"}}, "test-model", mappingPath)
		Expect(err).ToNot(HaveOccurred())
		Expect(c.IsComplexityModelEnabled()).To(BeFalse())
	})

	It("errors when a model rule and model_id are set but the mapping path is empty", func() {
		_, err := buildWith([]config.ComplexityRule{modelRule("r", 0.5)}, "test-model", "")
		Expect(err).To(HaveOccurred())
	})
})

// End-to-end-ish coverage: drives the real EvaluateAllSignals pipeline (used-signal
// gating -> readiness -> dispatcher -> evaluateComplexitySignal -> model path) with the
// inference seam mocked, so the full routing-signal path is exercised without a model.
var _ = Describe("complexity model signal pipeline (EvaluateAllSignals)", func() {
	const query = "design a fault-tolerant distributed consensus protocol"

	// Build a classifier whose complexity signal is referenced by a decision (so it is
	// treated as a "used" signal) and backed by the mocked trained classifier.
	buildPipeline := func(mock *MockComplexityInference) *Classifier {
		cfg := &config.RouterConfig{}
		cfg.ComplexityRules = []config.ComplexityRule{modelRule("needs_reasoning", 0.5)}
		cfg.ComplexityModel.Classifier.ModelID = "test-complexity-model"
		cfg.Decisions = []config.Decision{{
			Name: "escalate_reasoning",
			Rules: config.RuleCombination{
				Operator:   "OR",
				Conditions: []config.RuleCondition{{Type: config.SignalTypeComplexity, Name: "needs_reasoning:hard"}},
			},
		}}
		c, err := newClassifierWithOptions(cfg,
			withComplexityModel(difficultyMapping(), &MockComplexityInitializer{}, mock))
		Expect(err).ToNot(HaveOccurred())
		return c
	}

	mockReturning := func(class int, confidence float32) *MockComplexityInference {
		m := &MockComplexityInference{responseMap: make(map[string]mockComplexityInferenceResponse)}
		m.setResponse(query, class, confidence, nil)
		return m
	}

	It("emits the model-predicted difficulty end to end for a hard prompt", func() {
		c := buildPipeline(mockReturning(2, 0.9)) // class 2 -> hard

		results := c.EvaluateAllSignals(query)

		Expect(results).ToNot(BeNil())
		Expect(results.MatchedComplexityRules).To(ContainElement("needs_reasoning:hard"))
		Expect(results.SignalConfidences["complexity:needs_reasoning:hard"]).To(BeNumerically("~", 0.9, 0.0001))
	})

	It("emits easy end to end for an easy prompt", func() {
		c := buildPipeline(mockReturning(0, 0.95)) // class 0 -> easy

		results := c.EvaluateAllSignals(query)

		Expect(results.MatchedComplexityRules).To(ContainElement("needs_reasoning:easy"))
	})
})
