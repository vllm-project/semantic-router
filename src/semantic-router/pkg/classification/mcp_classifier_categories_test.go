package classification

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func expectCategoryPrompt(mapping *CategoryMapping, category, expected string, expectedOK bool) {
	prompt, ok := mapping.GetCategorySystemPrompt(category)
	Expect(ok).To(Equal(expectedOK))
	Expect(prompt).To(Equal(expected))
}

func expectCategoryDescription(mapping *CategoryMapping, category, expected string, expectedOK bool) {
	description, ok := mapping.GetCategoryDescription(category)
	Expect(ok).To(Equal(expectedOK))
	Expect(description).To(Equal(expected))
}

func expectClassifierPrompt(classifier *Classifier, category, expected string, expectedOK bool) {
	prompt, ok := classifier.GetCategorySystemPrompt(category)
	Expect(ok).To(Equal(expectedOK))
	Expect(prompt).To(Equal(expected))
}

func expectClassifierDescription(classifier *Classifier, category, expected string, expectedOK bool) {
	description, ok := classifier.GetCategoryDescription(category)
	Expect(ok).To(Equal(expectedOK))
	Expect(description).To(Equal(expected))
}

var _ = Describe("MCP Category Classifier list categories", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, _ = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
	})

	It("should fail when the client is not initialized", func() {
		mcpClassifier.client = nil

		_, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not initialized"))
	})

	It("should parse a valid category list", func() {
		mockClient.callToolResult = newMCPTextResult(`{"categories": ["math", "science", "technology", "history", "general"]}`)

		mapping, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).ToNot(HaveOccurred())
		Expect(mapping).ToNot(BeNil())
		Expect(mapping.CategoryToIdx).To(HaveLen(5))
		Expect(mapping.CategoryToIdx["math"]).To(Equal(0))
		Expect(mapping.CategoryToIdx["science"]).To(Equal(1))
		Expect(mapping.CategoryToIdx["technology"]).To(Equal(2))
		Expect(mapping.CategoryToIdx["history"]).To(Equal(3))
		Expect(mapping.CategoryToIdx["general"]).To(Equal(4))
		Expect(mapping.IdxToCategory["0"]).To(Equal("math"))
		Expect(mapping.IdxToCategory["4"]).To(Equal("general"))
	})

	It("should return an empty mapping for an empty category list", func() {
		mockClient.callToolResult = newMCPTextResult(`{"categories": []}`)

		mapping, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).ToNot(HaveOccurred())
		Expect(mapping).ToNot(BeNil())
		Expect(mapping.CategoryToIdx).To(HaveLen(0))
		Expect(mapping.IdxToCategory).To(HaveLen(0))
	})

	It("should fail when the MCP tool returns an error", func() {
		mockClient.callToolResult = newMCPErrorResult("error loading categories")

		_, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("returned error"))
	})

	It("should fail on invalid JSON", func() {
		mockClient.callToolResult = newMCPTextResult(`invalid json`)

		_, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to parse"))
	})
})

var _ = Describe("MCP Category Classifier category metadata", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, _ = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
	})

	It("should preserve per-category system prompts and descriptions", func() {
		mockClient.callToolResult = newMCPTextResult(`{
			"categories": ["math", "science", "technology"],
			"category_system_prompts": {
				"math": "You are a mathematics expert. Show step-by-step solutions.",
				"science": "You are a science expert. Provide evidence-based answers.",
				"technology": "You are a technology expert. Include practical examples."
			},
			"category_descriptions": {
				"math": "Mathematical and computational queries",
				"science": "Scientific concepts and queries",
				"technology": "Technology and computing topics"
			}
		}`)

		mapping, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).ToNot(HaveOccurred())
		Expect(mapping.CategorySystemPrompts).To(HaveLen(3))
		Expect(mapping.CategoryDescriptions).To(HaveLen(3))
		expectCategoryPrompt(mapping, "math", "You are a mathematics expert. Show step-by-step solutions.", true)
		expectCategoryPrompt(mapping, "science", "You are a science expert. Provide evidence-based answers.", true)
		expectCategoryDescription(mapping, "math", "Mathematical and computational queries", true)
	})

	It("should tolerate missing system prompts", func() {
		mockClient.callToolResult = newMCPTextResult(`{"categories": ["math", "science"]}`)

		mapping, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).ToNot(HaveOccurred())
		Expect(mapping.CategoryToIdx).To(HaveLen(2))
		expectCategoryPrompt(mapping, "math", "", false)
	})

	It("should retain only the provided partial system prompts", func() {
		mockClient.callToolResult = newMCPTextResult(`{
			"categories": ["math", "science", "history"],
			"category_system_prompts": {
				"math": "You are a mathematics expert.",
				"science": "You are a science expert."
			}
		}`)

		mapping, err := mcpClassifier.ListCategories(context.Background())

		Expect(err).ToNot(HaveOccurred())
		Expect(mapping.CategoryToIdx).To(HaveLen(3))
		Expect(mapping.CategorySystemPrompts).To(HaveLen(2))
		expectCategoryPrompt(mapping, "math", "You are a mathematics expert.", true)
		expectCategoryPrompt(mapping, "history", "", false)
	})
})

var _ = Describe("CategoryMapping system prompt methods", func() {
	var mapping *CategoryMapping

	BeforeEach(func() {
		mapping = &CategoryMapping{
			CategoryToIdx: map[string]int{"math": 0, "science": 1, "tech": 2},
			IdxToCategory: map[string]string{"0": "math", "1": "science", "2": "tech"},
			CategorySystemPrompts: map[string]string{
				"math":    "You are a mathematics expert. Show step-by-step solutions.",
				"science": "You are a science expert. Provide evidence-based answers.",
			},
			CategoryDescriptions: map[string]string{
				"math":    "Mathematical queries",
				"science": "Scientific queries",
				"tech":    "Technology queries",
			},
		}
	})

	It("should return a system prompt when present", func() {
		expectCategoryPrompt(mapping, "math", "You are a mathematics expert. Show step-by-step solutions.", true)
	})

	It("should return false when a category has no system prompt", func() {
		expectCategoryPrompt(mapping, "tech", "", false)
	})

	It("should return false for missing categories", func() {
		expectCategoryPrompt(mapping, "nonexistent", "", false)
		expectCategoryDescription(mapping, "nonexistent", "", false)
	})

	It("should return false when system prompts are nil", func() {
		mapping.CategorySystemPrompts = nil
		expectCategoryPrompt(mapping, "math", "", false)
	})

	It("should return descriptions when present", func() {
		expectCategoryDescription(mapping, "math", "Mathematical queries", true)
	})
})

var _ = Describe("MCP helper functions", func() {
	It("should create an MCP category initializer", func() {
		initializer := createMCPCategoryInitializer()
		Expect(initializer).ToNot(BeNil())
		_, ok := initializer.(*MCPCategoryClassifier)
		Expect(ok).To(BeTrue())
	})

	It("should create inference from an MCP initializer", func() {
		initializer := &MCPCategoryClassifier{}
		inference := createMCPCategoryInference(initializer)
		Expect(inference).To(Equal(initializer))
	})

	It("should return nil for a non-MCP initializer", func() {
		type fakeInitializer struct{}
		fakeInit := struct {
			fakeInitializer
			MCPCategoryInitializer
		}{}

		Expect(createMCPCategoryInference(&fakeInit)).To(BeNil())
	})

	It("should wire MCP fields onto the classifier", func() {
		classifier := &Classifier{}
		initializer := &MCPCategoryClassifier{}
		inference := createMCPCategoryInference(initializer)

		withMCPCategory(initializer, inference)(classifier)

		Expect(classifier.mcpCategoryInitializer).To(Equal(initializer))
		Expect(classifier.mcpCategoryInference).To(Equal(inference))
	})
})

var _ = Describe("Classifier per-category system prompts", func() {
	var classifier *Classifier

	BeforeEach(func() {
		cfg := &config.RouterConfig{}
		cfg.Enabled = true

		classifier = &Classifier{
			Config: cfg,
			CategoryMapping: &CategoryMapping{
				CategoryToIdx: map[string]int{"math": 0, "science": 1, "tech": 2},
				IdxToCategory: map[string]string{"0": "math", "1": "science", "2": "tech"},
				CategorySystemPrompts: map[string]string{
					"math":    "You are a mathematics expert. Show step-by-step solutions with clear explanations.",
					"science": "You are a science expert. Provide evidence-based answers grounded in research.",
					"tech":    "You are a technology expert. Include practical examples and code snippets.",
				},
				CategoryDescriptions: map[string]string{
					"math":    "Mathematical and computational queries",
					"science": "Scientific concepts and queries",
					"tech":    "Technology and computing topics",
				},
			},
		}
	})

	It("should return the category-specific system prompt", func() {
		prompt, ok := classifier.GetCategorySystemPrompt("math")
		Expect(ok).To(BeTrue())
		Expect(prompt).To(ContainSubstring("mathematics expert"))
		Expect(prompt).To(ContainSubstring("step-by-step solutions"))
	})

	It("should preserve distinct prompts per category", func() {
		mathPrompt, mathOK := classifier.GetCategorySystemPrompt("math")
		sciencePrompt, scienceOK := classifier.GetCategorySystemPrompt("science")
		techPrompt, techOK := classifier.GetCategorySystemPrompt("tech")

		Expect(mathOK).To(BeTrue())
		Expect(scienceOK).To(BeTrue())
		Expect(techOK).To(BeTrue())
		Expect(mathPrompt).ToNot(Equal(sciencePrompt))
		Expect(mathPrompt).ToNot(Equal(techPrompt))
		Expect(sciencePrompt).ToNot(Equal(techPrompt))
	})

	It("should return empty values for missing categories or mappings", func() {
		expectClassifierPrompt(classifier, "nonexistent", "", false)
		expectClassifierDescription(classifier, "nonexistent", "", false)

		classifier.CategoryMapping = nil
		expectClassifierPrompt(classifier, "math", "", false)
		expectClassifierDescription(classifier, "math", "", false)
	})

	It("should return category descriptions", func() {
		expectClassifierDescription(classifier, "math", "Mathematical and computational queries", true)
	})
})
