package classification

import (
	"context"
	"errors"

	"github.com/mark3labs/mcp-go/mcp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("MCP Category Classifier initialization", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		cfg           *config.RouterConfig
	)

	BeforeEach(func() {
		mcpClassifier, _, cfg = newTestMCPCategoryClassifier()
	})

	It("should reject a nil config", func() {
		err := mcpClassifier.Init(nil)

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("config is nil"))
	})

	It("should reject disabled MCP configuration", func() {
		cfg.Enabled = false

		err := mcpClassifier.Init(cfg)

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not enabled"))
	})
})

var _ = Describe("MCP Category Classifier tool discovery common names", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
		cfg           *config.RouterConfig
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, cfg = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
		mcpClassifier.config = cfg
		cfg.ToolName = ""
	})

	DescribeTable("should discover the expected well-known tool",
		func(tools []mcp.Tool, expected string) {
			mockClient.getToolsResult = tools

			err := mcpClassifier.discoverClassificationTool()

			Expect(err).ToNot(HaveOccurred())
			Expect(mcpClassifier.toolName).To(Equal(expected))
		},
		Entry("classify_text", []mcp.Tool{{Name: "classify_text", Description: "Classifies text into categories"}}, "classify_text"),
		Entry("classify", []mcp.Tool{{Name: "classify", Description: "Classify text"}}, "classify"),
		Entry("categorize", []mcp.Tool{{Name: "categorize", Description: "Categorize text"}}, "categorize"),
		Entry("categorize_text", []mcp.Tool{{Name: "categorize_text", Description: "Categorize text into categories"}}, "categorize_text"),
	)

	It("should honor an explicitly configured tool name", func() {
		cfg.ToolName = "my_classifier"

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("my_classifier"))
	})
})

var _ = Describe("MCP Category Classifier tool discovery precedence", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
		cfg           *config.RouterConfig
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, cfg = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
		mcpClassifier.config = cfg
		cfg.ToolName = ""
	})

	It("should prioritize classify_text over other common names", func() {
		mockClient.getToolsResult = []mcp.Tool{
			{Name: "categorize", Description: "Categorize"},
			{Name: "classify_text", Description: "Main classifier"},
			{Name: "classify", Description: "Classify"},
		}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("classify_text"))
	})

	It("should prefer common names over pattern matches", func() {
		mockClient.getToolsResult = []mcp.Tool{
			{Name: "my_classification_tool", Description: "Custom classifier"},
			{Name: "classify", Description: "Built-in classifier"},
		}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("classify"))
	})

	It("should discover by a classification pattern in the tool name", func() {
		mockClient.getToolsResult = []mcp.Tool{{Name: "text_classification", Description: "Some description"}}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("text_classification"))
	})

	It("should discover by a classification pattern in the description", func() {
		mockClient.getToolsResult = []mcp.Tool{{Name: "analyze_text", Description: "Tool for text classification"}}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("analyze_text"))
	})

	It("should match classification patterns case-insensitively", func() {
		mockClient.getToolsResult = []mcp.Tool{{Name: "TextClassification", Description: "Classify documents"}}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("TextClassification"))
	})

	It("should match classif in descriptions case-insensitively", func() {
		mockClient.getToolsResult = []mcp.Tool{{Name: "my_tool", Description: "This tool performs Classification tasks"}}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).ToNot(HaveOccurred())
		Expect(mcpClassifier.toolName).To(Equal("my_tool"))
	})
})

var _ = Describe("MCP Category Classifier tool discovery failures", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
		cfg           *config.RouterConfig
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, cfg = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
		mcpClassifier.config = cfg
		cfg.ToolName = ""
	})

	It("should fail when no tools are available", func() {
		mockClient.getToolsResult = []mcp.Tool{}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("no tools available"))
	})

	It("should fail when no classification tool matches", func() {
		mockClient.getToolsResult = []mcp.Tool{
			{Name: "foo", Description: "Does foo"},
			{Name: "bar", Description: "Does bar"},
		}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("no classification tool found"))
	})

	It("should include available tool names in the failure message", func() {
		mockClient.getToolsResult = []mcp.Tool{
			{Name: "tool1", Description: "Does something"},
			{Name: "tool2", Description: "Does another thing"},
		}

		err := mcpClassifier.discoverClassificationTool()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("tool1"))
		Expect(err.Error()).To(ContainSubstring("tool2"))
	})
})

var _ = Describe("MCP Category Classifier close", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, _ = newTestMCPCategoryClassifier()
	})

	It("should succeed when the client is nil", func() {
		err := mcpClassifier.Close()
		Expect(err).ToNot(HaveOccurred())
	})

	It("should close the client when present", func() {
		mcpClassifier.client = mockClient

		err := mcpClassifier.Close()

		Expect(err).ToNot(HaveOccurred())
		Expect(mockClient.connected).To(BeFalse())
	})

	It("should surface close failures", func() {
		mcpClassifier.client = mockClient
		mockClient.closeError = errors.New("close failed")

		err := mcpClassifier.Close()

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("close failed"))
	})
})

var _ = Describe("MCP Category Classifier classify", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, _ = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
		mcpClassifier.toolName = "classify_text"
	})

	It("should fail when the client is not initialized", func() {
		mcpClassifier.client = nil

		_, err := mcpClassifier.Classify(context.Background(), "test")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not initialized"))
	})

	It("should surface tool call failures", func() {
		mockClient.callToolError = errors.New("tool call failed")

		_, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("tool call failed"))
	})

	It("should fail when the MCP tool returns an error result", func() {
		mockClient.callToolResult = newMCPErrorResult("error message")

		_, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("returned error"))
	})

	It("should fail when the MCP tool returns empty content", func() {
		mockClient.callToolResult = &mcp.CallToolResult{IsError: false, Content: []mcp.Content{}}

		_, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("empty content"))
	})

	It("should parse a valid classification response", func() {
		mockClient.callToolResult = newMCPTextResult(`{"class": 2, "confidence": 0.95, "model": "openai/gpt-oss-20b", "use_reasoning": true}`)

		result, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).ToNot(HaveOccurred())
		Expect(result.Class).To(Equal(2))
		Expect(result.Confidence).To(BeNumerically("~", 0.95, 0.001))
	})

	It("should parse routing metadata alongside the classification", func() {
		mockClient.callToolResult = newMCPTextResult(`{"class": 1, "confidence": 0.85, "model": "openai/gpt-oss-20b", "use_reasoning": false}`)

		result, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).ToNot(HaveOccurred())
		Expect(result.Class).To(Equal(1))
		Expect(result.Confidence).To(BeNumerically("~", 0.85, 0.001))
	})

	It("should fail on invalid JSON", func() {
		mockClient.callToolResult = newMCPTextResult(`invalid json`)

		_, err := mcpClassifier.Classify(context.Background(), "test text")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to parse"))
	})
})

var _ = Describe("MCP Category Classifier probabilities", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
	)

	BeforeEach(func() {
		mcpClassifier, mockClient, _ = newTestMCPCategoryClassifier()
		mcpClassifier.client = mockClient
		mcpClassifier.toolName = "classify_text"
	})

	It("should fail when the client is not initialized", func() {
		mcpClassifier.client = nil

		_, err := mcpClassifier.ClassifyWithProbabilities(context.Background(), "test")

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not initialized"))
	})

	It("should parse a valid probabilities response", func() {
		mockClient.callToolResult = newMCPTextResult(`{"class": 1, "confidence": 0.85, "probabilities": [0.10, 0.85, 0.05], "model": "openai/gpt-oss-20b", "use_reasoning": true}`)

		result, err := mcpClassifier.ClassifyWithProbabilities(context.Background(), "test text")

		Expect(err).ToNot(HaveOccurred())
		Expect(result.Class).To(Equal(1))
		Expect(result.Confidence).To(BeNumerically("~", 0.85, 0.001))
		Expect(result.Probabilities).To(HaveLen(3))
		Expect(result.Probabilities[0]).To(BeNumerically("~", 0.10, 0.001))
		Expect(result.Probabilities[1]).To(BeNumerically("~", 0.85, 0.001))
		Expect(result.Probabilities[2]).To(BeNumerically("~", 0.05, 0.001))
	})
})
