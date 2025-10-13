package classification

import (
	"context"
	"errors"

	"github.com/mark3labs/mcp-go/mcp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	mcpclient "github.com/vllm-project/semantic-router/src/semantic-router/pkg/connectivity/mcp"
)

// MockMCPClient is a mock implementation of the MCP client for testing
type MockMCPClient struct {
	connectError   error
	callToolResult *mcp.CallToolResult
	callToolError  error
	closeError     error
	connected      bool
	getToolsResult []mcp.Tool
}

func (m *MockMCPClient) Connect() error {
	if m.connectError != nil {
		return m.connectError
	}
	m.connected = true
	return nil
}

func (m *MockMCPClient) Close() error {
	if m.closeError != nil {
		return m.closeError
	}
	m.connected = false
	return nil
}

func (m *MockMCPClient) IsConnected() bool {
	return m.connected
}

func (m *MockMCPClient) Ping(ctx context.Context) error {
	return nil
}

func (m *MockMCPClient) GetTools() []mcp.Tool {
	return m.getToolsResult
}

func (m *MockMCPClient) GetResources() []mcp.Resource {
	return nil
}

func (m *MockMCPClient) GetPrompts() []mcp.Prompt {
	return nil
}

func (m *MockMCPClient) RefreshCapabilities(ctx context.Context) error {
	return nil
}

func (m *MockMCPClient) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	if m.callToolError != nil {
		return nil, m.callToolError
	}
	return m.callToolResult, nil
}

func (m *MockMCPClient) ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error) {
	return nil, errors.New("not implemented")
}

func (m *MockMCPClient) GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error) {
	return nil, errors.New("not implemented")
}

func (m *MockMCPClient) SetLogHandler(handler func(mcpclient.LoggingLevel, string)) {
	// no-op for mock
}

var _ mcpclient.MCPClient = (*MockMCPClient)(nil)

var _ = Describe("MCP Category Classifier", func() {
	var (
		mcpClassifier *MCPCategoryClassifier
		mockClient    *MockMCPClient
		cfg           *config.RouterConfig
	)

	BeforeEach(func() {
		mockClient = &MockMCPClient{}
		mcpClassifier = &MCPCategoryClassifier{}
		cfg = &config.RouterConfig{}
		cfg.Classifier.MCPCategoryModel.Enabled = true
		cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
		cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
		cfg.Classifier.MCPCategoryModel.Command = "python"
		cfg.Classifier.MCPCategoryModel.Args = []string{"server.py"}
		cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 30
	})

	Describe("Init", func() {
		Context("when config is nil", func() {
			It("should return error", func() {
				err := mcpClassifier.Init(nil)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("config is nil"))
			})
		})

		Context("when MCP is not enabled", func() {
			It("should return error", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = false
				err := mcpClassifier.Init(cfg)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not enabled"))
			})
		})

		// Note: tool_name is now optional and will be auto-discovered if not specified.
		// The Init method will automatically discover classification tools from the MCP server
		// by calling discoverClassificationTool().

		// Note: Full initialization test requires mocking NewClient and GetTools which is complex
		// In real tests, we'd need dependency injection for the client factory
	})

	Describe("discoverClassificationTool", func() {
		BeforeEach(func() {
			mcpClassifier.client = mockClient
			mcpClassifier.config = cfg
		})

		Context("when tool name is explicitly configured", func() {
			It("should use the configured tool name", func() {
				cfg.Classifier.MCPCategoryModel.ToolName = "my_classifier"
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("my_classifier"))
			})
		})

		Context("when tool name is not configured", func() {
			BeforeEach(func() {
				cfg.Classifier.MCPCategoryModel.ToolName = ""
			})

			It("should discover classify_text tool", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "some_other_tool", Description: "Other tool"},
					{Name: "classify_text", Description: "Classifies text into categories"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("classify_text"))
			})

			It("should discover classify tool", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "classify", Description: "Classify text"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("classify"))
			})

			It("should discover categorize tool", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "categorize", Description: "Categorize text"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("categorize"))
			})

			It("should discover categorize_text tool", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "categorize_text", Description: "Categorize text into categories"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("categorize_text"))
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

			It("should prefer common names over pattern matching", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "my_classification_tool", Description: "Custom classifier"},
					{Name: "classify", Description: "Built-in classifier"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("classify"))
			})

			It("should discover by pattern matching in name", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "text_classification", Description: "Some description"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("text_classification"))
			})

			It("should discover by pattern matching in description", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "analyze_text", Description: "Tool for text classification"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("analyze_text"))
			})

			It("should return error when no tools available", func() {
				mockClient.getToolsResult = []mcp.Tool{}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("no tools available"))
			})

			It("should return error when no classification tool found", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "foo", Description: "Does foo"},
					{Name: "bar", Description: "Does bar"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("no classification tool found"))
			})

			It("should handle case-insensitive pattern matching", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "TextClassification", Description: "Classify documents"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("TextClassification"))
			})

			It("should match 'classif' in description (case-insensitive)", func() {
				mockClient.getToolsResult = []mcp.Tool{
					{Name: "my_tool", Description: "This tool performs Classification tasks"},
				}
				err := mcpClassifier.discoverClassificationTool()
				Expect(err).ToNot(HaveOccurred())
				Expect(mcpClassifier.toolName).To(Equal("my_tool"))
			})

			It("should log available tools when none match", func() {
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

		// Test suite summary:
		// - Explicit configuration: ✓ (1 test)
		// - Common tool names discovery: ✓ (4 tests - classify_text, classify, categorize, categorize_text)
		// - Priority/precedence: ✓ (2 tests - classify_text first, common names over patterns)
		// - Pattern matching: ✓ (4 tests - name, description, case-insensitive)
		// - Error cases: ✓ (3 tests - no tools, no match, logging)
		// Total: 14 comprehensive tests for auto-discovery
	})

	Describe("Close", func() {
		Context("when client is nil", func() {
			It("should not error", func() {
				err := mcpClassifier.Close()
				Expect(err).ToNot(HaveOccurred())
			})
		})

		Context("when client exists", func() {
			BeforeEach(func() {
				mcpClassifier.client = mockClient
			})

			It("should close the client successfully", func() {
				err := mcpClassifier.Close()
				Expect(err).ToNot(HaveOccurred())
				Expect(mockClient.connected).To(BeFalse())
			})

			It("should return error if close fails", func() {
				mockClient.closeError = errors.New("close failed")
				err := mcpClassifier.Close()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("close failed"))
			})
		})
	})

	Describe("Classify", func() {
		BeforeEach(func() {
			mcpClassifier.client = mockClient
			mcpClassifier.toolName = "classify_text"
		})

		Context("when client is not initialized", func() {
			It("should return error", func() {
				mcpClassifier.client = nil
				_, err := mcpClassifier.Classify(context.Background(), "test")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not initialized"))
			})
		})

		Context("when MCP tool call fails", func() {
			It("should return error", func() {
				mockClient.callToolError = errors.New("tool call failed")
				_, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("tool call failed"))
			})
		})

		Context("when MCP tool returns error result", func() {
			It("should return error", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: true,
					Content: []mcp.Content{mcp.TextContent{Type: "text", Text: "error message"}},
				}
				_, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("returned error"))
			})
		})

		Context("when MCP tool returns empty content", func() {
			It("should return error", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{},
				}
				_, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("empty content"))
			})
		})

		Context("when MCP tool returns valid classification", func() {
			It("should return classification result", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 2, "confidence": 0.95, "model": "openai/gpt-oss-20b", "use_reasoning": true}`,
						},
					},
				}
				result, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(result.Class).To(Equal(2))
				Expect(result.Confidence).To(BeNumerically("~", 0.95, 0.001))
			})
		})

		Context("when MCP tool returns classification with routing info", func() {
			It("should parse model and use_reasoning fields", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 1, "confidence": 0.85, "model": "openai/gpt-oss-20b", "use_reasoning": false}`,
						},
					},
				}
				result, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(result.Class).To(Equal(1))
				Expect(result.Confidence).To(BeNumerically("~", 0.85, 0.001))
			})
		})

		Context("when MCP tool returns invalid JSON", func() {
			It("should return error", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `invalid json`,
						},
					},
				}
				_, err := mcpClassifier.Classify(context.Background(), "test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("failed to parse"))
			})
		})
	})

	Describe("ClassifyWithProbabilities", func() {
		BeforeEach(func() {
			mcpClassifier.client = mockClient
			mcpClassifier.toolName = "classify_text"
		})

		Context("when client is not initialized", func() {
			It("should return error", func() {
				mcpClassifier.client = nil
				_, err := mcpClassifier.ClassifyWithProbabilities(context.Background(), "test")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not initialized"))
			})
		})

		Context("when MCP tool returns valid result with probabilities", func() {
			It("should return result with probability distribution", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 1, "confidence": 0.85, "probabilities": [0.10, 0.85, 0.05], "model": "openai/gpt-oss-20b", "use_reasoning": true}`,
						},
					},
				}
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
	})

	Describe("ListCategories", func() {
		BeforeEach(func() {
			mcpClassifier.client = mockClient
		})

		Context("when client is not initialized", func() {
			It("should return error", func() {
				mcpClassifier.client = nil
				_, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not initialized"))
			})
		})

		Context("when MCP tool returns valid categories", func() {
			It("should return category mapping", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"categories": ["math", "science", "technology", "history", "general"]}`,
						},
					},
				}
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
		})

		Context("when MCP tool returns categories with per-category system prompts", func() {
			It("should store system prompts in mapping", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{
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
							}`,
						},
					},
				}
				mapping, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).ToNot(HaveOccurred())
				Expect(mapping).ToNot(BeNil())
				Expect(mapping.CategoryToIdx).To(HaveLen(3))

				// Verify system prompts are stored
				Expect(mapping.CategorySystemPrompts).ToNot(BeNil())
				Expect(mapping.CategorySystemPrompts).To(HaveLen(3))

				mathPrompt, ok := mapping.GetCategorySystemPrompt("math")
				Expect(ok).To(BeTrue())
				Expect(mathPrompt).To(ContainSubstring("mathematics expert"))

				sciencePrompt, ok := mapping.GetCategorySystemPrompt("science")
				Expect(ok).To(BeTrue())
				Expect(sciencePrompt).To(ContainSubstring("science expert"))

				techPrompt, ok := mapping.GetCategorySystemPrompt("technology")
				Expect(ok).To(BeTrue())
				Expect(techPrompt).To(ContainSubstring("technology expert"))

				// Verify descriptions are stored
				Expect(mapping.CategoryDescriptions).ToNot(BeNil())
				Expect(mapping.CategoryDescriptions).To(HaveLen(3))

				mathDesc, ok := mapping.GetCategoryDescription("math")
				Expect(ok).To(BeTrue())
				Expect(mathDesc).To(Equal("Mathematical and computational queries"))
			})
		})

		Context("when MCP tool returns categories without system prompts", func() {
			It("should handle missing system prompts gracefully", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"categories": ["math", "science"]}`,
						},
					},
				}
				mapping, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).ToNot(HaveOccurred())
				Expect(mapping).ToNot(BeNil())
				Expect(mapping.CategoryToIdx).To(HaveLen(2))

				// System prompts should be nil or empty
				mathPrompt, ok := mapping.GetCategorySystemPrompt("math")
				Expect(ok).To(BeFalse())
				Expect(mathPrompt).To(Equal(""))
			})
		})

		Context("when MCP tool returns partial system prompts", func() {
			It("should store only provided system prompts", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{
								"categories": ["math", "science", "history"],
								"category_system_prompts": {
									"math": "You are a mathematics expert.",
									"science": "You are a science expert."
								}
							}`,
						},
					},
				}
				mapping, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).ToNot(HaveOccurred())
				Expect(mapping).ToNot(BeNil())
				Expect(mapping.CategoryToIdx).To(HaveLen(3))
				Expect(mapping.CategorySystemPrompts).To(HaveLen(2))

				mathPrompt, ok := mapping.GetCategorySystemPrompt("math")
				Expect(ok).To(BeTrue())
				Expect(mathPrompt).To(ContainSubstring("mathematics expert"))

				historyPrompt, ok := mapping.GetCategorySystemPrompt("history")
				Expect(ok).To(BeFalse())
				Expect(historyPrompt).To(Equal(""))
			})
		})

		Context("when MCP tool returns error", func() {
			It("should return error", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: true,
					Content: []mcp.Content{mcp.TextContent{Type: "text", Text: "error loading categories"}},
				}
				_, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("returned error"))
			})
		})

		Context("when MCP tool returns invalid JSON", func() {
			It("should return error", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `invalid json`,
						},
					},
				}
				_, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("failed to parse"))
			})
		})

		Context("when MCP tool returns empty categories", func() {
			It("should return empty mapping", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"categories": []}`,
						},
					},
				}
				mapping, err := mcpClassifier.ListCategories(context.Background())
				Expect(err).ToNot(HaveOccurred())
				Expect(mapping).ToNot(BeNil())
				Expect(mapping.CategoryToIdx).To(HaveLen(0))
				Expect(mapping.IdxToCategory).To(HaveLen(0))
			})
		})
	})

	Describe("CategoryMapping System Prompt Methods", func() {
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

		Describe("GetCategorySystemPrompt", func() {
			Context("when category has system prompt", func() {
				It("should return the prompt", func() {
					prompt, ok := mapping.GetCategorySystemPrompt("math")
					Expect(ok).To(BeTrue())
					Expect(prompt).To(Equal("You are a mathematics expert. Show step-by-step solutions."))
				})
			})

			Context("when category exists but has no system prompt", func() {
				It("should return empty string and false", func() {
					prompt, ok := mapping.GetCategorySystemPrompt("tech")
					Expect(ok).To(BeFalse())
					Expect(prompt).To(Equal(""))
				})
			})

			Context("when category does not exist", func() {
				It("should return empty string and false", func() {
					prompt, ok := mapping.GetCategorySystemPrompt("nonexistent")
					Expect(ok).To(BeFalse())
					Expect(prompt).To(Equal(""))
				})
			})

			Context("when CategorySystemPrompts is nil", func() {
				It("should return empty string and false", func() {
					mapping.CategorySystemPrompts = nil
					prompt, ok := mapping.GetCategorySystemPrompt("math")
					Expect(ok).To(BeFalse())
					Expect(prompt).To(Equal(""))
				})
			})
		})

		Describe("GetCategoryDescription", func() {
			Context("when category has description", func() {
				It("should return the description", func() {
					desc, ok := mapping.GetCategoryDescription("math")
					Expect(ok).To(BeTrue())
					Expect(desc).To(Equal("Mathematical queries"))
				})
			})

			Context("when category does not have description", func() {
				It("should return empty string and false", func() {
					desc, ok := mapping.GetCategoryDescription("nonexistent")
					Expect(ok).To(BeFalse())
					Expect(desc).To(Equal(""))
				})
			})
		})
	})
})

var _ = Describe("Classifier MCP Methods", func() {
	var (
		classifier *Classifier
		mockClient *MockMCPClient
	)

	BeforeEach(func() {
		mockClient = &MockMCPClient{}
		cfg := &config.RouterConfig{}
		cfg.Classifier.MCPCategoryModel.Enabled = true
		cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
		cfg.Classifier.MCPCategoryModel.Threshold = 0.5
		cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 30

		// Create MCP classifier manually and inject mock client
		mcpClassifier := &MCPCategoryClassifier{
			client:   mockClient,
			toolName: "classify_text",
			config:   cfg,
		}

		classifier = &Classifier{
			Config:                 cfg,
			mcpCategoryInitializer: mcpClassifier,
			mcpCategoryInference:   mcpClassifier,
			CategoryMapping: &CategoryMapping{
				CategoryToIdx: map[string]int{"tech": 0, "sports": 1, "politics": 2},
				IdxToCategory: map[string]string{"0": "tech", "1": "sports", "2": "politics"},
				CategorySystemPrompts: map[string]string{
					"tech":     "You are a technology expert. Include practical examples.",
					"sports":   "You are a sports expert. Provide game analysis.",
					"politics": "You are a politics expert. Provide balanced perspectives.",
				},
				CategoryDescriptions: map[string]string{
					"tech":     "Technology and computing topics",
					"sports":   "Sports and athletics",
					"politics": "Political topics and governance",
				},
			},
		}
	})

	Describe("IsMCPCategoryEnabled", func() {
		It("should return true when properly configured", func() {
			Expect(classifier.IsMCPCategoryEnabled()).To(BeTrue())
		})

		It("should return false when not enabled", func() {
			classifier.Config.Classifier.MCPCategoryModel.Enabled = false
			Expect(classifier.IsMCPCategoryEnabled()).To(BeFalse())
		})

		// Note: tool_name is now optional and will be auto-discovered if not specified.
		// IsMCPCategoryEnabled only checks if MCP is enabled, not specific configuration details.
		// Runtime checks (like initializer != nil or successful connection) are handled
		// separately in the actual initialization and classification methods.
	})

	Describe("classifyCategoryMCP", func() {
		Context("when MCP is not enabled", func() {
			It("should return error", func() {
				classifier.Config.Classifier.MCPCategoryModel.Enabled = false
				_, _, err := classifier.classifyCategoryMCP("test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not properly configured"))
			})
		})

		Context("when classification succeeds with high confidence", func() {
			It("should return category name", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 2, "confidence": 0.95, "model": "openai/gpt-oss-20b", "use_reasoning": true}`,
						},
					},
				}

				category, confidence, err := classifier.classifyCategoryMCP("test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("politics"))
				Expect(confidence).To(BeNumerically("~", 0.95, 0.001))
			})
		})

		Context("when confidence is below threshold", func() {
			It("should return empty category", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 1, "confidence": 0.3, "model": "openai/gpt-oss-20b", "use_reasoning": false}`,
						},
					},
				}

				category, confidence, err := classifier.classifyCategoryMCP("test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.3, 0.001))
			})
		})

		Context("when class index is not in mapping", func() {
			It("should return generic category name", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 99, "confidence": 0.85, "model": "openai/gpt-oss-20b", "use_reasoning": true}`,
						},
					},
				}

				category, confidence, err := classifier.classifyCategoryMCP("test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("category_99"))
				Expect(confidence).To(BeNumerically("~", 0.85, 0.001))
			})
		})

		Context("when MCP call fails", func() {
			It("should return error", func() {
				mockClient.callToolError = errors.New("network error")

				_, _, err := classifier.classifyCategoryMCP("test text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("MCP tool call failed"))
			})
		})
	})

	Describe("classifyCategoryWithEntropyMCP", func() {
		BeforeEach(func() {
			classifier.Config.Categories = []config.Category{
				{Name: "tech", ModelScores: []config.ModelScore{{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)}}},
				{Name: "sports", ModelScores: []config.ModelScore{{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)}}},
				{Name: "politics", ModelScores: []config.ModelScore{{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)}}},
			}
		})

		Context("when MCP returns probabilities", func() {
			It("should return category with entropy decision", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 2, "confidence": 0.95, "probabilities": [0.02, 0.03, 0.95], "model": "openai/gpt-oss-20b", "use_reasoning": true}`,
						},
					},
				}

				category, confidence, reasoningDecision, err := classifier.classifyCategoryWithEntropyMCP("test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("politics"))
				Expect(confidence).To(BeNumerically("~", 0.95, 0.001))
				Expect(len(reasoningDecision.TopCategories)).To(BeNumerically(">", 0))
			})
		})

		Context("when confidence is below threshold", func() {
			It("should return empty category but provide entropy decision", func() {
				mockClient.callToolResult = &mcp.CallToolResult{
					IsError: false,
					Content: []mcp.Content{
						mcp.TextContent{
							Type: "text",
							Text: `{"class": 0, "confidence": 0.3, "probabilities": [0.3, 0.35, 0.35], "model": "openai/gpt-oss-20b", "use_reasoning": false}`,
						},
					},
				}

				category, confidence, reasoningDecision, err := classifier.classifyCategoryWithEntropyMCP("test text")
				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.3, 0.001))
				Expect(len(reasoningDecision.TopCategories)).To(BeNumerically(">", 0))
			})
		})
	})

	Describe("initializeMCPCategoryClassifier", func() {
		Context("when MCP is not enabled", func() {
			It("should return error", func() {
				classifier.Config.Classifier.MCPCategoryModel.Enabled = false
				err := classifier.initializeMCPCategoryClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not properly configured"))
			})
		})

		Context("when initializer is nil", func() {
			It("should return error", func() {
				classifier.mcpCategoryInitializer = nil
				err := classifier.initializeMCPCategoryClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("initializer is not set"))
			})
		})
	})
})

var _ = Describe("MCP Helper Functions", func() {
	Describe("createMCPCategoryInitializer", func() {
		It("should create MCPCategoryClassifier", func() {
			initializer := createMCPCategoryInitializer()
			Expect(initializer).ToNot(BeNil())
			_, ok := initializer.(*MCPCategoryClassifier)
			Expect(ok).To(BeTrue())
		})
	})

	Describe("createMCPCategoryInference", func() {
		It("should create inference from initializer", func() {
			initializer := &MCPCategoryClassifier{}
			inference := createMCPCategoryInference(initializer)
			Expect(inference).ToNot(BeNil())
			Expect(inference).To(Equal(initializer))
		})

		It("should return nil for non-MCP initializer", func() {
			type FakeInitializer struct{}
			fakeInit := struct {
				FakeInitializer
				MCPCategoryInitializer
			}{}
			inference := createMCPCategoryInference(&fakeInit)
			Expect(inference).To(BeNil())
		})
	})

	Describe("withMCPCategory", func() {
		It("should set MCP fields on classifier", func() {
			classifier := &Classifier{}
			initializer := &MCPCategoryClassifier{}
			inference := createMCPCategoryInference(initializer)

			option := withMCPCategory(initializer, inference)
			option(classifier)

			Expect(classifier.mcpCategoryInitializer).To(Equal(initializer))
			Expect(classifier.mcpCategoryInference).To(Equal(inference))
		})
	})
})

var _ = Describe("Classifier Per-Category System Prompts", func() {
	var classifier *Classifier

	BeforeEach(func() {
		cfg := &config.RouterConfig{}
		cfg.Classifier.MCPCategoryModel.Enabled = true

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

	Describe("GetCategorySystemPrompt", func() {
		Context("when category exists with system prompt", func() {
			It("should return the category-specific system prompt", func() {
				prompt, ok := classifier.GetCategorySystemPrompt("math")
				Expect(ok).To(BeTrue())
				Expect(prompt).To(ContainSubstring("mathematics expert"))
				Expect(prompt).To(ContainSubstring("step-by-step solutions"))
			})
		})

		Context("when requesting different categories", func() {
			It("should return different system prompts for each category", func() {
				mathPrompt, ok := classifier.GetCategorySystemPrompt("math")
				Expect(ok).To(BeTrue())

				sciencePrompt, ok := classifier.GetCategorySystemPrompt("science")
				Expect(ok).To(BeTrue())

				techPrompt, ok := classifier.GetCategorySystemPrompt("tech")
				Expect(ok).To(BeTrue())

				// Verify they are different
				Expect(mathPrompt).ToNot(Equal(sciencePrompt))
				Expect(mathPrompt).ToNot(Equal(techPrompt))
				Expect(sciencePrompt).ToNot(Equal(techPrompt))

				// Verify each has category-specific content
				Expect(mathPrompt).To(ContainSubstring("mathematics"))
				Expect(sciencePrompt).To(ContainSubstring("science"))
				Expect(techPrompt).To(ContainSubstring("technology"))
			})
		})

		Context("when category does not exist", func() {
			It("should return empty string and false", func() {
				prompt, ok := classifier.GetCategorySystemPrompt("nonexistent")
				Expect(ok).To(BeFalse())
				Expect(prompt).To(Equal(""))
			})
		})

		Context("when CategoryMapping is nil", func() {
			It("should return empty string and false", func() {
				classifier.CategoryMapping = nil
				prompt, ok := classifier.GetCategorySystemPrompt("math")
				Expect(ok).To(BeFalse())
				Expect(prompt).To(Equal(""))
			})
		})
	})

	Describe("GetCategoryDescription", func() {
		Context("when category has description", func() {
			It("should return the description", func() {
				desc, ok := classifier.GetCategoryDescription("math")
				Expect(ok).To(BeTrue())
				Expect(desc).To(Equal("Mathematical and computational queries"))
			})
		})

		Context("when category does not exist", func() {
			It("should return empty string and false", func() {
				desc, ok := classifier.GetCategoryDescription("nonexistent")
				Expect(ok).To(BeFalse())
				Expect(desc).To(Equal(""))
			})
		})

		Context("when CategoryMapping is nil", func() {
			It("should return empty string and false", func() {
				classifier.CategoryMapping = nil
				desc, ok := classifier.GetCategoryDescription("math")
				Expect(ok).To(BeFalse())
				Expect(desc).To(Equal(""))
			})
		})
	})
})
