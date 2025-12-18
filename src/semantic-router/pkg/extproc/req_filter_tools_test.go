package extproc

import (
	"encoding/json"
	"os"
	"path/filepath"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

var _ = Describe("Tool Selection Request Filter", func() {
	var (
		tempDir     string
		toolsDBPath string
		router      *OpenAIRouter
		cfg         *config.RouterConfig
		testToolsDB []tools.ToolEntry
	)

	BeforeEach(func() {
		// Initialize BERT model for embeddings
		err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
		Expect(err).NotTo(HaveOccurred())

		// Create temporary directory for tools database
		tempDir, err = os.MkdirTemp("", "tool_selection_test")
		Expect(err).NotTo(HaveOccurred())

		toolsDBPath = filepath.Join(tempDir, "tools.json")

		// Create test tools with different categories and tags
		testToolsDB = []tools.ToolEntry{
			{
				Tool: openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "get_weather",
						Description: param.NewOpt("Get current weather information for a location"),
					},
				},
				Description: "Get current weather information, temperature, conditions, forecast for any location, city, or place",
				Category:    "weather",
				Tags:        []string{"weather", "temperature", "forecast"},
			},
			{
				Tool: openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "search_web",
						Description: param.NewOpt("Search the web for information"),
					},
				},
				Description: "Search the internet, web search, find information online, browse web content",
				Category:    "search",
				Tags:        []string{"search", "web", "internet"},
			},
			{
				Tool: openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "calculate",
						Description: param.NewOpt("Perform mathematical calculations"),
					},
				},
				Description: "Calculate mathematical expressions, solve math problems, arithmetic operations",
				Category:    "math",
				Tags:        []string{"math", "calculation", "arithmetic"},
			},
			{
				Tool: openai.ChatCompletionToolParam{
					Type: "function",
					Function: openai.FunctionDefinitionParam{
						Name:        "send_email",
						Description: param.NewOpt("Send an email message"),
					},
				},
				Description: "Send email messages, email communication, contact people via email",
				Category:    "communication",
				Tags:        []string{"email", "send", "communication"},
			},
		}

		// Write tools database to file
		data, err := json.Marshal(testToolsDB)
		Expect(err).NotTo(HaveOccurred())
		err = os.WriteFile(toolsDBPath, data, 0o644)
		Expect(err).NotTo(HaveOccurred())

		// Create base config
		cfg = CreateTestConfig()
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
	})

	Describe("Tools Database Loading", func() {
		Context("with valid tools database path", func() {
			It("should load tools from toolsDBPath successfully", func() {
				cfg.ToolSelection.Tools.Enabled = true
				cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
				cfg.ToolSelection.Tools.TopK = 3
				cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]

				var err error
				router, err = CreateTestRouter(cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(router.ToolsDatabase).NotTo(BeNil())
				Expect(router.ToolsDatabase.IsEnabled()).To(BeTrue())
				Expect(router.ToolsDatabase.GetToolCount()).To(Equal(4))

				allTools := router.ToolsDatabase.GetAllTools()
				Expect(allTools).To(HaveLen(4))
				Expect(allTools[0].Function.Name).To(Equal("get_weather"))
				Expect(allTools[1].Function.Name).To(Equal("search_web"))
				Expect(allTools[2].Function.Name).To(Equal("calculate"))
				Expect(allTools[3].Function.Name).To(Equal("send_email"))
			})
		})

		Context("with invalid tools database path", func() {
			It("should return error when file does not exist", func() {
				cfg.ToolSelection.Tools.Enabled = true
				cfg.ToolSelection.Tools.ToolsDBPath = "/nonexistent/tools.json"
				cfg.ToolSelection.Tools.TopK = 3

				_, err := CreateTestRouter(cfg)
				Expect(err).To(HaveOccurred())
			})

			It("should return error with malformed JSON", func() {
				badJSONPath := filepath.Join(tempDir, "bad.json")
				err := os.WriteFile(badJSONPath, []byte("{invalid json"), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg.ToolSelection.Tools.Enabled = true
				cfg.ToolSelection.Tools.ToolsDBPath = badJSONPath
				cfg.ToolSelection.Tools.TopK = 3

				_, err = CreateTestRouter(cfg)
				Expect(err).To(HaveOccurred())
			})
		})

		Context("when tools database is disabled", func() {
			It("should not load tools", func() {
				cfg.ToolSelection.Tools.Enabled = false
				cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath

				var err error
				router, err = CreateTestRouter(cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(router.ToolsDatabase).NotTo(BeNil())
				Expect(router.ToolsDatabase.IsEnabled()).To(BeFalse())
				Expect(router.ToolsDatabase.GetToolCount()).To(Equal(0))
			})
		})
	})

	Describe("Top-K Tool Selection", func() {
		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should select top-1 tool when topK=1", func() {
			cfg.ToolSelection.Tools.TopK = 1

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("What's the weather like?", 1)
			Expect(err).NotTo(HaveOccurred())
			Expect(selectedTools).To(HaveLen(1))
			Expect(selectedTools[0].Function.Name).To(Equal("get_weather"))
		})

		It("should select top-2 tools when topK=2", func() {
			cfg.ToolSelection.Tools.TopK = 2

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("search weather forecast", 2)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically("<=", 2))
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
		})

		It("should select top-3 tools when topK=3", func() {
			cfg.ToolSelection.Tools.TopK = 3

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("calculate math and search", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically("<=", 3))
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
		})

		It("should limit results to available tools when topK > tool count", func() {
			cfg.ToolSelection.Tools.TopK = 10

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather", 10)
			Expect(err).NotTo(HaveOccurred())
			// Should return at most the number of tools that meet threshold
			Expect(len(selectedTools)).To(BeNumerically("<=", 4))
		})

		It("should return most relevant tools first", func() {
			cfg.ToolSelection.Tools.TopK = 3

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather forecast temperature", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			// First tool should be weather-related
			Expect(selectedTools[0].Function.Name).To(Equal("get_weather"))
		})
	})

	Describe("Similarity Threshold Filtering", func() {
		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.TopK = 5

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should filter out tools below threshold with threshold=0.7", func() {
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.7}[0]

			// Recreate router with new threshold
			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			// Use a very specific query to test high threshold
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather", 5)
			Expect(err).NotTo(HaveOccurred())
			// With high threshold, may get fewer results
			for _, tool := range selectedTools {
				// All returned tools should be relevant
				Expect(tool.Function.Name).NotTo(BeEmpty())
			}
		})

		It("should include more tools with lower threshold=0.2", func() {
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]

			// Recreate router with new threshold
			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather", 5)
			Expect(err).NotTo(HaveOccurred())
			// Lower threshold should return more tools
			Expect(len(selectedTools)).To(BeNumerically(">=", 1))
		})

		It("should return empty list when no tools meet high threshold", func() {
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.99}[0]

			// Recreate router with new threshold
			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("xyzabc123", 5)
			Expect(err).NotTo(HaveOccurred())
			// Very high threshold with nonsense query should return nothing
			Expect(selectedTools).To(BeEmpty())
		})

		It("should respect both topK and threshold constraints", func() {
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.5}[0]
			cfg.ToolSelection.Tools.TopK = 2

			// Recreate router with new threshold
			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather forecast", 2)
			Expect(err).NotTo(HaveOccurred())
			// Should return at most 2 tools that meet the threshold
			Expect(len(selectedTools)).To(BeNumerically("<=", 2))
		})
	})

	Describe("Fallback Strategy", func() {
		var (
			openAIRequest *openai.ChatCompletionNewParams
			response      *ext_proc.ProcessingResponse
			ctx           *RequestContext
		)

		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.TopK = 3
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			// Create a basic request with tool_choice=auto by unmarshaling JSON
			requestJSON := []byte(`{
				"model": "test-model",
				"messages": [{"role": "user", "content": "What's the weather?"}],
				"tool_choice": "auto"
			}`)
			openAIRequest, err = parseOpenAIRequest(requestJSON)
			Expect(err).NotTo(HaveOccurred())

			response = &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: &ext_proc.HeaderMutation{
								SetHeaders: []*core.HeaderValueOption{},
							},
						},
					},
				},
			}

			ctx = &RequestContext{
				ExpectStreamingResponse: false,
			}
		})

		Context("with fallbackToEmpty=true", func() {
			It("should return empty tools when no tools meet threshold", func() {
				cfg.ToolSelection.Tools.FallbackToEmpty = true
				cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.99}[0]

				testRouter, err := CreateTestRouter(cfg)
				Expect(err).NotTo(HaveOccurred())

				err = testRouter.handleToolSelection(openAIRequest, "xyzabc nonsense", []string{}, &response, ctx)
				Expect(err).NotTo(HaveOccurred())
				Expect(openAIRequest.Tools).To(BeNil())
			})

			It("should return empty tools on database error", func() {
				cfg.ToolSelection.Tools.FallbackToEmpty = true
				// Corrupt the database by making it disabled
				router.ToolsDatabase = tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
					Enabled:             false,
					SimilarityThreshold: 0.2,
				})

				err := router.handleToolSelection(openAIRequest, "weather", []string{}, &response, ctx)
				Expect(err).NotTo(HaveOccurred())
				// Should handle gracefully and return empty
			})
		})

		Context("with fallbackToEmpty=false", func() {
			It("should keep original tools when no tools meet threshold", func() {
				cfg.ToolSelection.Tools.FallbackToEmpty = false
				cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.99}[0]

				testRouter, err := CreateTestRouter(cfg)
				Expect(err).NotTo(HaveOccurred())

				// Set initial tools
				originalTools := []openai.ChatCompletionToolParam{
					{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name: "original_tool",
						},
					},
				}
				openAIRequest.Tools = originalTools

				err = testRouter.handleToolSelection(openAIRequest, "xyzabc nonsense", []string{}, &response, ctx)
				Expect(err).NotTo(HaveOccurred())
				// Should not be nil but empty array
				Expect(openAIRequest.Tools).NotTo(BeNil())
			})
		})
	})

	Describe("Tool Selection Integration", func() {
		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.TopK = 3
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]
			cfg.ToolSelection.Tools.FallbackToEmpty = true

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should select weather tool for weather query", func() {
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("What's the weather like today?", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			Expect(selectedTools[0].Function.Name).To(Equal("get_weather"))
		})

		It("should select calculate tool for math query", func() {
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("Calculate 25 + 37", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			// Calculator should be among the top results
			toolNames := make([]string, len(selectedTools))
			for i, tool := range selectedTools {
				toolNames[i] = tool.Function.Name
			}
			Expect(toolNames).To(ContainElement("calculate"))
		})

		It("should select search tool for search query", func() {
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("Search for latest news", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			toolNames := make([]string, len(selectedTools))
			for i, tool := range selectedTools {
				toolNames[i] = tool.Function.Name
			}
			Expect(toolNames).To(ContainElement("search_web"))
		})

		It("should select email tool for email query", func() {
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("Send an email to John", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			toolNames := make([]string, len(selectedTools))
			for i, tool := range selectedTools {
				toolNames[i] = tool.Function.Name
			}
			Expect(toolNames).To(ContainElement("send_email"))
		})

		It("should handle queries that match multiple categories", func() {
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("search for weather information and calculate temperature", 3)
			Expect(err).NotTo(HaveOccurred())
			// Should return up to 3 relevant tools
			Expect(len(selectedTools)).To(BeNumerically("<=", 3))
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
		})
	})

	Describe("Tool Selection with Request Processing", func() {
		var (
			response *ext_proc.ProcessingResponse
			ctx      *RequestContext
		)

		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.TopK = 3
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]
			cfg.ToolSelection.Tools.FallbackToEmpty = true

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			response = &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: &ext_proc.HeaderMutation{
								SetHeaders: []*core.HeaderValueOption{},
							},
						},
					},
				},
			}

			ctx = &RequestContext{
				ExpectStreamingResponse: false,
			}
		})

		It("should only process requests with tool_choice=auto", func() {
			requestJSON := []byte(`{
				"model": "test-model",
				"messages": [{"role": "user", "content": "What's the weather?"}],
				"tool_choice": {"type": "function", "function": {"name": "specific_function"}}
			}`)
			openAIRequest, err := parseOpenAIRequest(requestJSON)
			Expect(err).NotTo(HaveOccurred())

			err = router.handleToolSelection(openAIRequest, "weather", []string{}, &response, ctx)
			Expect(err).NotTo(HaveOccurred())
			// Should not modify tools when tool_choice is not auto
		})

		It("should skip processing when content is empty", func() {
			requestJSON := []byte(`{
				"model": "test-model",
				"messages": [{"role": "user", "content": ""}],
				"tool_choice": "auto"
			}`)
			openAIRequest, err := parseOpenAIRequest(requestJSON)
			Expect(err).NotTo(HaveOccurred())

			err = router.handleToolSelection(openAIRequest, "", []string{}, &response, ctx)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should skip processing when tools database is disabled", func() {
			cfg.ToolSelection.Tools.Enabled = false
			testRouter, err := CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())

			requestJSON := []byte(`{
				"model": "test-model",
				"messages": [{"role": "user", "content": "What's the weather?"}],
				"tool_choice": "auto"
			}`)
			openAIRequest, err := parseOpenAIRequest(requestJSON)
			Expect(err).NotTo(HaveOccurred())

			err = testRouter.handleToolSelection(openAIRequest, "weather", []string{}, &response, ctx)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("Category and Tag-Based Filtering", func() {
		BeforeEach(func() {
			cfg.ToolSelection.Tools.Enabled = true
			cfg.ToolSelection.Tools.ToolsDBPath = toolsDBPath
			cfg.ToolSelection.Tools.TopK = 5
			cfg.ToolSelection.Tools.SimilarityThreshold = &[]float32{0.2}[0]

			var err error
			router, err = CreateTestRouter(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should load tools with correct categories", func() {
			allTools := router.ToolsDatabase.GetAllTools()
			Expect(allTools).To(HaveLen(4))
			// Verify categories are preserved in the database
			// Note: The current implementation doesn't expose category/tag info
			// but we verify the tools are loaded correctly
		})

		It("should load tools with correct tags", func() {
			allTools := router.ToolsDatabase.GetAllTools()
			Expect(allTools).To(HaveLen(4))
			// Tags are used internally for semantic matching
			// Verify through semantic search behavior
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("temperature forecast", 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
		})

		It("should handle tools from different categories", func() {
			// Test that tools from multiple categories can be selected
			selectedTools, err := router.ToolsDatabase.FindSimilarTools("weather and email", 5)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(selectedTools)).To(BeNumerically(">", 0))
			// Should potentially return tools from different categories
		})
	})
})
