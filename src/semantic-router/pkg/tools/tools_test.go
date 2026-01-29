package tools_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func TestTools(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Tools Suite")
}

var _ = BeforeSuite(func() {
	// Initialize BERT model once for all cache tests (Linux only)
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	Expect(err).NotTo(HaveOccurred())

	// Initialize embedding models (ModelFactory) for tools tests
	// Try to find Qwen3 or Gemma models in the models directory
	qwen3Path := "../../../../models/mom-embedding-pro"
	gemmaPath := "../../../../models/mom-embedding-flash"

	// Check if at least one model exists
	qwen3Exists := false
	gemmaExists := false
	if _, statErr := os.Stat(qwen3Path); statErr == nil {
		qwen3Exists = true
		GinkgoWriter.Printf("Found Qwen3 embedding model at %s\n", qwen3Path)
	}
	if _, statErr := os.Stat(gemmaPath); statErr == nil {
		gemmaExists = true
		GinkgoWriter.Printf("Found Gemma embedding model at %s\n", gemmaPath)
	}

	// Initialize ModelFactory if at least one model is available
	if qwen3Exists || gemmaExists {
		qwen3ToUse := ""
		gemmaToUse := ""
		if qwen3Exists {
			qwen3ToUse = qwen3Path
		}
		if gemmaExists {
			gemmaToUse = gemmaPath
		}

		GinkgoWriter.Printf("Initializing ModelFactory with Qwen3=%s, Gemma=%s\n", qwen3ToUse, gemmaToUse)
		err = candle_binding.InitEmbeddingModels(qwen3ToUse, gemmaToUse, "", true)
		if err != nil {
			// Log warning but don't fail - tests will skip if ModelFactory is not initialized
			GinkgoWriter.Printf("Warning: Failed to initialize embedding models: %v\n", err)
			GinkgoWriter.Printf("Tools tests requiring ModelFactory will be skipped\n")
		} else {
			GinkgoWriter.Printf("✓ ModelFactory initialized successfully\n")
		}
	} else {
		GinkgoWriter.Printf("Warning: No embedding models found at %s or %s\n", qwen3Path, gemmaPath)
		GinkgoWriter.Printf("Tools tests requiring ModelFactory will be skipped\n")
	}
})

// Helper function to check if ModelFactory is initialized
func isModelFactoryInitialized() bool {
	// Try to get embedding models info to check if ModelFactory is initialized
	_, err := candle_binding.GetEmbeddingModelsInfo()
	return err == nil
}

var _ = Describe("ToolsDatabase", func() {
	Describe("NewToolsDatabase", func() {
		It("should create enabled and disabled databases", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			Expect(db).NotTo(BeNil())
			Expect(db.IsEnabled()).To(BeTrue())

			db2 := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			Expect(db2).NotTo(BeNil())
			Expect(db2.IsEnabled()).To(BeFalse())
		})
	})

	Describe("LoadToolsFromFile", func() {
		var (
			tempDir      string
			toolFilePath string
		)

		BeforeEach(func() {
			var err error
			tempDir, err = os.MkdirTemp("", "tools_test")
			Expect(err).NotTo(HaveOccurred())

			toolFilePath = filepath.Join(tempDir, "tools.json")
			toolsData := []tools.ToolEntry{
				{
					Tool: openai.ChatCompletionToolParam{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name:        "weather",
							Description: param.NewOpt("Get weather info"),
						},
					},
					Description: "Get weather info",
					Tags:        []string{"weather", "info"},
					Category:    "utility",
				},
				{
					Tool: openai.ChatCompletionToolParam{
						Type: "function",
						Function: openai.FunctionDefinitionParam{
							Name:        "news",
							Description: param.NewOpt("Get latest news"),
						},
					},
					Description: "Get latest news",
					Tags:        []string{"news"},
					Category:    "information",
				},
			}
			data, err := json.Marshal(toolsData)
			Expect(err).NotTo(HaveOccurred())
			err = os.WriteFile(toolFilePath, data, 0o644)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			os.RemoveAll(tempDir)
		})

		It("should load tools from file when enabled", func() {
			if !isModelFactoryInitialized() {
				Skip("Skipping test: ModelFactory not initialized (embedding models not available)")
			}

			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			err := db.LoadToolsFromFile(toolFilePath)
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(2))
			toolsList := db.GetAllTools()
			Expect(toolsList).To(HaveLen(2))
			// Check that both tools are loaded (order may vary due to concurrent processing)
			toolNames := []string{toolsList[0].Function.Name, toolsList[1].Function.Name}
			Expect(toolNames).To(ContainElement("weather"))
			Expect(toolNames).To(ContainElement("news"))
		})

		It("should do nothing if disabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			err := db.LoadToolsFromFile(toolFilePath)
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(0))
		})

		It("should return error if file does not exist", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			err := db.LoadToolsFromFile("/nonexistent/tools.json")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to read tools file"))
		})

		It("should return error if file is invalid JSON", func() {
			badFile := filepath.Join(tempDir, "bad.json")
			Expect(os.WriteFile(badFile, []byte("{invalid json"), 0o644)).To(Succeed())
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			err := db.LoadToolsFromFile(badFile)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to parse tools JSON"))
		})
	})

	Describe("AddTool", func() {
		It("should add tool when enabled", func() {
			if !isModelFactoryInitialized() {
				Skip("Skipping test: ModelFactory not initialized (embedding models not available)")
			}

			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			tool := openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "calculator",
					Description: param.NewOpt("Simple calculator"),
				},
			}
			err := db.AddTool(tool, "Simple calculator", "utility", []string{"math"})
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(1))
			allTools := db.GetAllTools()
			Expect(allTools[0].Function.Name).To(Equal("calculator"))
		})

		It("should do nothing if disabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			tool := openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "calculator",
					Description: param.NewOpt("Simple calculator"),
				},
			}
			err := db.AddTool(tool, "Simple calculator", "utility", []string{"math"})
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(0))
		})
	})

	Describe("FindSimilarTools", func() {
		var db *tools.ToolsDatabase

		BeforeEach(func() {
			if !isModelFactoryInitialized() {
				Skip("Skipping test: ModelFactory not initialized (embedding models not available)")
			}

			db = tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.2, // Lower threshold for more lenient matching
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "weather",
					Description: param.NewOpt("Get weather info"),
				},
			}, "Get weather info", "utility", []string{"weather", "info"})
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "news",
					Description: param.NewOpt("Get latest news"),
				},
			}, "Get latest news", "information", []string{"news"})
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "calculator",
					Description: param.NewOpt("Simple calculator"),
				},
			}, "Simple calculator", "utility", []string{"math"})
		})

		It("should find similar tools for a relevant query", func() {
			results, err := db.FindSimilarTools("weather", 2)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).NotTo(BeEmpty())
			Expect(results[0].Function.Name).To(Equal("weather"))
		})

		It("should return at most topK results", func() {
			results, err := db.FindSimilarTools("info", 1)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(results)).To(BeNumerically("<=", 1))
		})

		It("should return empty if disabled", func() {
			db2 := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			results, err := db2.FindSimilarTools("weather", 2)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})
	})

	Describe("GetAllTools", func() {
		It("should return all tools when enabled", func() {
			if !isModelFactoryInitialized() {
				Skip("Skipping test: ModelFactory not initialized (embedding models not available)")
			}

			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "weather",
					Description: param.NewOpt("Get weather info"),
				},
			}, "Get weather info", "utility", []string{"weather"})
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "news",
					Description: param.NewOpt("Get latest news"),
				},
			}, "Get latest news", "information", []string{"news"})
			allTools := db.GetAllTools()
			Expect(allTools).To(HaveLen(2))
		})

		It("should return empty if disabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			allTools := db.GetAllTools()
			Expect(allTools).To(BeEmpty())
		})
	})

	Describe("GetToolCount", func() {
		It("should return correct count when enabled", func() {
			if !isModelFactoryInitialized() {
				Skip("Skipping test: ModelFactory not initialized (embedding models not available)")
			}

			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			Expect(db.GetToolCount()).To(Equal(0))
			_ = db.AddTool(openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        "weather",
					Description: param.NewOpt("Get weather info"),
				},
			}, "Get weather info", "utility", []string{"weather"})
			Expect(db.GetToolCount()).To(Equal(1))
		})

		It("should return zero if disabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             false,
				ModelType:           "qwen3",
				TargetDimension:     768,
			})
			Expect(db.GetToolCount()).To(Equal(0))
		})
	})
})
