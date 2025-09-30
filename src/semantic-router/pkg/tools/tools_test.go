package tools_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestTools(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Tools Suite")
}

var _ = BeforeSuite(func() {
	// Initialize BERT model once for all cache tests (Linux only)
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	Expect(err).NotTo(HaveOccurred())
})

var _ = Describe("ToolsDatabase", func() {
	Describe("NewToolsDatabase", func() {
		It("should create enabled and disabled databases", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
			})
			Expect(db).NotTo(BeNil())
			Expect(db.IsEnabled()).To(BeTrue())

			db2 := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             false,
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
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
			})
			err := db.LoadToolsFromFile(toolFilePath)
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(2))
			toolsList := db.GetAllTools()
			Expect(toolsList).To(HaveLen(2))
			Expect(toolsList[0].Function.Name).To(Equal("weather"))
			Expect(toolsList[1].Function.Name).To(Equal("news"))
		})

		It("should do nothing if disabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             false,
			})
			err := db.LoadToolsFromFile(toolFilePath)
			Expect(err).NotTo(HaveOccurred())
			Expect(db.GetToolCount()).To(Equal(0))
		})

		It("should return error if file does not exist", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
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
			})
			err := db.LoadToolsFromFile(badFile)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to parse tools JSON"))
		})
	})

	Describe("AddTool", func() {
		It("should add tool when enabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
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
			db = tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.7,
				Enabled:             true,
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
			})
			results, err := db2.FindSimilarTools("weather", 2)
			Expect(err).NotTo(HaveOccurred())
			Expect(results).To(BeEmpty())
		})
	})

	Describe("GetAllTools", func() {
		It("should return all tools when enabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
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
			})
			allTools := db.GetAllTools()
			Expect(allTools).To(BeEmpty())
		})
	})

	Describe("GetToolCount", func() {
		It("should return correct count when enabled", func() {
			db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
				SimilarityThreshold: 0.8,
				Enabled:             true,
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
			})
			Expect(db.GetToolCount()).To(Equal(0))
		})
	})
})
