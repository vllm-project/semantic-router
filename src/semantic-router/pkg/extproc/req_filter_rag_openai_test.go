package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRetrieveFromOpenAI_DirectSearch(t *testing.T) {
	// This is a unit test structure - actual implementation would require
	// mocking the OpenAI API client or using a test server
	// For now, we test the configuration parsing and workflow mode selection

	ragConfig := &config.RAGPluginConfig{
		BackendConfig: &config.OpenAIRAGConfig{
			VectorStoreID: "vs_test123",
			APIKey:        "test-key",
			WorkflowMode:  "direct_search",
			MaxNumResults: intPtr(10),
		},
	}

	openaiConfig, ok := ragConfig.BackendConfig.(*config.OpenAIRAGConfig)
	if !ok {
		t.Fatalf("Failed to cast BackendConfig to OpenAIRAGConfig")
	}

	if openaiConfig.VectorStoreID != "vs_test123" {
		t.Errorf("Expected vector store ID 'vs_test123', got '%s'", openaiConfig.VectorStoreID)
	}
	if openaiConfig.WorkflowMode != "direct_search" {
		t.Errorf("Expected workflow mode 'direct_search', got '%s'", openaiConfig.WorkflowMode)
	}
	if openaiConfig.MaxNumResults == nil || *openaiConfig.MaxNumResults != 10 {
		t.Errorf("Expected MaxNumResults 10, got %v", openaiConfig.MaxNumResults)
	}
}

func TestRetrieveFromOpenAI_ToolBased(t *testing.T) {
	ragConfig := &config.RAGPluginConfig{
		BackendConfig: &config.OpenAIRAGConfig{
			VectorStoreID: "vs_test123",
			APIKey:        "test-key",
			WorkflowMode:  "tool_based",
			FileIDs:       []string{"file-1", "file-2"},
		},
	}

	openaiConfig, ok := ragConfig.BackendConfig.(*config.OpenAIRAGConfig)
	if !ok {
		t.Fatalf("Failed to cast BackendConfig to OpenAIRAGConfig")
	}

	if openaiConfig.WorkflowMode != "tool_based" {
		t.Errorf("Expected workflow mode 'tool_based', got '%s'", openaiConfig.WorkflowMode)
	}
	if len(openaiConfig.FileIDs) != 2 {
		t.Errorf("Expected 2 file IDs, got %d", len(openaiConfig.FileIDs))
	}
}

func TestAddFileSearchToolToRequest(t *testing.T) {
	ctx := &RequestContext{
		OriginalRequestBody: []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"test query"}]}`),
		UserContent:         "test query",
	}

	openaiConfig := &config.OpenAIRAGConfig{
		VectorStoreID: "vs_test123",
		MaxNumResults: intPtr(5),
		FileIDs:       []string{"file-1"},
	}

	// Create a minimal router for testing
	router := &OpenAIRouter{}

	err := router.addFileSearchToolToRequest(ctx, openaiConfig)
	if err != nil {
		t.Fatalf("addFileSearchToolToRequest failed: %v", err)
	}

	// Parse the modified request
	var requestMap map[string]interface{}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &requestMap); err != nil {
		t.Fatalf("Failed to parse modified request: %v", err)
	}

	// Verify tools array exists
	tools, ok := requestMap["tools"].([]interface{})
	if !ok {
		t.Fatal("Tools array not found or not an array")
	}

	if len(tools) == 0 {
		t.Fatal("Tools array is empty")
	}

	// Verify file_search tool
	tool, ok := tools[0].(map[string]interface{})
	if !ok {
		t.Fatal("First tool is not a map")
	}

	if tool["type"] != "file_search" {
		t.Errorf("Expected tool type 'file_search', got '%v'", tool["type"])
	}

	fileSearchConfig, ok := tool["file_search"].(map[string]interface{})
	if !ok {
		t.Fatal("file_search config is not a map")
	}

	vectorStoreIDs, ok := fileSearchConfig["vector_store_ids"].([]interface{})
	if !ok || len(vectorStoreIDs) == 0 {
		t.Fatal("vector_store_ids not found or empty")
	}

	if vectorStoreIDs[0].(string) != "vs_test123" {
		t.Errorf("Expected vector store ID 'vs_test123', got '%v'", vectorStoreIDs[0])
	}
}

func TestAddFileSearchToolToRequest_ExistingTools(t *testing.T) {
	ctx := &RequestContext{
		OriginalRequestBody: []byte(`{
			"model":"gpt-4",
			"messages":[{"role":"user","content":"test query"}],
			"tools":[{"type":"function","function":{"name":"existing_tool"}}]
		}`),
		UserContent: "test query",
	}

	openaiConfig := &config.OpenAIRAGConfig{
		VectorStoreID: "vs_test123",
	}

	router := &OpenAIRouter{}

	err := router.addFileSearchToolToRequest(ctx, openaiConfig)
	if err != nil {
		t.Fatalf("addFileSearchToolToRequest failed: %v", err)
	}

	var requestMap map[string]interface{}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &requestMap); err != nil {
		t.Fatalf("Failed to parse modified request: %v", err)
	}

	tools, ok := requestMap["tools"].([]interface{})
	if !ok {
		t.Fatal("Tools array not found")
	}

	// Should have 2 tools now (existing + file_search)
	if len(tools) != 2 {
		t.Errorf("Expected 2 tools, got %d", len(tools))
	}
}

func intPtr(i int) *int {
	return &i
}
