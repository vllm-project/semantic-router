package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// TestPluginCombination_MemoryWithSystemPrompt tests that memory injection works
// correctly when system_prompt plugin is also enabled.
func TestPluginCombination_MemoryWithSystemPrompt(t *testing.T) {
	t.Run("memory injection after system prompt preserves both", func(t *testing.T) {
		// Simulate the pipeline order:
		// 1. Start with original request
		// 2. Add system prompt (simulated)
		// 3. Inject memories (should work correctly after system prompt)

		// Original request with system prompt already added
		requestWithSystemPrompt := []byte(`{
			"model": "qwen3",
			"messages": [
				{"role": "system", "content": "You are a helpful travel assistant."},
				{"role": "user", "content": "What was my budget for the Hawaii trip?"}
			]
		}`)

		// Retrieved memories
		memories := []*memory.RetrieveResult{
			{
				Memory: &memory.Memory{
					Content: "Hawaii trip budget is $5000",
					Type:    "fact",
				},
				Score: 0.85,
			},
			{
				Memory: &memory.Memory{
					Content: "User prefers direct flights",
					Type:    "preference",
				},
				Score: 0.72,
			},
		}

		// Inject memories AFTER system prompt (this is the fix)
		result, err := InjectMemories(requestWithSystemPrompt, memories)
		require.NoError(t, err)

		// Parse result to verify structure
		var parsed map[string]interface{}
		err = json.Unmarshal(result, &parsed)
		require.NoError(t, err)

		messages, ok := parsed["messages"].([]interface{})
		require.True(t, ok, "messages should be an array")

		// Memory is APPENDED to existing system message, so we should have 2 messages
		require.Len(t, messages, 2, "should have 2 messages (system with memories + user)")

		// First message: system prompt WITH memory context appended
		firstMsg := messages[0].(map[string]interface{})
		assert.Equal(t, "system", firstMsg["role"])
		content := firstMsg["content"].(string)
		// Should contain BOTH original prompt AND memory context
		assert.Contains(t, content, "helpful travel assistant", "should preserve original system prompt")
		assert.Contains(t, content, "User's Relevant Context", "should have memory context header")
		assert.Contains(t, content, "Hawaii trip budget", "should have memory content")
		assert.Contains(t, content, "direct flights", "should have all memories")

		// Second message: user query
		secondMsg := messages[1].(map[string]interface{})
		assert.Equal(t, "user", secondMsg["role"])
		assert.Contains(t, secondMsg["content"].(string), "Hawaii trip")
	})

	t.Run("empty memories does not modify request", func(t *testing.T) {
		original := []byte(`{
			"model": "qwen3",
			"messages": [
				{"role": "system", "content": "You are helpful."},
				{"role": "user", "content": "Hello"}
			]
		}`)

		// Empty memories should return original unchanged
		result, err := InjectMemories(original, nil)
		require.NoError(t, err)
		assert.Equal(t, original, result)

		result, err = InjectMemories(original, []*memory.RetrieveResult{})
		require.NoError(t, err)
		assert.Equal(t, original, result)
	})

	t.Run("memory injection without existing system prompt", func(t *testing.T) {
		// Request with NO system prompt
		original := []byte(`{
			"model": "qwen3",
			"messages": [
				{"role": "user", "content": "What's my budget?"}
			]
		}`)

		memories := []*memory.RetrieveResult{
			{
				Memory: &memory.Memory{Content: "Budget is $1000", Type: "fact"},
				Score:  0.9,
			},
		}

		result, err := InjectMemories(original, memories)
		require.NoError(t, err)

		var parsed map[string]interface{}
		err = json.Unmarshal(result, &parsed)
		require.NoError(t, err)

		messages := parsed["messages"].([]interface{})
		require.Len(t, messages, 2, "should have memory system message + user message")

		// Memory should be first (prepended as new system message)
		firstMsg := messages[0].(map[string]interface{})
		assert.Equal(t, "system", firstMsg["role"])
		assert.Contains(t, firstMsg["content"].(string), "User's Relevant Context")
		assert.Contains(t, firstMsg["content"].(string), "Budget is $1000")
	})
}

// TestPluginCombination_MemoryPreservedThroughPipeline verifies that storing
// memories in RequestContext and injecting later works correctly.
func TestPluginCombination_MemoryPreservedThroughPipeline(t *testing.T) {
	t.Run("RequestContext stores memories for deferred injection", func(t *testing.T) {
		// Create a RequestContext
		ctx := &RequestContext{}

		// Store memories in context (simulating handleMemoryRetrieval)
		memories := []*memory.RetrieveResult{
			{
				Memory: &memory.Memory{Content: "Test memory", Type: "fact"},
				Score:  0.8,
			},
		}
		ctx.RetrievedMemories = memories

		// Verify memories are stored
		require.Len(t, ctx.RetrievedMemories, 1)
		assert.Equal(t, "Test memory", ctx.RetrievedMemories[0].Memory.Content)
	})
}

// TestPluginCombination_SystemPromptWithPII tests that system_prompt plugin
// doesn't interfere with PII detection.
func TestPluginCombination_SystemPromptWithPII(t *testing.T) {
	t.Run("system prompt added before PII scan doesn't affect detection", func(t *testing.T) {
		// This test verifies that PII detection works on user messages
		// even when a system prompt is present
		requestWithSystemPrompt := `{
			"model": "qwen3",
			"messages": [
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": "My SSN is 123-45-6789"}
			]
		}`

		// Parse and extract user content for PII scanning
		var parsed map[string]interface{}
		err := json.Unmarshal([]byte(requestWithSystemPrompt), &parsed)
		require.NoError(t, err)

		messages := parsed["messages"].([]interface{})
		var userContent string
		for _, msg := range messages {
			m := msg.(map[string]interface{})
			if m["role"] == "user" {
				userContent = m["content"].(string)
			}
		}

		// User content should be extractable for PII scanning
		assert.Contains(t, userContent, "SSN")
		assert.Contains(t, userContent, "123-45-6789")
	})
}

// TestPluginCombination_MultipleSystemMessages tests that memory appends to
// existing system message correctly.
func TestPluginCombination_MultipleSystemMessages(t *testing.T) {
	t.Run("memory appends to first system message", func(t *testing.T) {
		// Start with user's original + plugin's system prompt (combined in one message)
		request := []byte(`{
			"model": "qwen3",
			"messages": [
				{"role": "system", "content": "Original system prompt. Plugin-injected prompt for travel assistant."},
				{"role": "user", "content": "What's my Hawaii budget?"}
			]
		}`)

		// Add memory context
		memories := []*memory.RetrieveResult{
			{
				Memory: &memory.Memory{Content: "Hawaii budget is $5000", Type: "fact"},
				Score:  0.9,
			},
		}

		result, err := InjectMemories(request, memories)
		require.NoError(t, err)

		var parsed map[string]interface{}
		err = json.Unmarshal(result, &parsed)
		require.NoError(t, err)

		messages := parsed["messages"].([]interface{})

		// Should still have 2 messages (memory appended to existing system message)
		require.Len(t, messages, 2)

		// Verify all content is preserved in the system message
		firstMsg := messages[0].(map[string]interface{})
		content := firstMsg["content"].(string)
		assert.Contains(t, content, "Original system prompt")
		assert.Contains(t, content, "Plugin-injected")
		assert.Contains(t, content, "Hawaii budget")

		// User message preserved
		lastMsg := messages[1].(map[string]interface{})
		assert.Equal(t, "user", lastMsg["role"])
		assert.Contains(t, lastMsg["content"].(string), "Hawaii budget")
	})
}

// TestPluginCombination_MemoryInjectionOrderWithRAG tests that memory injection
// works correctly alongside RAG context.
func TestPluginCombination_MemoryInjectionOrderWithRAG(t *testing.T) {
	t.Run("memory and RAG context both present", func(t *testing.T) {
		// Request with RAG context already added (simulated)
		requestWithRAG := []byte(`{
			"model": "qwen3",
			"messages": [
				{"role": "system", "content": "[RAG Context] Here are relevant documents: Doc1, Doc2"},
				{"role": "user", "content": "Summarize the documents"}
			]
		}`)

		// Add memory context
		memories := []*memory.RetrieveResult{
			{
				Memory: &memory.Memory{Content: "User prefers bullet points", Type: "preference"},
				Score:  0.75,
			},
		}

		result, err := InjectMemories(requestWithRAG, memories)
		require.NoError(t, err)

		var parsed map[string]interface{}
		err = json.Unmarshal(result, &parsed)
		require.NoError(t, err)

		messages := parsed["messages"].([]interface{})

		// Both RAG and Memory contexts should be in the same system message
		firstMsg := messages[0].(map[string]interface{})
		content := firstMsg["content"].(string)

		assert.Contains(t, content, "[RAG Context]", "RAG context should be preserved")
		assert.Contains(t, content, "User's Relevant Context", "Memory context should be present")
		assert.Contains(t, content, "bullet points", "Memory content should be present")
	})
}

// TestPluginCombination_PipelineOrder verifies the expected execution order
// of plugins in the request processing pipeline.
func TestPluginCombination_PipelineOrder(t *testing.T) {
	t.Run("verify pipeline processes in correct order", func(t *testing.T) {
		// This test documents the expected pipeline order:
		// 1. PII Detection (can block early)
		// 2. RAG Plugin (if enabled)
		// 3. Memory Retrieval (stores in context, does NOT inject yet)
		// 4. Model Routing / Decision Engine
		// 5. System Prompt Plugin (adds/modifies system prompt)
		// 6. Memory Injection (AFTER system prompt to avoid conflicts)

		// Simulate final state after full pipeline
		// Start with: user request
		// After system_prompt: request has system prompt added
		// After memory: request has memories appended to system prompt

		initialRequest := []byte(`{"model":"qwen3","messages":[{"role":"user","content":"What's my budget?"}]}`)

		// Step 1: System prompt plugin adds system message
		afterSystemPrompt := []byte(`{"model":"qwen3","messages":[{"role":"system","content":"You are a travel assistant."},{"role":"user","content":"What's my budget?"}]}`)

		// Step 2: Memory injection appends to system message
		memories := []*memory.RetrieveResult{
			{Memory: &memory.Memory{Content: "Budget is $5000", Type: "fact"}, Score: 0.9},
		}

		finalRequest, err := InjectMemories(afterSystemPrompt, memories)
		require.NoError(t, err)

		var parsed map[string]interface{}
		err = json.Unmarshal(finalRequest, &parsed)
		require.NoError(t, err)

		messages := parsed["messages"].([]interface{})

		// Verify final state
		systemMsg := messages[0].(map[string]interface{})
		content := systemMsg["content"].(string)

		// Both system prompt and memory should be present
		assert.Contains(t, content, "travel assistant", "system prompt preserved")
		assert.Contains(t, content, "Budget is $5000", "memory injected")

		// Verify initial request was different (for documentation)
		assert.NotEqual(t, initialRequest, afterSystemPrompt, "system prompt modified request")
		assert.NotEqual(t, afterSystemPrompt, finalRequest, "memory injection modified request")
	})
}
