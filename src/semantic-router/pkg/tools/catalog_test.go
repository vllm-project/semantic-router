package tools_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func catalogDatabase(vectors map[string][]float32) *tools.ToolsDatabase {
	return tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled: true, Provider: &stubToolEmbeddingProvider{embeddings: vectors},
	})
}

func catalogTool(name string) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{Function: openai.FunctionDefinitionParam{Name: name}}
}

func catalogFile(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tools.json")
	require.NoError(t, os.WriteFile(path, data, 0o600))
	return path
}

func catalogNames(catalog []openai.ChatCompletionToolParam) []string {
	names := make([]string, 0, len(catalog))
	for _, tool := range catalog {
		names = append(names, tool.Function.Name)
	}
	return names
}

func TestCatalogPermutationAndEqualScoreTopK(t *testing.T) {
	names := []string{"charlie", " alpha ", "bravo"}
	for turn := 0; turn < len(names); turn++ {
		names = append(names[1:], names[0])
		for _, fromFile := range []bool{true, false} {
			db := catalogDatabase(nil) // All vectors equal; no model downloads.
			entries := make([]tools.ToolEntry, 0, len(names))
			for _, name := range names {
				entries = append(entries, tools.ToolEntry{Tool: catalogTool(name), Description: name})
				if !fromFile {
					require.NoError(t, db.AddTool(catalogTool(name), name, "", nil))
				}
			}
			if fromFile {
				data, err := json.Marshal(entries)
				require.NoError(t, err)
				require.NoError(t, db.LoadToolsFromFile(catalogFile(t, data)))
			}
			require.Equal(t, []string{"alpha", "bravo", "charlie"}, catalogNames(db.GetAllTools()))
			selected, err := db.FindSimilarTools("query", 2)
			require.NoError(t, err)
			require.Equal(t, []string{"alpha", "bravo"}, catalogNames(selected))
		}
	}
}

func TestCatalogDuplicateRejectionIsAtomic(t *testing.T) {
	db := catalogDatabase(nil)
	require.NoError(t, db.AddTool(catalogTool("a"), "a", "", nil))
	require.Error(t, db.AddTool(catalogTool(" a "), "changed", "", nil))
	require.Error(t, db.AddTool(catalogTool(" "), "empty", "", nil))
	for _, names := range [][]string{{"b", " b "}, {"b", "a"}} {
		entries := []tools.ToolEntry{{Tool: catalogTool(names[0])}, {Tool: catalogTool(names[1])}}
		data, err := json.Marshal(entries)
		require.NoError(t, err)
		require.Error(t, db.LoadToolsFromFile(catalogFile(t, data)))
		require.Equal(t, []string{"a"}, catalogNames(db.GetAllTools()))
	}
}

func TestCatalogRejectsMalformedRankingValues(t *testing.T) {
	for _, vector := range [][]float32{{float32(math.NaN())}, {float32(math.Inf(1))}, {float32(math.Inf(-1))}, {}} {
		db := catalogDatabase(map[string][]float32{"bad": vector})
		require.Error(t, db.AddTool(catalogTool("bad"), "bad", "", nil))
		data := []byte(`[{"tool":{"function":{"name":"good"}},"description":"good"},{"tool":{"function":{"name":"bad"}},"description":"bad"}]`)
		require.Error(t, db.LoadToolsFromFile(catalogFile(t, data)))
		require.Zero(t, db.GetToolCount())
		require.NoError(t, db.AddTool(catalogTool("good"), "good", "", nil))
		_, err := db.FindSimilarTools("bad", 1)
		require.Error(t, err)
	}
	db := catalogDatabase(map[string][]float32{"huge": {math.MaxFloat32}})
	require.NoError(t, db.AddTool(catalogTool("huge"), "huge", "", nil))
	_, err := db.FindSimilarTools("huge", 1) // Finite inputs whose dot product overflows.
	require.Error(t, err)
	nan := float32(math.NaN())
	_, err = db.FindSimilarToolsWithScoresMinSimilarity("query", 1, &nan)
	require.Error(t, err)
}

func TestCatalogFilePreservesExactNumbersAndReadIsolation(t *testing.T) {
	const source = `[{"tool":{"type":"function","function":{"name":"number","description":"exact","strict":true,"parameters":{"type":"object","properties":{"n":{"enum":[9007199254740992,9007199254740993],"minimum":1.234567890123456789}}}}},"description":"number","tags":["original"]}]`
	db := catalogDatabase(nil)
	require.NoError(t, db.LoadToolsFromFile(catalogFile(t, []byte(source))))
	all := db.GetAllTools()
	properties := all[0].Function.Parameters["properties"].(map[string]any)
	number := properties["n"].(map[string]any)
	require.Equal(t, []any{json.RawMessage("9007199254740992"), json.RawMessage("9007199254740993")}, number["enum"])
	require.Equal(t, "1.234567890123456789", string(number["minimum"].(json.RawMessage)))
	require.True(t, all[0].Function.Strict.Value)
	require.Equal(t, "exact", all[0].Function.Description.Value)
	before, err := json.Marshal(all[0])
	require.NoError(t, err)
	require.Contains(t, string(before), `"enum":[9007199254740992,9007199254740993]`)
	require.Contains(t, string(before), `"minimum":1.234567890123456789`)
	number["minimum"].(json.RawMessage)[0] = '9'
	number["enum"].([]any)[0] = "mutated"
	results, err := db.FindSimilarToolsWithScores("query", 1)
	require.NoError(t, err)
	results[0].Entry.Tags[0] = "mutated"
	results[0].Entry.Embedding[0] = 42
	results[0].Entry.Tool.Function.Parameters["properties"] = nil
	after, err := json.Marshal(db.GetAllTools()[0])
	require.NoError(t, err)
	require.Equal(t, before, after)
	fresh, err := db.FindSimilarToolsWithScores("query", 1)
	require.NoError(t, err)
	require.Equal(t, []string{"original"}, fresh[0].Entry.Tags)
	require.Equal(t, []float32{0, 0, 1}, fresh[0].Entry.Embedding)
}

func TestCatalogIncrementalIsolationAndDistinctNumericFingerprints(t *testing.T) {
	vector := []float32{0, 0, 1}
	db := catalogDatabase(map[string][]float32{"number": vector})
	tool := catalogTool("number")
	tool.Function.Parameters = openai.FunctionParameters{"enum": []any{json.Number("9007199254740993")}}
	tags := []string{"original"}
	require.NoError(t, db.AddTool(tool, "number", "", tags))
	vector[0], tags[0] = 99, "mutated"
	tool.Function.Parameters["enum"].([]any)[0] = json.Number("9007199254740992")
	stored := db.GetAllTools()[0]
	wire, err := json.Marshal(stored)
	require.NoError(t, err)
	require.Contains(t, string(wire), `"enum":[9007199254740993]`)
	precise, err := json.Marshal(stored.Function.Parameters)
	require.NoError(t, err)
	lower, err := json.Marshal(tool.Function.Parameters)
	require.NoError(t, err)
	a := llmprotocol.Tool{Name: "number", InputSchema: precise}
	b := llmprotocol.Tool{Name: "number", InputSchema: lower}
	require.NotEqual(t, tools.ToolDefinitionFingerprint(a), tools.ToolDefinitionFingerprint(b))
	results, err := db.FindSimilarToolsWithScores("query", 1)
	require.NoError(t, err)
	require.Equal(t, []string{"original"}, results[0].Entry.Tags)
	require.Equal(t, []float32{0, 0, 1}, results[0].Entry.Embedding)
}
