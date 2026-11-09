package extproc

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Run the actual tools plugin with the shipped decision configuration: its
// policy is static, so no decision model is asked.
func TestPrivacyRecipeToolsRespectLocalBoundary(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "privacy", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg}
	for _, d := range cfg.Decisions {
		t.Run(d.Name, func(t *testing.T) {
			request := &llmprotocol.Request{Generation: 1, Tools: []llmprotocol.Tool{makeTool("local_search"), makeTool("local_read"), makeTool("external_upload")}, ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}
			ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: &d}
			router.applyToolsPlugin(request, &ext_proc.ProcessingResponse{}, ctx)
			var names []string
			for _, tool := range ctx.SemanticRequest.Tools {
				names = append(names, tool.Name)
			}
			want := []string{"local_search", "local_read"}
			if d.Name == "cloud_frontier_reasoning" {
				want = append(want, "external_upload")
			}
			if d.Name == "local_security_containment" {
				want = nil
			}
			if !slices.Equal(names, want) {
				t.Fatalf("tools=%v want %v", names, want)
			}
		})
	}
}
