package config

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

// agentSkillDir is the single source of the public agent skill. An agent
// writes its YAML blocks as they stand, so each one loads with the Router's
// parser, as the checked docs pages do. Its path is in the router-checked-docs
// domain, so editing it runs this test in PR CI; the CLI's unit suite checks
// its command lines (src/vllm-sr/tests/test_agent_skill.py).
var agentSkillDir = repoRel("tools", "agent", "skills", "vllm-sr-agent-operations")

// notRouterDocument reports a block that another tool reads: Helm values,
// which the Helm gate renders, and Kubernetes resources.
func notRouterDocument(fence docFence) bool {
	if fence.helmValues() {
		return true
	}
	var document map[string]interface{}
	if yaml.Unmarshal([]byte(fence.body), &document) != nil {
		return false
	}
	_, apiVersion := document["apiVersion"]
	_, kind := document["kind"]
	return apiVersion && kind
}

func TestAgentSkillYAMLExamplesLoad(t *testing.T) {
	root := referenceConfigRepoRoot(t)
	references, err := filepath.Glob(filepath.Join(root, agentSkillDir, "references", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	pages := append([]string{filepath.Join(root, agentSkillDir, "SKILL.md")}, references...)
	checked := 0
	for _, path := range pages {
		page, err := filepath.Rel(root, path)
		if err != nil {
			t.Fatal(err)
		}
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		checked += checkYAMLExamples(t, page, content, notRouterDocument)
	}
	if checked < 10 {
		t.Fatalf("the agent skill has %d YAML examples to check; its pages moved", checked)
	}
}
