package config

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Pages and files whose YAML examples the Router must load. Each one is also
// a path of the router-checked-docs domain, so editing it runs this test in
// PR CI. A block titled values.yaml holds Helm chart values; the Helm gate
// renders those (deploy/helm/test_data_volume.py).
var routerCheckedDocPages = slices.Concat([]string{
	repoRel("website", "docs", "installation", "protocol-compatibility.md"),
	repoRel("website", "docs", "installation", "storage-overview.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "installation", "storage-overview.md"),
	repoRel("website", "docs", "training", "router-embeddings-huggingface.md"),
}, bothLocales(
	repoRel("system-one", "get-started", "overview.md"),
	repoRel("system-one", "deploy", "deployments.md"),
	repoRel("system-one", "use", "router.md"),
	repoRel("system-one", "use", "custom-questions.md"),
	repoRel("tutorials", "signal", "overview.md"),
	repoRel("tutorials", "signal", "systemone.md"),
	repoRel("tutorials", "signal", "application", "complexity.md"),
	repoRel("tutorials", "signal", "application", "domain.md"),
	repoRel("tutorials", "signal", "application", "fact-check.md"),
	repoRel("tutorials", "signal", "application", "hallucination.md"),
	repoRel("tutorials", "signal", "application", "jailbreak.md"),
	repoRel("tutorials", "signal", "application", "modality.md"),
	repoRel("tutorials", "signal", "application", "pii.md"),
	repoRel("tutorials", "signal", "application", "refusal.md"),
	repoRel("tutorials", "signal", "application", "safety.md"),
	repoRel("tutorials", "signal", "application", "turn-relation.md"),
	repoRel("tutorials", "signal", "application", "user-feedback.md"),
	repoRel("installation", "migrate-to-v0-5.md"),
	repoRel("model-runtime", "deploy", "remote-services.md"),
))

// bothLocales is each English docs page and its zh-Hans translation.
func bothLocales(pages ...string) []string {
	var paths []string
	for _, page := range pages {
		paths = append(paths,
			repoRel("website", "docs", page),
			repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", page))
	}
	return paths
}

var routerCheckedConfigFiles = []string{
	repoRel("tools", "docker", "riscv64", "standalone.yaml"),
}

type docFence struct {
	lang string
	info string
	body string
	line int
}

func (f docFence) helmValues() bool {
	return strings.Contains(f.info, `title="values.yaml"`)
}

// removedForm marks an example of a form the Router no longer reads, as
// "```yaml removed": the migration guide's before-migration examples.
func (f docFence) removedForm() bool {
	return slices.Contains(strings.Fields(f.info), "removed")
}

// docFences returns the fenced blocks of a Markdown page, with a fence's own
// indentation removed from its lines so list-nested examples parse.
func docFences(page string) []docFence {
	var fences []docFence
	lines := strings.Split(page, "\n")
	for index := 0; index < len(lines); index++ {
		trimmed := strings.TrimLeft(lines[index], " ")
		if !strings.HasPrefix(trimmed, "```") {
			continue
		}
		indent := len(lines[index]) - len(trimmed)
		fence := docFence{line: index + 1, info: strings.TrimPrefix(trimmed, "```")}
		if info := strings.Fields(fence.info); len(info) > 0 {
			fence.lang = info[0]
		}
		var body []string
		for index++; index < len(lines); index++ {
			if strings.HasPrefix(strings.TrimLeft(lines[index], " "), "```") {
				break
			}
			line := lines[index]
			if len(line) >= indent && strings.TrimSpace(line[:indent]) == "" {
				line = line[indent:]
			}
			body = append(body, line)
		}
		fence.body = strings.Join(body, "\n")
		fences = append(fences, fence)
	}
	return fences
}

// docExampleDocument completes a page's example into a whole document: the
// version, and an empty recipe when the example declares none.
func docExampleDocument(t *testing.T, body string) []byte {
	t.Helper()
	var document map[string]interface{}
	if err := yaml.Unmarshal([]byte(body), &document); err != nil {
		t.Fatalf("example is not YAML: %v\n%s", err, body)
	}
	if _, ok := document["version"]; !ok {
		document["version"] = "v0.5"
	}
	if _, ok := document["recipes"]; !ok {
		document["recipes"] = []interface{}{map[string]interface{}{"name": "default"}}
	}
	encoded, err := yaml.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func exampleSetsRemovedBackendRefField(body string) bool {
	for _, field := range removedBackendRefFields {
		if strings.Contains(body, field+":") {
			return true
		}
	}
	return false
}

// checkYAMLExamples loads every YAML example of a page with the Router's
// parser, except the blocks skip names, and returns how many it checked. An
// example of a removed setting, or one marked removed, must be followed by a
// text block holding the exact message the Router returns for it.
func checkYAMLExamples(t *testing.T, page string, content []byte, skip func(docFence) bool) int {
	t.Helper()
	fences := docFences(string(content))
	checked := 0
	for index, fence := range fences {
		if fence.lang != "yaml" || skip(fence) {
			continue
		}
		checked++
		_, parseErr := ParseYAMLBytes(docExampleDocument(t, fence.body))
		if !exampleSetsRemovedBackendRefField(fence.body) && !fence.removedForm() {
			if parseErr != nil {
				t.Errorf("%s:%d: example does not load: %v\n%s", page, fence.line, parseErr, fence.body)
			}
			continue
		}
		if index+1 >= len(fences) || fences[index+1].lang != "text" {
			t.Errorf("%s:%d: a removed-setting example must be followed by the message it gets", page, fence.line)
			continue
		}
		want := strings.TrimSpace(fences[index+1].body)
		if parseErr == nil || parseErr.Error() != want {
			t.Errorf("%s:%d: example error = %v, the page shows %q", page, fence.line, parseErr, want)
		}
	}
	return checked
}

// Every YAML example on a checked page loads with the Router's parser.
func TestDocYAMLExamplesLoad(t *testing.T) {
	for _, page := range routerCheckedDocPages {
		t.Run(page, func(t *testing.T) {
			content, err := os.ReadFile(filepath.Join("..", "..", "..", "..", page))
			if err != nil {
				t.Fatal(err)
			}
			if checkYAMLExamples(t, page, content, docFence.helmValues) == 0 {
				t.Fatalf("%s has no YAML examples to check", page)
			}
		})
	}
}

func TestDocConfigFilesLoad(t *testing.T) {
	for _, file := range routerCheckedConfigFiles {
		t.Run(file, func(t *testing.T) {
			content, err := os.ReadFile(filepath.Join("..", "..", "..", "..", file))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ParseYAMLBytes(content); err != nil {
				t.Fatalf("%s does not load: %v", file, err)
			}
		})
	}
}
