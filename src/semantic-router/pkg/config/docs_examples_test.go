package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Pages and files whose YAML examples the Router must load. Each one is also
// a path of the router-checked-docs domain, so editing it runs this test in
// PR CI. A block titled values.yaml holds Helm chart values; the Helm gate
// renders those (deploy/helm/test_data_volume.py).
var routerCheckedDocPages = []string{
	repoRel("website", "docs", "installation", "protocol-compatibility.md"),
	repoRel("website", "docs", "installation", "storage-overview.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "installation", "storage-overview.md"),
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
// version and an empty routing section when the example omits them.
func docExampleDocument(t *testing.T, body string) []byte {
	t.Helper()
	var document map[string]interface{}
	if err := yaml.Unmarshal([]byte(body), &document); err != nil {
		t.Fatalf("example is not YAML: %v\n%s", err, body)
	}
	if _, ok := document["version"]; !ok {
		document["version"] = "v0.3"
	}
	if _, ok := document["routing"]; !ok {
		document["routing"] = map[string]interface{}{}
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

// Every YAML example on a checked page loads with the Router's parser. An
// example of a removed setting must be followed by a text block holding the
// exact message the Router returns for it.
func TestDocYAMLExamplesLoad(t *testing.T) {
	for _, page := range routerCheckedDocPages {
		t.Run(page, func(t *testing.T) {
			content, err := os.ReadFile(filepath.Join("..", "..", "..", "..", page))
			if err != nil {
				t.Fatal(err)
			}
			fences := docFences(string(content))
			checked := 0
			for index, fence := range fences {
				if fence.lang != "yaml" || fence.helmValues() {
					continue
				}
				checked++
				_, parseErr := ParseYAMLBytes(docExampleDocument(t, fence.body))
				if !exampleSetsRemovedBackendRefField(fence.body) {
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
			if checked == 0 {
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
