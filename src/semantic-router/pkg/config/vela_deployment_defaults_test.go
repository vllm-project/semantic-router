package config

import (
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestVelaDeploymentDownloadsMatchRegistryPinsAndPaths(t *testing.T) {
	pattern := regexp.MustCompile(`\("(llm-semantic-router/Vela-[^"]+)", "(Vela-[^"]+)", "([a-f0-9]{40})"\)`)
	// OpenShift's provider fixtures share one Router deployment and its pinned preloads.
	for _, asset := range []string{"deploy/kserve/deployment.yaml", "deploy/openshift/deployment.yaml"} {
		t.Run(asset, func(t *testing.T) {
			data := string(mustReadRepoFile(t, asset))
			matches := pattern.FindAllStringSubmatch(data, -1)
			want := 4
			if strings.Contains(asset, "kserve") {
				want = 3
			}
			if len(matches) != want {
				t.Fatalf("found %d pinned preloads, want %d", len(matches), want)
			}
			if !strings.Contains(data, "Vela-1.0-Encoder-307M-Guard") || strings.Contains(data, "mmbert32k-jailbreak-detector-merged") {
				t.Fatal("deployment still preloads the old default Guard")
			}
			for _, match := range matches {
				spec := GetModelByPath("models/" + match[2])
				if spec == nil || spec.RepoID != match[1] || spec.Revision != match[3] {
					t.Fatalf("preload identity disagrees with Router registry: %v", match)
				}
			}
			if !strings.Contains(data, "revision=revision") || !strings.Contains(data, `if not has_model or revision != "main":`) {
				t.Fatal("immutable preload can be bypassed by old file presence")
			}
			if !strings.Contains(data, `ignore_patterns=["reproduction/*", "reproducibility/*", "lora/*"]`) {
				t.Fatal("optional training artifacts would be preloaded")
			}
		})
	}
}

func TestKServeDefaultAndExplicitOldEmbeddingSelection(t *testing.T) {
	source := string(mustReadRepoFile(t, "deploy/kserve/deploy.sh"))
	defaultPattern := regexp.MustCompile(`(?m)^EMBEDDING_MODEL="([^"]+)"$`)
	defaultMatch := defaultPattern.FindStringSubmatch(source)
	if len(defaultMatch) != 2 || "models/"+defaultMatch[1] != DefaultGlobalConfig().MmBertModelPath {
		t.Fatal("KServe and Router default embeddings disagree")
	}
	start := strings.Index(source, "resolve_embedding_settings() {")
	if start < 0 {
		t.Fatal("missing embedding selector")
	}
	end := strings.Index(source[start:], "\n}\n")
	if end < 0 {
		t.Fatal("unterminated embedding selector")
	}
	function := source[start : start+end+3]
	for _, name := range []string{defaultMatch[1], "mom-embedding-ultra", "mmbert", "mmbert-embedding", "mmbert-embed-32k-2d-matryoshka"} {
		t.Run(name, func(t *testing.T) {
			// Exercise only the pure selector; no oc, downloads, or deployment actions.
			script := function + "\nresolve_embedding_settings \"$1\"\nprintf '%s\\n%s\\n%s\\n' \"$EMBEDDING_MODEL\" \"$EMBEDDING_MODEL_REPO\" \"$EMBEDDING_MODEL_REVISION\"\n"
			output, err := exec.Command("bash", "-c", script, "selector-test", name).CombinedOutput()
			if err != nil {
				t.Fatalf("selector: %v: %s", err, output)
			}
			fields := strings.Split(strings.TrimSpace(string(output)), "\n")
			if len(fields) != 3 {
				t.Fatalf("invalid selector output: %s", output)
			}
			spec := GetModelByPath("models/" + fields[0])
			if spec == nil || spec.RepoID != fields[1] || filepath.Base(spec.LocalPath) != fields[0] {
				t.Fatalf("selected directory and registry disagree: %v", fields)
			}
			if name == defaultMatch[1] {
				if fields[2] != spec.Revision {
					t.Fatal("Vela revision differs")
				}
			} else if strings.Contains(fields[1], "Vela") || fields[2] != "main" {
				t.Fatal("explicit old embedding was promoted")
			}
		})
	}
}
