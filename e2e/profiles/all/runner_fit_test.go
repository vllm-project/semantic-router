package all

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
	"k8s.io/apimachinery/pkg/api/resource"
)

// A GitHub-hosted runner has 4 vCPUs, and the Kind worker that holds the
// Router also holds the gateways, the backends and the DaemonSets. A Router
// asking for more than half of the runner leaves its pod Pending until the
// Helm wait times out, which is how model-runtime-vela3 failed its first
// nightly with cpu: "4".
func TestProfileRouterRequestsFitAGitHubRunner(t *testing.T) {
	maxCPU := resource.MustParse("2")
	files, err := filepath.Glob("../*/values.yaml")
	if err != nil || len(files) == 0 {
		t.Fatalf("no profile values found: %v", err)
	}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		var values struct {
			Resources struct {
				Requests map[string]string `yaml:"requests"`
			} `yaml:"resources"`
		}
		if err = yaml.Unmarshal(data, &values); err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		requested, ok := values.Resources.Requests["cpu"]
		if !ok {
			continue
		}
		cpu, parseErr := resource.ParseQuantity(requested)
		if parseErr != nil {
			t.Fatalf("%s: cpu request %q: %v", file, requested, parseErr)
		}
		if cpu.Cmp(maxCPU) > 0 {
			t.Errorf("%s requests %s CPU for the Router, more than %s", file, requested, maxCPU.String())
		}
	}
}
