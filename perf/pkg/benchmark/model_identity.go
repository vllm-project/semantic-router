package benchmark

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

const ModelIdentityPrefix = "PERF_MODEL_IDENTITY "

type ModelArtifact struct {
	RepoID         string `json:"repo_id"`
	Revision       string `json:"revision"`
	ContentsSHA256 string `json:"contents_sha256"`
}

type ModelIdentity struct {
	Artifacts map[string]ModelArtifact `json:"artifacts"`
	Provider  string                   `json:"provider"`
	Device    string                   `json:"device"`
	Precision string                   `json:"precision"`
	Protocol  string                   `json:"protocol"`
}

func IsModelBenchmark(name string) bool {
	return strings.HasPrefix(name, "BenchmarkClassify") || strings.HasPrefix(name, "BenchmarkCGO") || strings.HasPrefix(name, "BenchmarkCache")
}

func parseModelIdentity(line string) (string, *ModelIdentity, error) {
	rest := strings.TrimPrefix(line, ModelIdentityPrefix)
	name, data, ok := strings.Cut(rest, " ")
	if !ok {
		return "", nil, fmt.Errorf("malformed model identity record")
	}
	var identity ModelIdentity
	if err := json.Unmarshal([]byte(data), &identity); err != nil {
		return "", nil, err
	}
	if identity.Provider == "" || identity.Device == "" || len(identity.Artifacts) == 0 {
		return "", nil, fmt.Errorf("incomplete model identity for %s", name)
	}
	for task, artifact := range identity.Artifacts {
		if artifact.RepoID == "" || artifact.Revision == "" || artifact.ContentsSHA256 == "" {
			return "", nil, fmt.Errorf("incomplete %s artifact identity for %s", task, name)
		}
	}
	return name, &identity, nil
}

// OverlayModelBaseline replaces model-dependent numbers only with a measured
// base-revision run. The same current harness and checkpoint must run on both sides.
func OverlayModelBaseline(baseline, current, models *Baseline) error {
	if models.GitCommit == "" || models.GitCommit == "unknown" {
		return fmt.Errorf("model baseline must identify its measured source commit")
	}
	measured := 0
	for name := range models.Benchmarks {
		if IsModelBenchmark(name) {
			measured++
			if _, ok := current.Benchmarks[name]; !ok {
				return fmt.Errorf("current run did not measure model benchmark %s", name)
			}
		}
	}
	if measured == 0 {
		return fmt.Errorf("model baseline contains no measured model benchmarks")
	}
	for name, metric := range current.Benchmarks {
		if !IsModelBenchmark(name) {
			continue
		}
		prior, ok := models.Benchmarks[name]
		if !ok {
			return fmt.Errorf("model baseline did not measure %s", name)
		}
		if metric.ModelIdentity == nil || prior.ModelIdentity == nil || !reflect.DeepEqual(metric.ModelIdentity, prior.ModelIdentity) {
			return fmt.Errorf("model identity mismatch for %s; measure the same checkpoint and execution settings on the base revision", name)
		}
		baseline.Benchmarks[name] = prior
	}
	return nil
}
