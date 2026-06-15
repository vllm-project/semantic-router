package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func hasProjectionDependency(scores []config.ProjectionScore) bool {
	for _, s := range scores {
		for _, inp := range s.Inputs {
			if strings.EqualFold(strings.TrimSpace(inp.Type), config.SignalTypeProjection) {
				return true
			}
		}
	}
	return false
}

func buildScoreAdjacency(scores []config.ProjectionScore, outputToSource map[string]string) map[string][]string {
	adj := make(map[string][]string, len(scores))
	for _, s := range scores {
		for _, inp := range s.Inputs {
			if !strings.EqualFold(strings.TrimSpace(inp.Type), config.SignalTypeProjection) {
				continue
			}
			vs := strings.ToLower(strings.TrimSpace(inp.ValueSource))
			if vs == "confidence" {
				if src, ok := outputToSource[inp.Name]; ok {
					adj[s.Name] = append(adj[s.Name], src)
				}
			} else {
				adj[s.Name] = append(adj[s.Name], inp.Name)
			}
		}
	}
	return adj
}

func topologicalScoreOrder(scores []config.ProjectionScore, mappings []config.ProjectionMapping) []config.ProjectionScore {
	if !hasProjectionDependency(scores) {
		return scores
	}

	byName := make(map[string]config.ProjectionScore, len(scores))
	for _, s := range scores {
		byName[s.Name] = s
	}

	outputToSource := make(map[string]string)
	for _, m := range mappings {
		for _, out := range m.Outputs {
			if out.Name != "" {
				outputToSource[out.Name] = m.Source
			}
		}
	}

	adj := buildScoreAdjacency(scores, outputToSource)
	state := make(map[string]int, len(scores))
	ordered := make([]config.ProjectionScore, 0, len(scores))

	var visit func(name string)
	visit = func(name string) {
		if state[name] != 0 {
			return
		}
		state[name] = 1
		for _, dep := range adj[name] {
			if _, ok := byName[dep]; ok {
				visit(dep)
			}
		}
		if s, ok := byName[name]; ok {
			ordered = append(ordered, s)
		}
	}

	for _, s := range scores {
		visit(s.Name)
	}

	return ordered
}
