package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func dslFormatCyclePath(path []string, name string) string {
	cycle := make([]string, len(path)+1)
	copy(cycle, path)
	cycle[len(path)] = name
	for i, n := range cycle {
		if n == name {
			return strings.Join(cycle[i:], " -> ")
		}
	}
	return strings.Join(cycle, " -> ")
}

func (v *Validator) dslOutputToSourceScore() map[string]string {
	m := make(map[string]string, len(v.prog.ProjectionMappings))
	for _, mapping := range v.prog.ProjectionMappings {
		for _, output := range mapping.Outputs {
			if output != nil && output.Name != "" {
				m[output.Name] = mapping.Source
			}
		}
	}
	return m
}

func (v *Validator) isProjectionOutputDefined(name string) bool {
	for _, mapping := range v.prog.ProjectionMappings {
		for _, output := range mapping.Outputs {
			if output != nil && output.Name == name {
				return true
			}
		}
	}
	return false
}

func buildDSLScoreCycleAdj(
	scores []*ProjectionScoreDecl,
	outputToSource map[string]string,
) (map[string][]string, map[string]Position) {
	adj := make(map[string][]string, len(scores))
	posMap := make(map[string]Position, len(scores))
	for _, score := range scores {
		posMap[score.Name] = score.Pos
		for _, input := range score.Inputs {
			if !strings.EqualFold(input.SignalType, config.SignalTypeProjection) {
				continue
			}
			dep := input.SignalName
			if strings.EqualFold(input.ValueSource, "confidence") {
				if src, ok := outputToSource[dep]; ok {
					dep = src
				}
			}
			adj[score.Name] = append(adj[score.Name], dep)
		}
	}
	return adj, posMap
}

func (v *Validator) checkProjectionScoreCycles() {
	outputToSource := v.dslOutputToSourceScore()
	adj, posMap := buildDSLScoreCycleAdj(v.prog.ProjectionScores, outputToSource)

	if len(adj) == 0 {
		return
	}

	const (
		unvisited = 0
		visiting  = 1
		visited   = 2
	)
	state := make(map[string]int, len(v.prog.ProjectionScores))
	var path []string

	var visit func(name string)
	visit = func(name string) {
		if state[name] == visited {
			return
		}
		if state[name] == visiting {
			v.addDiag(
				DiagConstraint,
				posMap[name],
				fmt.Sprintf("PROJECTION score dependency cycle detected: %s", dslFormatCyclePath(path, name)),
				nil,
			)
			return
		}
		state[name] = visiting
		path = append(path, name)
		for _, dep := range adj[name] {
			visit(dep)
		}
		path = path[:len(path)-1]
		state[name] = visited
	}

	for _, score := range v.prog.ProjectionScores {
		if state[score.Name] == unvisited {
			visit(score.Name)
		}
	}
}
