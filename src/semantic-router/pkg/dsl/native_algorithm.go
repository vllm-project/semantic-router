package dsl

import (
	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Native algorithm fields use the same typed contracts as canonical YAML.
// The DSL owns syntax only; it must not discard budgets or stage predicates.
type nativeAlgorithmFields struct {
	Budget  *config.AlgorithmBudget     `yaml:"budget"`
	Quality *config.NativeQualityConfig `yaml:"quality"`
	Stages  []config.CascadeStage       `yaml:"stages"`
}

func compileCascadeAlgorithm(c *Compiler, algorithm *config.AlgorithmConfig, fields map[string]Value) {
	encoded, err := yaml.Marshal(fieldsToMap(fields))
	if err != nil {
		c.addError(Position{}, "invalid cascade fields: %v", err)
		return
	}
	var payload nativeAlgorithmFields
	if err = yaml.UnmarshalStrict(encoded, &payload); err != nil {
		c.addError(Position{}, "invalid cascade fields: %v", err)
		return
	}
	algorithm.Budget, algorithm.Quality, algorithm.Stages = payload.Budget, payload.Quality, payload.Stages
	if algorithm.Budget == nil {
		c.addError(Position{}, "cascade requires algorithm.budget")
	} else if err = algorithm.Budget.Validate(); err != nil {
		c.addError(Position{}, "invalid cascade budget: %v", err)
	}
	if algorithm.Quality == nil || len(algorithm.Stages) == 0 {
		c.addError(Position{}, "cascade requires quality and stages")
	}
}

func cascadeAlgorithmToFields(algorithm *config.AlgorithmConfig, fields map[string]Value) {
	payload := nativeAlgorithmFields{Budget: algorithm.Budget, Quality: algorithm.Quality, Stages: algorithm.Stages}
	encoded, err := yaml.Marshal(payload)
	if err != nil {
		return
	}
	var raw map[string]interface{}
	if yaml.Unmarshal(encoded, &raw) != nil {
		return
	}
	for key, value := range interfaceMapObjectValue(raw).Fields {
		fields[key] = value
	}
}
