package dsl

import (
	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The escalate block and the executors' budget use the same typed contracts
// as canonical YAML. The DSL owns syntax only; it must not discard budgets,
// stage predicates or accept checks.

// executorFields are an ALGORITHM block's fields that belong to the
// algorithm itself rather than to its type's block.
var executorFields = []string{"budget", "on_error", "minimum_candidates"}

func compileEscalateAlgorithm(c *Compiler, algorithm *config.AlgorithmConfig, fields map[string]Value) {
	block := fieldsToMap(fields)
	for _, name := range executorFields {
		delete(block, name)
	}
	var escalate config.EscalateAlgorithmConfig
	if err := decodeStrict(block, &escalate); err != nil {
		c.addError(Position{}, "invalid escalate fields: %v", err)
		return
	}
	algorithm.Escalate = &escalate
}

func escalateAlgorithmToFields(algorithm *config.AlgorithmConfig, fields map[string]Value) {
	if algorithm.Escalate != nil {
		mergeEncodedFields(*algorithm.Escalate, fields)
	}
}

// compileAlgorithmBudget reads an executor's budget field.
func (c *Compiler) compileAlgorithmBudget(spec *AlgoSpec, algorithm *config.AlgorithmConfig) {
	value, present := spec.Fields["budget"]
	if !present {
		return
	}
	var budget config.AlgorithmBudget
	if err := decodeStrict(valueToInterface(value), &budget); err != nil {
		c.addError(spec.Pos, "invalid algorithm budget: %v", err)
		return
	}
	if err := budget.Validate(); err != nil {
		c.addError(spec.Pos, "invalid algorithm budget: %v", err)
		return
	}
	algorithm.Budget = &budget
}

func algorithmBudgetToFields(budget *config.AlgorithmBudget, fields map[string]Value) {
	if budget == nil {
		return
	}
	encoded := map[string]Value{}
	mergeEncodedFields(*budget, encoded)
	fields["budget"] = ObjectValue{Fields: encoded}
}

func decodeStrict(value interface{}, target interface{}) error {
	encoded, err := yaml.Marshal(value)
	if err != nil {
		return err
	}
	return yaml.UnmarshalStrict(encoded, target)
}

func mergeEncodedFields(value interface{}, fields map[string]Value) {
	encoded, err := yaml.Marshal(value)
	if err != nil {
		return
	}
	var raw map[string]interface{}
	if yaml.Unmarshal(encoded, &raw) != nil {
		return
	}
	for key, field := range interfaceMapObjectValue(raw).Fields {
		fields[key] = field
	}
}
