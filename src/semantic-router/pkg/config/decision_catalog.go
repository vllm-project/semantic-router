package config

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
)

// The model runtime's built-in decision models, projected from its registry
// by dashboard/frontend/scripts/generate-decision-runtime-catalog.py: what
// each answers and reads. The load checks what a rule's deployment must
// answer against it; a model it does not list is checked against its card
// when the generation prepares.
//
//go:embed decision_catalog.generated.json
var decisionCatalogJSON []byte

// DecisionCatalogModel is one built-in decision model: its question types
// and the input modalities a decisions request to it may carry.
type DecisionCatalogModel struct {
	ID            string   `json:"id"`
	Family        string   `json:"family"`
	QuestionTypes []string `json:"question_types"`
	Modalities    []string `json:"modalities"`
}

var decisionCatalog = sync.OnceValue(func() []DecisionCatalogModel {
	var catalog struct {
		Models []DecisionCatalogModel `json:"models"`
	}
	if err := json.Unmarshal(decisionCatalogJSON, &catalog); err != nil {
		panic("invalid generated decision catalog: " + err.Error())
	}
	return catalog.Models
})

// DecisionCatalog lists the built-in decision models.
func DecisionCatalog() []DecisionCatalogModel {
	models := slices.Clone(decisionCatalog())
	for index := range models {
		models[index].QuestionTypes = slices.Clone(models[index].QuestionTypes)
		models[index].Modalities = slices.Clone(models[index].Modalities)
	}
	return models
}

// deploymentAnswers is what the model a deployment serves answers and reads,
// as far as the configuration tells: a built-in decision model's catalog
// entry, and what a systemone deployment declares. A nil list is unknown
// until the model's card says.
func (c *RouterConfig) deploymentAnswers(name string) (questionTypes, modalities []string) {
	deployment, ok := c.ModelDeployments[name]
	if !ok {
		return nil, nil
	}
	if deployment.IsSystemOne() {
		if declared := deployment.DeclaredQuestionTypes(); len(declared) > 0 {
			questionTypes = declared
		}
		if declared := deployment.DeclaredModalities(); len(declared) > 0 {
			modalities = declared
		}
		return questionTypes, modalities
	}
	if !deployment.IsModelRuntime() || deployment.Artifact == "" {
		return nil, nil
	}
	repo := deployment.Artifact
	if spec := registeredModel(repo); spec != nil {
		repo = spec.RepoID
	}
	for _, model := range decisionCatalog() {
		if model.ID == repo {
			return model.QuestionTypes, model.Modalities
		}
	}
	return nil, nil
}

// validateSystemOneAnswerer checks a rule against what its deployment is
// known at load to answer and read.
func validateSystemOneAnswerer(cfg *RouterConfig, path string, rule SystemOneRule) error {
	name := cfg.SystemOneRuleDeployment(rule)
	questionTypes, modalities := cfg.deploymentAnswers(name)
	return CheckSystemOneAnswerer(path, name, rule, questionTypes, modalities)
}

// CheckSystemOneAnswerer refuses a rule its deployment cannot answer: a span
// question to a model without span answers, and media input the model does
// not read. A nil list is unknown and passes; the generation checks it
// against the model's card when it prepares.
func CheckSystemOneAnswerer(path, deployment string, rule SystemOneRule, questionTypes, modalities []string) error {
	if rule.Question.Type == SystemOneQuestionSpan && questionTypes != nil && !slices.Contains(questionTypes, SystemOneQuestionSpan) {
		return CheckError(path, CodeSpanRequired, fmt.Sprintf("resolves to deployment %q, which cannot answer span questions; bind it to a span-capable deployment (rule deployment)", deployment))
	}
	for _, medium := range rule.Input.Media() {
		if modalities != nil && !slices.Contains(modalities, medium) {
			return CheckError(path, CodeInputModalityUnsupported, fmt.Sprintf("input.%s resolves to deployment %q, which does not read %s input; bind it to a deployment that does (rule deployment)", medium, deployment, medium))
		}
	}
	return nil
}
