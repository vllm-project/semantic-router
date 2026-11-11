package config

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
)

// The model runtime's built-in decision models, projected from its registry
// by dashboard/frontend/scripts/generate-decision-runtime-catalog.py: what
// each answers and reads. The load checks what a rule's deployment must
// answer against it; a model it does not list is checked against its card
// when the generation prepares.
//
//go:embed decision_catalog.generated.json
var decisionCatalogJSON []byte

// DecisionCatalogModel is one built-in decision model: its question types,
// the input modalities a decisions request to it may carry, and, for a model
// with span answers, the span heads it has.
type DecisionCatalogModel struct {
	ID            string   `json:"id"`
	Family        string   `json:"family"`
	QuestionTypes []string `json:"question_types"`
	Modalities    []string `json:"modalities"`
	SpanHeads     []string `json:"span_heads,omitempty"`
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
		models[index].SpanHeads = slices.Clone(models[index].SpanHeads)
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
		} else if capabilities, known := deployment.SchemaCapabilities(); known && !capabilities.Has(systemonecodec.CapabilitySpan) {
			// A schema without span questions answers the standard types.
			questionTypes = []string{SystemOneQuestionChoice, SystemOneQuestionNoul, SystemOneQuestionScore}
		}
		if declared := deployment.DeclaredModalities(); len(declared) > 0 {
			modalities = declared
		}
		return questionTypes, modalities
	}
	if model, ok := c.catalogModel(name); ok {
		return model.QuestionTypes, model.Modalities
	}
	return nil, nil
}

// DeploymentQuestionTypes are the question types of the model a deployment
// serves, as far as the configuration tells (deploymentAnswers); nil is
// unknown.
func (c *RouterConfig) DeploymentQuestionTypes(name string) []string {
	questionTypes, _ := c.deploymentAnswers(name)
	return slices.Clone(questionTypes)
}

// catalogModel is the built-in decision model a model_runtime deployment
// serves, when the catalog lists it.
func (c *RouterConfig) catalogModel(name string) (DecisionCatalogModel, bool) {
	deployment, ok := c.ModelDeployments[name]
	if !ok || !deployment.IsModelRuntime() || deployment.Artifact == "" {
		return DecisionCatalogModel{}, false
	}
	repo := deployment.Artifact
	if spec := registeredModel(repo); spec != nil {
		repo = spec.RepoID
	}
	for _, model := range decisionCatalog() {
		if model.ID == repo {
			return model, true
		}
	}
	return DecisionCatalogModel{}, false
}

// deploymentSpanHeads is the span heads of the built-in decision model a
// model_runtime deployment serves; nil is unknown, as for a systemone
// deployment, which declares none.
func (c *RouterConfig) deploymentSpanHeads(name string) []string {
	model, _ := c.catalogModel(name)
	return model.SpanHeads
}

// validateSystemOneAnswerer checks a rule against what its deployment is
// known at load to answer and read, and a span question's head against the
// heads its model has.
func validateSystemOneAnswerer(cfg *RouterConfig, path string, rule SystemOneRule) error {
	name := cfg.SystemOneRuleDeployment(rule)
	questionTypes, modalities := cfg.deploymentAnswers(name)
	if err := CheckSystemOneAnswerer(path, name, rule, questionTypes, modalities); err != nil {
		return err
	}
	if err := checkSchemaCarries(cfg, path, name, rule); err != nil {
		return err
	}
	return checkSpanHead(path, name, rule, cfg.deploymentSpanHeads(name))
}

// CodeSchemaUnsupported is the load error of a rule that needs what the
// schema of its systemone deployment cannot carry.
const CodeSchemaUnsupported = "schema_unsupported"

// checkSchemaCarries refuses a rule that needs what the schema of its
// systemone deployment cannot carry: proof that its whole input was read
// (coverage: full, which full-coverage presets set), or video input.
func checkSchemaCarries(cfg *RouterConfig, path, name string, rule SystemOneRule) error {
	deployment, ok := cfg.ModelDeployments[name]
	if !ok || !deployment.IsSystemOne() {
		return nil
	}
	capabilities, known := deployment.SchemaCapabilities()
	if !known {
		return nil
	}
	schema := deployment.EffectiveSchema()
	if rule.FullCoverage() && !capabilities.Has(systemonecodec.CapabilityFullInput) {
		return CheckError(path, CodeSchemaUnsupported, fmt.Sprintf("reads its whole input (coverage: full) and resolves to deployment %q, whose schema %s has no input_coverage to prove it; set coverage: truncate, or bind it to a deployment that reports coverage (schema vllm-sr/5.0)", name, schema))
	}
	if rule.Input.Video && !capabilities.Has(systemonecodec.CapabilityVideos) {
		return CheckError(path, CodeSchemaUnsupported, fmt.Sprintf("input.video resolves to deployment %q, whose schema %s carries no video", name, schema))
	}
	return nil
}

// CodeSpanHeadUnsupported is the load error of a span question whose head
// its deployment's model does not have.
const CodeSpanHeadUnsupported = "span_head_unsupported"

// checkSpanHead refuses a span question that names a head the model lacks:
// the Vela 3.0 models have only the router head. nil heads are unknown and
// pass; the model refuses the head when it is asked.
func checkSpanHead(path, deployment string, rule SystemOneRule, spanHeads []string) error {
	head := rule.Question.Head
	if rule.Question.Type != SystemOneQuestionSpan || head == "" || spanHeads == nil || slices.Contains(spanHeads, head) {
		return nil
	}
	return CheckError(path, CodeSpanHeadUnsupported, fmt.Sprintf("resolves to deployment %q, whose model has span heads %s and no %s head; drop head, or bind it to a deployment whose model has one (rule deployment)", deployment, strings.Join(spanHeads, ", "), head))
}

// CodeQuestionTypeUnsupported is the load error of a question whose type its
// deployment does not answer; a span question keeps CodeSpanRequired.
const CodeQuestionTypeUnsupported = "question_type_unsupported"

// answersQuestionType reports whether a model that answers questionTypes
// answers a question of questionType: natively, or a set question as one noul
// question per label, which is how a model that answers no set question is
// asked one (modelservice.CompileTask).
func answersQuestionType(questionTypes []string, questionType string) bool {
	return slices.Contains(questionTypes, questionType) ||
		(questionType == SystemOneQuestionSet && slices.Contains(questionTypes, SystemOneQuestionNoul))
}

// CheckSystemOneAnswerer refuses a rule its deployment cannot answer: a
// question of a type the model does not answer (a span question to a model
// without span answers, a score question to vela3-fast), and media input the
// model does not read. A nil list is unknown and passes; the generation
// checks it against the model's card when it prepares.
func CheckSystemOneAnswerer(path, deployment string, rule SystemOneRule, questionTypes, modalities []string) error {
	return checkAnswerer(path, deployment, rule, questionTypes, modalities, "it", "rule deployment")
}

// CheckApplicationAnswerer refuses an application rule its deployment cannot
// answer, as CheckSystemOneAnswerer does a systemone rule; the hint names the
// bindings of the rule's type. scope is the rule's recipe path
// (RouterConfig.ScopePath of ""). A span rule whose preset the model defines
// (presets) asks that question and needs no span answers.
func CheckApplicationAnswerer(path, scope, deployment string, rule ApplicationRule, questionTypes, presets, modalities []string) error {
	bindings := fmt.Sprintf("rule deployment, %smodel_bindings.%s or global.model_catalog.bindings.%s", scope, rule.Type, rule.Type)
	if slices.Contains(presets, spanPresets[rule.Type]) {
		questionTypes = nil
	}
	return checkAnswerer(path, deployment, rule.Rule, questionTypes, modalities, rule.Type, bindings)
}

func checkAnswerer(path, deployment string, rule SystemOneRule, questionTypes, modalities []string, subject, bindings string) error {
	questionType := rule.Question.Type
	if questionTypes != nil && !answersQuestionType(questionTypes, questionType) {
		if questionType == SystemOneQuestionSpan {
			return CheckError(path, CodeSpanRequired, fmt.Sprintf("resolves to deployment %q, which cannot answer span questions; bind %s to a span-capable deployment (%s)", deployment, subject, bindings))
		}
		return CheckError(path, CodeQuestionTypeUnsupported, fmt.Sprintf("resolves to deployment %q, which answers %s questions and cannot answer %s questions; bind %s to a deployment that does (%s)", deployment, strings.Join(questionTypes, ", "), questionType, subject, bindings))
	}
	for _, medium := range rule.Input.Media() {
		if modalities != nil && !slices.Contains(modalities, medium) {
			return CheckError(path, CodeInputModalityUnsupported, fmt.Sprintf("input.%s resolves to deployment %q, which does not read %s input; bind %s to a deployment that does (%s)", medium, deployment, medium, subject, bindings))
		}
	}
	return nil
}
