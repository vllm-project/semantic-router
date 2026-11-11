package config

import (
	"fmt"
	"strings"
)

// DecisionModelBinding selects one declared resource. Model identity belongs
// only to deployments; this reference never guesses an artifact or alias.
type DecisionModelBinding struct {
	Deployment string `yaml:"deployment" json:"deployment" jsonschema:"required,minLength=1"`
}

const (
	DefaultDecisionDeployment = "primary"
	DefaultDecisionModel      = DecisionModelVela3Fast
	DecisionModelVela3Fast    = "vela3-fast"
	DecisionModelVela2_03B    = "Vela-2.0-0.3B"
	DecisionModelVela2_08B    = "Vela-2.0-0.8B"
	DecisionModelVela2_4B     = "Vela-2.0-4B"
	DecisionModelVela2_9B     = "Vela-2.0-9B"
	DecisionModelVela1        = "Vela-1.0"
)

// DecisionModelSpec is a projection of the selected deployment: its name and
// the model it serves. It is not a catalog of allowed model families.
type DecisionModelSpec struct {
	Name  string
	Model string
}

func decisionModelSpec(name string, deployment ModelDeployment) DecisionModelSpec {
	model := deployment.ServedModel(name)
	if registered := GetModelByPath(model); registered != nil {
		model = registered.LocalPath
	} else {
		for _, registered := range DefaultModelRegistry {
			if registered.RepoID == model {
				model = registered.LocalPath
				break
			}
		}
	}
	return DecisionModelSpec{Name: name, Model: model}
}

// validateDecisionModel checks system.decision_model: it names a declared
// model_runtime or systemone deployment. Built-in signals follow it by name
// (model_resolution.go); nothing is copied into module fields.
func validateDecisionModel(resolved *CanonicalGlobal) error {
	system := &resolved.ModelCatalog.System
	name := strings.TrimSpace(system.DecisionModel.Deployment)
	if name == "" || name != system.DecisionModel.Deployment {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment must name a declared deployment")
	}
	deployment, exists := resolved.ModelCatalog.Deployments[name]
	if !exists {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment: unknown deployment %q; %s", name, declaredDeployments(resolved.ModelCatalog.Deployments))
	}
	if !deployment.IsModelRuntime() && !deployment.IsSystemOne() {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment %q must use provider model_runtime or systemone", name)
	}
	return nil
}

// DecisionModelSpec projects the selected deployment's model identity. A
// configuration built in code without a decision model, and a remote
// decision model, report vela3-fast's.
func (c *RouterConfig) DecisionModelSpec() DecisionModelSpec {
	name, deployment, _, err := c.DecisionModelDeployment()
	if err != nil && (c == nil || c.DecisionModel == "" || c.RemoteDecisionModel()) {
		deployment, _ = ImplicitModelRuntimeDeployment(Vela3SignalModel, true)
	}
	return decisionModelSpec(name, deployment)
}

// DecisionModelResource resolves the decision model's deployment of either
// kind: a model_runtime one, as DecisionModelDeployment does, or a systemone
// one.
func (c *RouterConfig) DecisionModelResource() (name string, deployment ModelDeployment, ok bool, err error) {
	if c.RemoteDecisionModel() {
		name = c.DecisionModel
		if name == "" {
			name = DefaultDecisionDeployment
		}
		return name, c.ModelDeployments[name], true, nil
	}
	return c.DecisionModelDeployment()
}

// RemoteDecisionModel reports whether the decision model is a systemone
// deployment.
func (c *RouterConfig) RemoteDecisionModel() bool {
	if c == nil {
		return false
	}
	name := c.DecisionModel
	if name == "" {
		name = DefaultDecisionDeployment
	}
	return c.IsSystemOneDeployment(name)
}

// RequiresGPU reports whether a module model runs on a GPU only.
func RequiresGPU(model string) bool {
	spec := GetModelByPath(strings.TrimSpace(model))
	return spec != nil && spec.RequiresGPU
}

// DecisionModelDeployment resolves only an exact declaration key. Canonical
// defaults declare primary explicitly; no resource is created at request time.
func (c *RouterConfig) DecisionModelDeployment() (name string, deployment ModelDeployment, ok bool, err error) {
	if c == nil {
		return "", ModelDeployment{}, false, fmt.Errorf("decision model requires configuration")
	}
	name = c.DecisionModel
	if name == "" {
		name = DefaultDecisionDeployment
	}
	deployment, ok = c.ModelDeployments[name]
	if !ok {
		return name, ModelDeployment{}, false, fmt.Errorf("unknown decision deployment %q", name)
	}
	if !deployment.IsModelRuntime() {
		return name, deployment, false, fmt.Errorf("decision deployment %q must use provider model_runtime", name)
	}
	return name, deployment.WithDefaults(), true, nil
}

// DecisionQuestionDeployment returns the deployment a decision question asks:
// its own, or the decision model's shared deployment when it names none.
func (c *RouterConfig) SystemOneRuleDeployment(rule SystemOneRule) string {
	return c.deploymentOrDecisionModel(rule.Deployment)
}

// DecisionSelectorDeployment returns the deployment a decision selector asks:
// its own, or the decision model's shared deployment when it names none, so
// the model that answered the request's signals also chooses its model.
func (c *RouterConfig) SystemOneSelectorDeployment(selector SystemOneSelectionConfig) string {
	return c.deploymentOrDecisionModel(selector.Deployment)
}

// EscalateCheckDeployment returns the deployment a Chat escalation asks
// about its answers: the answer check's own, or the decision model's shared
// deployment, which also answers the refusal check.
func (c *RouterConfig) EscalateCheckDeployment(accept *EscalateAccept) string {
	if accept != nil && accept.SystemOne != nil {
		return c.deploymentOrDecisionModel(accept.SystemOne.Deployment)
	}
	return c.deploymentOrDecisionModel("")
}

// EscalateChecksAnswers reports whether an algorithm is a Chat escalation
// whose answers a System One model judges: by answer_check or refusal.
func (a *AlgorithmConfig) EscalateChecksAnswers() bool {
	return a != nil && a.Type == DecisionAlgorithmEscalate && a.Escalate != nil && a.Escalate.Accept != nil &&
		(a.Escalate.Accept.SystemOne != nil || a.Escalate.Accept.Refusal != nil)
}

// deploymentOrDecisionModel returns the named deployment, or the decision
// model's shared deployment for an empty name; empty when there is none.
func (c *RouterConfig) deploymentOrDecisionModel(deployment string) string {
	if named := strings.TrimSpace(deployment); named != "" {
		return named
	}
	if c.RemoteDecisionModel() {
		if c.DecisionModel == "" {
			return DefaultDecisionDeployment
		}
		return c.DecisionModel
	}
	name, _, ok, err := c.DecisionModelDeployment()
	if !ok || err != nil {
		return ""
	}
	return name
}

// ImplicitDeploymentRequiresGPU reports whether an implicit deployment serves
// a built-in model that runs on a GPU only.
func ImplicitDeploymentRequiresGPU(name string, deployment ModelDeployment) bool {
	if !strings.HasPrefix(name, ImplicitDeploymentPrefix) {
		return false
	}
	for i := range DefaultModelRegistry {
		if spec := &DefaultModelRegistry[i]; spec.RequiresGPU && spec.RepoID == deployment.Artifact {
			return true
		}
	}
	return false
}
