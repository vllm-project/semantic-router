package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// routerConfigOf writes the ConfigMap document of a resource and loads it as
// the Router does; the document comes back as a YAML tree.
func routerConfigOf(t *testing.T, spec vllmv1alpha1.ConfigSpec) (map[string]interface{}, *routerconfig.RouterConfig) {
	t.Helper()
	r := &SemanticRouterReconciler{}
	data, err := r.generateConfigYAML(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: spec}})
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]interface{}
	if unmarshalErr := yaml.Unmarshal([]byte(data), &document); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	cfg, err := routerconfig.ParseYAMLBytes([]byte(data))
	if err != nil {
		t.Fatalf("the Router refuses the operator's document: %v", err)
	}
	return document, cfg
}

func modelCatalogOf(t *testing.T, document map[string]interface{}) map[string]interface{} {
	t.Helper()
	global, _ := document["global"].(map[string]interface{})
	catalog, ok := global["model_catalog"].(map[string]interface{})
	if !ok {
		t.Fatalf("the document has no global.model_catalog: %v", document)
	}
	return catalog
}

// writtenKeys lists the paths under tree that end in one of keys.
func writtenKeys(tree interface{}, path string, keys ...string) []string {
	var found []string
	switch node := tree.(type) {
	case map[string]interface{}:
		for key, value := range node {
			child := path + "." + key
			if slices.Contains(keys, key) {
				found = append(found, child)
			}
			found = append(found, writtenKeys(value, child, keys...)...)
		}
	case []interface{}:
		for i, value := range node {
			found = append(found, writtenKeys(value, fmt.Sprintf("%s[%d]", path, i), keys...)...)
		}
	}
	sort.Strings(found)
	return found
}

// builtInSignals are the module consumers the decision model answers when no
// binding names a deployment for them.
var builtInSignals = []string{
	"prompt_guard", "domain_classifier", "pii_classifier", "fact_check_classifier",
	"feedback_detector", "hallucination_detector", "modality_detector",
}

// answeredSignals are the built-in signals that ask a model: a modality
// detector whose method runs no classifier asks none.
func answeredSignals(cfg *routerconfig.RouterConfig) []string {
	signals := make([]string, 0, len(builtInSignals))
	for _, consumer := range builtInSignals {
		if _, _, classifies := cfg.ModalityClassifierModel(); consumer == "modality_detector" && !classifies {
			continue
		}
		signals = append(signals, consumer)
	}
	return signals
}

func moduleThresholds(cfg *routerconfig.RouterConfig) routerconfig.ModuleThresholds {
	return routerconfig.ModuleThresholds{
		PromptGuard: cfg.PromptGuard.Threshold,
		Domain:      cfg.CategoryModel.Threshold,
		PII:         cfg.PIIModel.Threshold,
		FactCheck:   cfg.HallucinationMitigation.FactCheckModel.Threshold,
		Feedback:    cfg.FeedbackDetector.Threshold,
	}
}

var onlyTheDecisionModel = map[string]interface{}{"decision_model": map[string]interface{}{"deployment": "primary"}}

func TestTheDecisionModelReachesEveryModuleWithItsThresholds(t *testing.T) {
	document, cfg := routerConfigOf(t, decisionDeploymentSpec(t, "primary", "vllm-sr/Vela-2.0-9B"))
	if system := modelCatalogOf(t, document)["system"]; !reflect.DeepEqual(system, onlyTheDecisionModel) {
		t.Fatalf("the ConfigMap must name the decision model and pin no module, got %v", system)
	}
	for _, consumer := range answeredSignals(cfg) {
		resolved, ok, err := cfg.ResolveTask(consumer)
		if err != nil || !ok || resolved.Name != "primary" || resolved.Source != routerconfig.ResolvedByDecisionModel {
			t.Fatalf("%s must follow the decision model, got %+v (%v)", consumer, resolved, err)
		}
	}
	if want := routerconfig.ModuleThresholdsOf(routerconfig.Vela2Model9B); moduleThresholds(cfg) != want {
		t.Fatalf("modules the resource leaves at defaults take the 9B's thresholds %+v, got %+v", want, moduleThresholds(cfg))
	}
}

func TestTheOperatorWritesOnlyTheThresholdsTheResourceSets(t *testing.T) {
	document, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		PromptGuard: &vllmv1alpha1.PromptGuardConfig{Enabled: true, Threshold: "0.6", UseCPU: true},
		Classifier:  &vllmv1alpha1.ClassifierConfig{PIIModel: &vllmv1alpha1.PIIModelConfig{Threshold: "0.3", UseCPU: true}},
	})
	catalog := modelCatalogOf(t, document)
	if !reflect.DeepEqual(catalog["system"], onlyTheDecisionModel) {
		t.Fatalf("the ConfigMap writes no system line but the decision model, got %v", catalog["system"])
	}
	written := writtenKeys(catalog["modules"], "modules", "threshold", "model_id", "model_ref", "model_path")
	if want := []string{"modules.classifier.pii.threshold", "modules.prompt_guard.threshold"}; !reflect.DeepEqual(written, want) {
		t.Fatalf("the ConfigMap writes %v, want only the resource's %v", written, want)
	}
	vela3 := routerconfig.ModuleThresholdsOf(routerconfig.Vela3SignalModel)
	if cfg.PromptGuard.Threshold != 0.6 || cfg.PIIModel.Threshold != 0.3 || cfg.CategoryModel.Threshold != vela3.Domain {
		t.Fatalf("the resource's thresholds stay and the others follow vela3-fast, got %+v", moduleThresholds(cfg))
	}
}

func TestAGuardTheResourceSetsKeepsItsModelAndThreshold(t *testing.T) {
	_, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "primary"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"primary":{"provider":"model_runtime","artifact":"vllm-sr/Vela-2.0-0.8B"}}`),
		PromptGuard:      &vllmv1alpha1.PromptGuardConfig{Enabled: true, ModelID: "models/Vela-1.0-Encoder-307M-Guard", Threshold: "0.6", UseCPU: true},
	})
	guard, ok, err := cfg.ResolveTask("prompt_guard")
	if err != nil || !ok || guard.Deployment.Artifact != routerconfig.GetModelByPath("models/Vela-1.0-Encoder-307M-Guard").RepoID || cfg.PromptGuard.Threshold != 0.6 {
		t.Fatalf("an explicit guard stays, got %+v at %v (%v)", guard, cfg.PromptGuard.Threshold, err)
	}
	if domain, _, _ := cfg.ResolveTask("domain_classifier"); domain.Name != "primary" {
		t.Fatalf("the other modules follow the decision model, got %+v", domain)
	}
	_, cfg = routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "primary"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"primary":{"provider":"model_runtime","artifact":"vllm-sr/Vela-2.0-0.8B"}}`),
		PromptGuard:      &vllmv1alpha1.PromptGuardConfig{Enabled: true, UseCPU: true},
	})
	guard, _, _ = cfg.ResolveTask("prompt_guard")
	if guard.Name != "primary" || cfg.PromptGuard.Threshold != routerconfig.ModuleThresholdsOf(routerconfig.Vela2Model08B).PromptGuard {
		t.Fatalf("a guard without model or threshold follows the decision model, got %+v at %v", guard, cfg.PromptGuard.Threshold)
	}
}

const systemOneDeployment = `{
	"provider": "systemone",
	"endpoint": "https://systemone.example.com/v1/systemone",
	"model": "vllm-sr/d3-ultra",
	"auth": {"header": "X-Api-Key", "prefix": "", "api_key_env": "SYSTEMONE_KEY"},
	"headers": {"X-Org": "acme"},
	"timeout_ms": 1500,
	"max_retries": 1,
	"limits": {"max_request_bytes": 262144, "max_response_bytes": 4194304},
	"capabilities": {"question_types": ["choice", "noul", "score", "set", "span"], "presets": ["pii", "halu"]}
}`

func TestASystemOneDecisionModelAnswersEveryBuiltInSignal(t *testing.T) {
	t.Setenv("SYSTEMONE_KEY", "operator-test-key")
	document, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "d3-remote"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"d3-remote":`+systemOneDeployment+`}`),
	})
	catalog := modelCatalogOf(t, document)
	deployments, _ := catalog["deployments"].(map[string]interface{})
	if got, want := jsonTree(t, deployments["d3-remote"]), jsonTree(t, json.RawMessage(systemOneDeployment)); !reflect.DeepEqual(got, want) {
		t.Fatalf("the ConfigMap changed the systemone deployment:\n got %v\nwant %v", got, want)
	}
	if want := map[string]interface{}{"decision_model": map[string]interface{}{"deployment": "d3-remote"}}; !reflect.DeepEqual(catalog["system"], want) {
		t.Fatalf("the ConfigMap must name only the decision model, got %v", catalog["system"])
	}
	if !cfg.RemoteDecisionModel() {
		t.Fatalf("the Router does not run the systemone decision model: %q", cfg.DecisionModel)
	}
	for _, consumer := range answeredSignals(cfg) {
		resolved, ok, err := cfg.ResolveTask(consumer)
		if err != nil || !ok || resolved.Name != "d3-remote" || !resolved.Deployment.IsSystemOne() || resolved.Source != routerconfig.ResolvedByDecisionModel {
			t.Fatalf("%s must follow the systemone decision model, got %+v (%v)", consumer, resolved, err)
		}
	}
	remote, calibrated := routerconfig.ThresholdsOfDeployment("d3-remote", cfg.ModelDeployments["d3-remote"])
	if calibrated || moduleThresholds(cfg) != remote {
		t.Fatalf("modules on the remote model take its thresholds %+v, not vela3-fast's; got %+v", remote, moduleThresholds(cfg))
	}
	if key, err := cfg.ModelDeployments["d3-remote"].Auth.Key(); err != nil || key != "operator-test-key" {
		t.Fatalf("auth.api_key_env does not read the Pod environment: %v", err)
	}
	for _, warning := range routerconfig.Warnings(cfg) {
		if warning.Code == "deprecated_model_field" {
			t.Errorf("the operator writes a deprecated model field: %+v", warning)
		}
	}
}

func TestASystemOneKeyReachesTheRouterFromASecret(t *testing.T) {
	key := corev1.EnvVar{Name: "SYSTEMONE_KEY", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
		LocalObjectReference: corev1.LocalObjectReference{Name: "systemone-credentials"}, Key: "api-key",
	}}}
	sr := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{
		Env: []corev1.EnvVar{key},
		Config: vllmv1alpha1.ConfigSpec{
			DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "d3-remote"},
			ModelDeployments: rawCanonicalRoutingJSON(t, `{"d3-remote":`+systemOneDeployment+`}`),
		},
	}}
	r := &SemanticRouterReconciler{}
	container := r.generateContainers(sr, GatewayModeStandalone)[0]
	if !slices.ContainsFunc(container.Env, func(env corev1.EnvVar) bool { return reflect.DeepEqual(env, key) }) {
		t.Fatalf("the Router container does not take the key from the Secret: %+v", container.Env)
	}
	data, err := r.generateConfigYAML(context.Background(), sr)
	if err != nil || !strings.Contains(data, "api_key_env: SYSTEMONE_KEY") {
		t.Fatalf("the ConfigMap must name the variable that holds the key: %v\n%s", err, data)
	}
}

func jsonTree(t *testing.T, value interface{}) interface{} {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var tree interface{}
	if err := json.Unmarshal(data, &tree); err != nil {
		t.Fatal(err)
	}
	return tree
}

func TestModelBindingsReachTheGlobalCatalog(t *testing.T) {
	t.Setenv("SYSTEMONE_KEY", "operator-test-key")
	document, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel: &vllmv1alpha1.DecisionModelBinding{Deployment: "d3-remote"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"d3-remote":`+systemOneDeployment+`,
			"local-vela":{"provider":"model_runtime","artifact":"vllm-sr/vela3-fast","device":"cpu"}}`),
		ModelBindings: rawCanonicalRoutingJSON(t, `{"pii":"local-vela","jailbreak":{"deployment":"local-vela"}}`),
	})
	if bindings, want := modelCatalogOf(t, document)["bindings"], map[string]interface{}{"pii": "local-vela", "jailbreak": "local-vela"}; !reflect.DeepEqual(bindings, want) {
		t.Fatalf("global.model_catalog.bindings = %v, want %v", bindings, want)
	}
	for consumer, want := range map[string]string{"pii_classifier": "local-vela", "prompt_guard": "local-vela", "domain_classifier": "d3-remote"} {
		if resolved, _, err := cfg.ResolveTask(consumer); err != nil || resolved.Name != want {
			t.Fatalf("%s resolves to %q (%v), want %q", consumer, resolved.Name, err, want)
		}
	}
	vela3 := routerconfig.ModuleThresholdsOf(routerconfig.Vela3SignalModel)
	if cfg.PIIModel.Threshold != vela3.PII || cfg.PromptGuard.Threshold != vela3.PromptGuard {
		t.Fatalf("a bound module takes the thresholds of the deployment that answers it, got %+v", moduleThresholds(cfg))
	}
}

func decisionDeploymentSpec(t *testing.T, name, artifact string) vllmv1alpha1.ConfigSpec {
	t.Helper()
	return vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: name},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"`+name+`":{"provider":"model_runtime","artifact":"`+artifact+`"}}`),
	}
}

func TestAnUnknownDecisionDeploymentFailsReconciliation(t *testing.T) {
	r := &SemanticRouterReconciler{}
	_, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: vllmv1alpha1.ConfigSpec{DecisionModel: &vllmv1alpha1.DecisionModelBinding{Deployment: "missing"}}}})
	if err == nil || !strings.Contains(err.Error(), "config.decision_model.deployment") {
		t.Fatalf("want exact deployment error, got %v", err)
	}
}

func TestADecisionModelOnAnotherProviderFailsReconciliation(t *testing.T) {
	for _, provider := range []string{"classify", "candle"} {
		r := &SemanticRouterReconciler{}
		_, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: vllmv1alpha1.ConfigSpec{
			DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "judge"},
			ModelDeployments: rawCanonicalRoutingJSON(t, `{"judge":{"provider":"`+provider+`","endpoint":"https://judge.example.com/v1","model":"judge"}}`),
		}}})
		if err == nil || !strings.Contains(err.Error(), "config.decision_model.deployment") ||
			!strings.Contains(err.Error(), routerconfig.ModelRuntimeProvider) || !strings.Contains(err.Error(), routerconfig.SystemOneProvider) {
			t.Fatalf("a %s decision model must be refused naming both providers, got %v", provider, err)
		}
	}
}

func TestDecisionModelFamilyIsNotAnOperatorAllowlist(t *testing.T) {
	_, cfg := routerConfigOf(t, decisionDeploymentSpec(t, "kai", "vllm-sr/Decision-2.0-Kai-0.6B"))
	if cfg.DecisionModel != "kai" {
		t.Fatalf("default reference changed: %q", cfg.DecisionModel)
	}
}
