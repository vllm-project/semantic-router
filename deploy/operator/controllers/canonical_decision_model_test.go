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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
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

// applicationSignals are application signal types the decision model answers
// when no binding names a deployment for them.
var applicationSignals = []string{routerconfig.SignalTypeJailbreak, routerconfig.SignalTypePII, routerconfig.SignalTypeHallucination}

var onlyTheDecisionModel = map[string]interface{}{"decision_model": map[string]interface{}{"deployment": "primary"}}

func TestTheDecisionModelAnswersEveryApplicationSignal(t *testing.T) {
	document, cfg := routerConfigOf(t, decisionDeploymentSpec(t, "primary", "vllm-sr/Vela-2.0-9B"))
	catalog := modelCatalogOf(t, document)
	if !reflect.DeepEqual(catalog["system"], onlyTheDecisionModel) {
		t.Fatalf("the ConfigMap must name the decision model and pin no module, got %v", catalog["system"])
	}
	if written := writtenKeys(catalog["modules"], "modules", "threshold", "model_id", "model_ref", "model_path"); len(written) != 0 {
		t.Fatalf("the ConfigMap pins module models or thresholds: %v", written)
	}
	for _, signalType := range applicationSignals {
		if deployment, source := cfg.ApplicationTypeDeployment(signalType); deployment != "primary" || source != "default" {
			t.Fatalf("%s rules must follow the decision model, got %q (%s)", signalType, deployment, source)
		}
	}
	// A rule without a predicate takes the threshold calibrated on the 9B.
	for signalType, preset := range map[string]string{routerconfig.SignalTypeJailbreak: presets.Jailbreak, routerconfig.SignalTypePII: presets.PII} {
		want, calibrated := presets.Threshold("vllm-sr/Vela-2.0-9B", preset, presets.StageRequest)
		if got := cfg.PresetThreshold(signalType, preset); !calibrated || got != want {
			t.Fatalf("%s rules take %v, want the 9B's calibrated %v", signalType, got, want)
		}
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
	for _, signalType := range applicationSignals {
		if deployment, source := cfg.ApplicationTypeDeployment(signalType); deployment != "d3-remote" || source != "default" {
			t.Fatalf("%s rules must follow the systemone decision model, got %q (%s)", signalType, deployment, source)
		}
	}
	// A remote model has no calibration here, so rules take the neutral
	// threshold rather than vela3-fast's.
	if got := cfg.PresetThreshold(routerconfig.SignalTypeJailbreak, presets.Jailbreak); got != presets.NeutralThreshold {
		t.Fatalf("jailbreak rules on the remote model take %v, want %v", got, presets.NeutralThreshold)
	}
	if key, err := cfg.ModelDeployments["d3-remote"].Auth.Key(); err != nil || key != "operator-test-key" {
		t.Fatalf("auth.api_key_env does not read the Pod environment: %v", err)
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
	if domain, _ := cfg.ApplicationTypeDeployment(routerconfig.SignalTypeDomain); domain != "d3-remote" {
		t.Fatalf("domain rules ask %q, want the decision model d3-remote", domain)
	}
	for signalType, preset := range map[string]string{routerconfig.SignalTypePII: presets.PII, routerconfig.SignalTypeJailbreak: presets.Jailbreak} {
		if deployment, source := cfg.ApplicationTypeDeployment(signalType); deployment != "local-vela" || source != "global" {
			t.Fatalf("%s rules ask %q (%s), want the bound local-vela", signalType, deployment, source)
		}
		// A bound type takes the thresholds of the deployment that answers it.
		want, _ := presets.Threshold(routerconfig.GetModelByPath(routerconfig.Vela3SignalModel).RepoID, preset, presets.StageRequest)
		if got := cfg.PresetThreshold(signalType, preset); got != want {
			t.Fatalf("%s rules take %v, want vela3-fast's %v", signalType, got, want)
		}
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
