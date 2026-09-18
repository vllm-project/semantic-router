package evaluationplane

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func sequenceClassifierTestYAML() string {
	withExternal := strings.Replace(
		multiRecipeMixtureTestYAML,
		"  integrations:\n",
		"  model_catalog:\n"+
			"    external:\n"+
			"      - name: risk-score\n"+
			"        model_role: classification\n"+
			"        llm_endpoint:\n"+
			"          name: risk-http\n"+
			"          address: classifier.private.test\n"+
			"          port: 8080\n"+
			"        access_key: classifier-secret\n"+
			"  integrations:\n",
		1,
	)
	return strings.Replace(
		withExternal,
		"routing:\n  modelCards:",
		"routing:\n  signals:\n"+
			"    classifiers:\n"+
			"      - name: request-risk\n"+
			"        type: sequence_classifier\n"+
			"        model: risk-score\n"+
			"        labels: [SAFE, RISKY]\n"+
			"  modelCards:",
		1,
	)
}

func TestSequenceClassifierSupportDoesNotRequireLLMModelName(t *testing.T) {
	snapshot, err := ModelArmSnapshotFromYAML([]byte(sequenceClassifierTestYAML()), "revision")
	if err != nil {
		t.Fatalf("sequence classifier snapshot: %v", err)
	}
	mixture := mixtureForRecipe(t, snapshot, "default")
	support := requireSnapshotSupportModel(t, mixture, "risk-score")
	if !mixture.Ready || support.ProviderModelIDDigest != digestString("risk-score") ||
		!digestPattern.MatchString(support.ConfigDigest) ||
		!digestPattern.MatchString(support.BackendTopologyDigest) {
		t.Fatalf("sequence classifier support is not executable: mixture=%#v support=%#v", mixture, support)
	}
	for _, arm := range mixture.Mixture.ModelArms {
		if arm.Model == support.Model {
			t.Fatalf("selector-only sequence classifier entered the candidate pool: %#v", mixture.Mixture.ModelArms)
		}
	}
	if validationErr := validateMixtureContract(&mixture.Mixture); validationErr != nil {
		t.Fatalf("sequence classifier mixture contract: %v", validationErr)
	}
	encoded, err := json.Marshal(mixture.Mixture)
	if err != nil {
		t.Fatalf("marshal mixture: %v", err)
	}
	for _, privateValue := range []string{"classifier.private.test", "classifier-secret"} {
		if strings.Contains(string(encoded), privateValue) {
			t.Fatalf("public sequence classifier snapshot leaked %q: %s", privateValue, encoded)
		}
	}
}

func TestSequenceClassifierConfigAndTopologyDriftStayOrthogonal(t *testing.T) {
	base := mustSequenceClassifierMixture(t, sequenceClassifierTestYAML())
	baseSupport := requireSnapshotSupportModel(t, base, "risk-score")

	configYAML := strings.Replace(sequenceClassifierTestYAML(), "name: risk-http", "name: risk-http-v2", 1)
	config := mustSequenceClassifierMixture(t, configYAML)
	configSupport := requireSnapshotSupportModel(t, config, "risk-score")
	if baseSupport.ConfigDigest == configSupport.ConfigDigest ||
		baseSupport.BackendTopologyDigest != configSupport.BackendTopologyDigest ||
		base.Mixture.SelectorDigest == config.Mixture.SelectorDigest {
		t.Fatalf("endpoint-name drift was not isolated to selector config: base=%#v changed=%#v", baseSupport, configSupport)
	}
	assertSelectorOnlyDrift(t, base, config)

	topologyYAML := strings.Replace(
		sequenceClassifierTestYAML(), "classifier.private.test", "replacement.private.test", 1,
	)
	topology := mustSequenceClassifierMixture(t, topologyYAML)
	topologySupport := requireSnapshotSupportModel(t, topology, "risk-score")
	if baseSupport.ConfigDigest != topologySupport.ConfigDigest ||
		baseSupport.BackendTopologyDigest == topologySupport.BackendTopologyDigest ||
		base.Mixture.SelectorDigest == topology.Mixture.SelectorDigest {
		t.Fatalf("address drift was not isolated to selector topology: base=%#v changed=%#v", baseSupport, topologySupport)
	}
	assertSelectorOnlyDrift(t, base, topology)
}

func TestSequenceClassifierFingerprintUsesEffectiveRuntimeFieldsOnly(t *testing.T) {
	base := mustSequenceClassifierMixture(t, sequenceClassifierTestYAML())
	baseSupport := requireSnapshotSupportModel(t, base, "risk-score")
	explicitDefaults := strings.Replace(
		sequenceClassifierTestYAML(),
		"          port: 8080\n",
		"          port: 8080\n"+
			"          protocol: http\n"+
			"        llm_timeout_seconds: 5\n"+
			"        max_request_bytes: 262144\n"+
			"        max_response_bytes: 1048576\n",
		1,
	)
	explicit := mustSequenceClassifierMixture(t, explicitDefaults)
	if got := requireSnapshotSupportModel(t, explicit, "risk-score"); got != baseSupport {
		t.Fatalf("explicit runtime defaults changed support semantics: base=%#v explicit=%#v", baseSupport, got)
	}

	llmOnly := strings.Replace(
		sequenceClassifierTestYAML(),
		"          port: 8080\n",
		"          port: 8080\n"+
			"          use_chat_template: true\n"+
			"          prompt_template: ignored-template\n"+
			"        llm_provider: ignored-provider\n"+
			"        llm_model_name: ignored/model\n"+
			"        parser_type: json\n"+
			"        threshold: 0.75\n"+
			"        max_tokens: 64\n"+
			"        temperature: 0.25\n",
		1,
	)
	ignored := mustSequenceClassifierMixture(t, llmOnly)
	if got := requireSnapshotSupportModel(t, ignored, "risk-score"); got != baseSupport {
		t.Fatalf("LLM-only fields changed sequence classifier semantics: base=%#v changed=%#v", baseSupport, got)
	}
}

func TestSequenceClassifierConfigDigestOwnsRuntimeIdentity(t *testing.T) {
	baseConfig := routerconfig.ExternalModelConfig{
		Name: "risk-score", ModelRole: routerconfig.ModelRoleClassification,
		ModelEndpoint: routerconfig.ClassifierVLLMEndpoint{
			Name: "risk-http", Address: "classifier.private.test", Port: 8080,
		},
	}
	base, valid := externalSelectorSupportModel(
		baseConfig, routerconfig.ClassifierSignalTypeSequenceClassifier,
	)
	if !valid {
		t.Fatal("base sequence classifier support is invalid")
	}

	tests := []struct {
		name        string
		backendType string
		mutate      func(*routerconfig.ExternalModelConfig)
	}{
		{name: "model role", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.ModelRole = routerconfig.ModelRoleGuardrail
		}},
		{name: "endpoint name", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.ModelEndpoint.Name = "risk-http-v2"
		}},
		{name: "timeout", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.TimeoutSeconds = 7
		}},
		{name: "request limit", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.MaxRequestBytes = 131072
		}},
		{name: "response limit", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.MaxResponseBytes = 524288
		}},
		{
			name: "backend type", backendType: routerconfig.ClassifierSignalTypeLLM,
			mutate: func(config *routerconfig.ExternalModelConfig) {
				config.ModelName = config.Name
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changedConfig := baseConfig
			test.mutate(&changedConfig)
			backendType := test.backendType
			if backendType == "" {
				backendType = routerconfig.ClassifierSignalTypeSequenceClassifier
			}
			changed, changedValid := externalSelectorSupportModel(changedConfig, backendType)
			if !changedValid || changed.ConfigDigest == base.ConfigDigest ||
				changed.BackendTopologyDigest != base.BackendTopologyDigest {
				t.Fatalf("config field did not stay config-owned: base=%#v changed=%#v", base, changed)
			}
		})
	}
}

func TestSequenceClassifierTopologyOwnsOnlyAddressPortAndProtocol(t *testing.T) {
	baseConfig := routerconfig.ExternalModelConfig{
		Name: "risk-score", ModelRole: routerconfig.ModelRoleClassification,
		ModelEndpoint: routerconfig.ClassifierVLLMEndpoint{
			Name: "risk-http", Address: "classifier.private.test", Port: 8080,
		},
	}
	base, valid := externalSelectorSupportModel(
		baseConfig, routerconfig.ClassifierSignalTypeSequenceClassifier,
	)
	if !valid {
		t.Fatal("base sequence classifier support is invalid")
	}

	tests := []struct {
		name   string
		mutate func(*routerconfig.ExternalModelConfig)
	}{
		{name: "address", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.ModelEndpoint.Address = "replacement.private.test"
		}},
		{name: "port", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.ModelEndpoint.Port = 8081
		}},
		{name: "protocol", mutate: func(config *routerconfig.ExternalModelConfig) {
			config.ModelEndpoint.Protocol = "https"
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changedConfig := baseConfig
			test.mutate(&changedConfig)
			changed, changedValid := externalSelectorSupportModel(
				changedConfig, routerconfig.ClassifierSignalTypeSequenceClassifier,
			)
			if !changedValid || changed.ConfigDigest != base.ConfigDigest ||
				changed.BackendTopologyDigest == base.BackendTopologyDigest {
				t.Fatalf("topology field crossed config ownership: base=%#v changed=%#v", base, changed)
			}
		})
	}
}

func mustSequenceClassifierMixture(t *testing.T, data string) MixtureTargetSnapshot {
	t.Helper()
	snapshot, err := ModelArmSnapshotFromYAML([]byte(data), "revision")
	if err != nil {
		t.Fatalf("sequence classifier snapshot: %v", err)
	}
	mixture := mixtureForRecipe(t, snapshot, "default")
	if !mixture.Ready {
		t.Fatalf("sequence classifier mixture is unavailable: %#v", mixture)
	}
	return mixture
}

func requireSnapshotSupportModel(
	t *testing.T,
	mixture MixtureTargetSnapshot,
	model string,
) SupportModel {
	t.Helper()
	for _, support := range mixture.Mixture.SupportModels {
		if support.Model == model {
			return support
		}
	}
	t.Fatalf("support model %q is unavailable in %#v", model, mixture.Mixture.SupportModels)
	return SupportModel{}
}

func assertSelectorOnlyDrift(t *testing.T, left, right MixtureTargetSnapshot) {
	t.Helper()
	if left.Mixture.RecipeDigest != right.Mixture.RecipeDigest ||
		left.Mixture.BindingDigest != right.Mixture.BindingDigest ||
		left.Mixture.PoolDigest != right.Mixture.PoolDigest ||
		left.Mixture.SelectorPolicyDigest != right.Mixture.SelectorPolicyDigest ||
		left.Mixture.AdaptationDigest != right.Mixture.AdaptationDigest ||
		left.BackendTopologyDigest != right.BackendTopologyDigest ||
		!reflect.DeepEqual(left.Mixture.ModelArms, right.Mixture.ModelArms) {
		t.Fatalf("selector support drift crossed another factor: left=%#v right=%#v", left, right)
	}
}
