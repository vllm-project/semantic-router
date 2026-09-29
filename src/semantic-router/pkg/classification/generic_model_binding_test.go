package classification

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestGenericBoundRulesUseDistinctEndpointsAndIndependentClose(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelBindings = map[string]config.ModelBinding{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{}
	for _, name := range []string{"first.rule", "second.rule"} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode([]httpClassifyLabelScore{{Label: name, Score: 0.75}, {Label: "other", Score: 0.25}})
		}))
		t.Cleanup(server.Close)
		cfg.ExternalModels = append(cfg.ExternalModels, config.ExternalModelConfig{Name: name, ModelRole: config.ModelRoleClassification, ModelEndpoint: endpointForTestServer(t, server)})
		cfg.ClassifierRules = append(cfg.ClassifierRules, config.ClassifierSignalRule{Name: name, Type: config.ClassifierSignalTypeLocal, ModelPath: "/obsolete/unloadable", Labels: []string{name, "other"}})
		cfg.ModelDeployments[name] = config.ModelDeployment{Provider: "http", ExternalModel: name}
		cfg.ModelBindings["classifier."+name] = config.ModelBinding{Deployment: name, Adapter: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractLabelDistribution}
	}
	models, err := newClassifierModelRuntime(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	builder := &classifierOptionBuilder{cfg: models.cfg, models: models}
	apply, err := builder.buildGenericClassifiersOption()
	if err != nil {
		t.Fatal(err)
	}
	classifier := &Classifier{}
	apply(classifier)
	t.Cleanup(func() { closeLabelClassifiers(classifier.genericClassifiers) })
	for name, task := range classifier.genericClassifiers {
		result, callErr := task.Classify(context.Background(), "hello")
		if callErr != nil || result.Scores[name] != 0.75 || result.Scores["other"] != 0.25 {
			t.Fatalf("%s result=%+v err=%v", name, result, callErr)
		}
	}
	first := classifier.genericClassifiers["first.rule"]
	if closeErr := first.(interface{ Close() error }).Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if _, callErr := first.Classify(context.Background(), "closed"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("closed first task error=%v", callErr)
	}
	result, err := classifier.genericClassifiers["second.rule"].Classify(context.Background(), "still active")
	if err != nil || result.Scores["second.rule"] != 0.75 {
		t.Fatalf("second task closed with first: %+v %v", result, err)
	}
	if cfg.ClassifierRules[0].ModelPath != "/obsolete/unloadable" {
		t.Fatal("canonical selector mutated")
	}
}
