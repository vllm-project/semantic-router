package controllers

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type operatorContractCorpus struct {
	SupportedVersion string                       `json:"supported_version"`
	SteadyState      []operatorContractSteadyCase `json:"steady_state"`
}

type operatorContractSteadyCase struct {
	Name              string `json:"name"`
	Input             string `json:"input"`
	Valid             bool   `json:"valid"`
	Error             string `json:"error"`
	NormalizedVersion string `json:"normalized_version"`
}

func TestOperatorOutputExecutesCanonicalContractGoldenCorpus(t *testing.T) {
	t.Parallel()

	corpus := loadOperatorContractCorpus(t)
	if corpus.SupportedVersion != routerconfig.CanonicalVersion {
		t.Fatalf("corpus version = %q, contract version = %q", corpus.SupportedVersion, routerconfig.CanonicalVersion)
	}

	reconciler := &SemanticRouterReconciler{}
	for _, test := range corpus.SteadyState {
		t.Run(test.Name, func(t *testing.T) {
			t.Parallel()
			assertOperatorContractCase(t, reconciler, test)
		})
	}
}

func loadOperatorContractCorpus(t *testing.T) operatorContractCorpus {
	t.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	data, err := os.ReadFile(filepath.Join(
		filepath.Dir(filename),
		"..", "..", "..",
		"src", "semantic-router", "pkg", "config", "testdata",
		"canonical_contract_cases.json",
	))
	if err != nil {
		t.Fatalf("read canonical contract corpus: %v", err)
	}
	var corpus operatorContractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode canonical contract corpus: %v", err)
	}
	return corpus
}

func assertOperatorContractCase(
	t *testing.T,
	reconciler *SemanticRouterReconciler,
	test operatorContractSteadyCase,
) {
	t.Helper()

	if _, err := routerconfig.ParseYAMLBytes([]byte(test.Input)); !test.Valid {
		if err == nil {
			t.Fatal("expected operator input contract rejection")
		}
		if !strings.Contains(err.Error(), test.Error) {
			t.Fatalf("expected %q in error, got %v", test.Error, err)
		}
		return
	} else if err != nil {
		t.Fatalf("expected operator input contract acceptance, got %v", err)
	}

	routing, err := routingOverrideFromCorpus(test.Input)
	if err != nil {
		t.Fatalf("extract routing override: %v", err)
	}
	canonical, err := reconciler.buildCanonicalConfig(
		context.Background(),
		&vllmv1alpha1.SemanticRouter{
			Spec: vllmv1alpha1.SemanticRouterSpec{
				Config: vllmv1alpha1.ConfigSpec{Routing: routing},
			},
		},
	)
	if err != nil {
		t.Fatalf("build operator canonical output: %v", err)
	}
	if canonical.Version != test.NormalizedVersion {
		t.Fatalf(
			"normalized version = %q, want %q",
			canonical.Version,
			test.NormalizedVersion,
		)
	}
}

func routingOverrideFromCorpus(input string) (*apiextensionsv1.JSON, error) {
	var document map[string]interface{}
	if err := yaml.Unmarshal([]byte(input), &document); err != nil {
		return nil, err
	}
	routing, ok := document["routing"]
	if !ok {
		return nil, nil
	}
	data, err := json.Marshal(routing)
	if err != nil {
		return nil, err
	}
	return &apiextensionsv1.JSON{Raw: data}, nil
}
