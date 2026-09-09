package controllers

import (
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func intPtr(value int) *int {
	return &value
}

// The CRD's complexity_model block must land on
// global.model_catalog.modules.complexity with the router's own field names,
// so the router's validator - not the operator - decides what is valid.
func assertOperatorComplexityModel(t *testing.T, module routerconfig.ComplexityModelConfig) {
	t.Helper()

	backend := module.Backend
	if backend == nil {
		t.Fatalf("complexity backend was not carried onto the canonical config: %#v", module)
	}
	if backend.Protocol != "http_classify" || backend.Contract != "score.v1" || backend.Model != "difficulty-scorer" {
		t.Fatalf("unexpected complexity backend: %#v", backend)
	}
	if backend.DeadlineMs == nil || *backend.DeadlineMs != 2500 {
		t.Fatalf("deadline_ms did not survive conversion: %#v", backend.DeadlineMs)
	}
}
