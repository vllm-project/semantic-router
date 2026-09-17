package native

import (
	"encoding/json"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
)

func TestRelevanceIdentityBindsUncachedCompilerEvidence(t *testing.T) {
	decode := func(raw string) ort.SessionEvidence {
		t.Helper()
		var session ort.SessionEvidence
		if err := json.Unmarshal([]byte(raw), &session); err != nil {
			t.Fatal(err)
		}
		return session
	}
	identity := func(session ort.SessionEvidence) string {
		t.Helper()
		data, err := json.Marshal(ortRelevanceExecution(session, "migraphx:0", "graph", "weights"))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	baseline := decode(`{"provider":"MIGraphXExecutionProvider","precision":"native","runtime_build":"fixed","compiler_flags":{"bf16":"false"}}`)
	rocblas := decode(`{"provider":"MIGraphXExecutionProvider","precision":"native","runtime_build":"fixed","compiler_flags":{"bf16":"false","environment:MIGRAPHX_SET_GEMM_PROVIDER":"rocblas"}}`)
	if baseline.CompilationCache != nil || rocblas.CompilationCache != nil {
		t.Fatal("fixture unexpectedly requires a compilation cache")
	}
	if identity(baseline) == identity(rocblas) {
		t.Fatal("different uncached compiler settings shared a ranking identity")
	}
	before := identity(rocblas)
	rocblas.ProfilePrefix = "new-process-profile"
	rocblas.Graph = "/relocated/model.onnx"
	rocblas.CompilationCache = &ort.CompilationCacheEvidence{Key: "observational-cache-key", State: "reused"}
	if before != identity(rocblas) {
		t.Fatal("observational paths or cache state changed ranking identity")
	}
}
