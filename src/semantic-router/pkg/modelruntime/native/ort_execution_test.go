package native

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestORTCompilationCacheOptionsAndResourceIdentity(t *testing.T) {
	root := t.TempDir()
	artifact := filepath.Join(root, "model")
	if err := os.Mkdir(artifact, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artifact, "model.onnx"), []byte("resource identity fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Adapter: "mmbert"}, Deployment: config.ModelDeployment{Artifact: artifact, Provider: "ort", Device: "migraphx:0", CompilationCacheDir: filepath.Join(root, "cache-a")}}
	runtime := New(nil)
	loads := 0
	for _, directory := range []string{filepath.Join(root, "cache-a"), filepath.Join(root, "cache-a"), filepath.Join(root, "cache-b")} {
		spec.Deployment.CompilationCacheDir = directory
		resource, err := runtime.ortResource(context.Background(), spec, "sequence", func(options ort.Options) (io.Closer, error) {
			loads++
			if options.CompilationCacheDir != directory || options.Provider != "migraphx" || options.AllowCPUFallback {
				t.Fatalf("cache was not passed to strict typed loader: %+v", options)
			}
			return io.NopCloser(strings.NewReader("")), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = resource.Close() })
	}
	if loads != 2 {
		t.Fatalf("cache destination must separate resource identity; got %d loads", loads)
	}
	for _, directory := range []string{artifact, filepath.Join(artifact, "cache"), "relative"} {
		spec.Deployment.CompilationCacheDir = directory
		if _, err := ortOptions(spec); err == nil {
			t.Fatalf("accepted unsafe cache location %q", directory)
		}
	}
	alias := filepath.Join(root, "model-alias")
	if err := os.Symlink(artifact, alias); err != nil {
		t.Fatal(err)
	}
	spec.Deployment.CompilationCacheDir = filepath.Join(alias, "not-created-yet")
	if _, err := ortOptions(spec); err == nil {
		t.Fatal("accepted symlinked cache parent inside artifact")
	}
	spec.Deployment.CompilationCacheDir = filepath.Join(root, "cache")
	spec.Deployment.Device = "cpu"
	if _, err := ortOptions(spec); err == nil {
		t.Fatal("accepted MIGraphX cache on CPU")
	}
	spec.Deployment.CompilationCacheDir = ""
	options, err := ortOptions(spec)
	if err != nil || options.Provider != "cpu" || options.CompilationCacheDir != "" {
		t.Fatalf("default CPU path changed: %+v %v", options, err)
	}
}

func TestORTROCmOptionsAndObservedExecutionMustAgree(t *testing.T) {
	spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Contract: config.RemoteClassifierContractLabelScores, Head: "onnx/model_fa_fp16.onnx"}, Deployment: config.ModelDeployment{Provider: "ort", Artifact: "model", Device: "rocm:3", Precision: "native", CustomOpsProfile: "ck_flash_attention", Input: config.ModelInputBudget{MaxTokens: 32768}}}
	options, err := ortOptions(spec)
	if err != nil {
		t.Fatal(err)
	}
	if options.Provider != "rocm" || options.DeviceID != 3 || options.AllowCPUFallback || options.CustomOpsProfile != "ck_flash_attention" || options.MaxInputTokens != 32768 {
		t.Fatalf("wrong options: %+v", options)
	}
	session := ort.SessionEvidence{Provider: "ROCMExecutionProvider", DeviceID: 3, Precision: "native", CPUFallbackDisabled: true, CustomOpsProfile: "ck_flash_attention", CustomOpsLibrary: "/usr/local/lib/libort_ck_flash_attn.so.1", CustomOpsSHA256: strings.Repeat("ab", 32)}
	info := ort.Info{ModelLimit: 32768, TaskLimit: 32768, Labels: []string{"a", "b"}, Sessions: []ort.SessionEvidence{session}}
	if _, err := ortCapability(spec, info); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*ort.SessionEvidence){
		"fallback":  func(s *ort.SessionEvidence) { s.CPUFallbackDisabled = false },
		"device":    func(s *ort.SessionEvidence) { s.DeviceID = 0 },
		"precision": func(s *ort.SessionEvidence) { s.Precision = "fp16" },
		"profile":   func(s *ort.SessionEvidence) { s.CustomOpsProfile = "none" },
		"library":   func(s *ort.SessionEvidence) { s.CustomOpsLibrary = "/tmp/untrusted.so" },
		"hash":      func(s *ort.SessionEvidence) { s.CustomOpsSHA256 = "unknown" },
	} {
		t.Run(name, func(t *testing.T) {
			bad := session
			change(&bad)
			info.Sessions = []ort.SessionEvidence{bad}
			if _, err := ortCapability(spec, info); err == nil {
				t.Fatal("accepted execution without matching evidence")
			}
		})
	}
	for _, device := range []string{"rocm:-1", "rocm:bad", "cpu"} {
		bad := spec
		bad.Deployment.Device = device
		if _, err := ortOptions(bad); err == nil {
			t.Fatalf("accepted invalid CK device %q", device)
		}
	}
	spec.Deployment.Precision = "fp16"
	if _, err := ortOptions(spec); err == nil {
		t.Fatal("silently enabled ROCm graph conversion")
	}
}

func TestORTWindowResourcesSeparatePhysicalAndDocumentBudgets(t *testing.T) {
	artifact := t.TempDir()
	if err := os.WriteFile(filepath.Join(artifact, "model.onnx"), []byte("identity fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Adapter: "modernbert"}, Deployment: config.ModelDeployment{Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 65536, Overflow: "reject"}}}
	runtime, loads := New(nil), 0
	for _, budget := range []int{65536, 65536, 262144} {
		spec.Deployment.Input.MaxTokens = budget
		resource, err := runtime.ortResourceWithExecutionLimit(context.Background(), spec, "sequence", 32768, func(options ort.Options) (io.Closer, error) {
			loads++
			if options.MaxInputTokens != 32768 || options.ExecutionMaxInputTokens != 32768 || options.DocumentMaxInputTokens != budget || options.Overflow != "reject" {
				t.Fatalf("mixed physical and document limits: %+v", options)
			}
			return io.NopCloser(strings.NewReader("")), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = resource.Close() })
	}
	if loads != 2 {
		t.Fatalf("document budget must separate native ownership, loads=%d", loads)
	}
}
