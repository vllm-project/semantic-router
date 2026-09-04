package modeldownload

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	yamlv3 "gopkg.in/yaml.v3"
)

func TestExternalGatewayResponsesProfileRequiresNoLocalModelDownloads(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test file path")
	}
	valuesPath := filepath.Clean(filepath.Join(
		filepath.Dir(file),
		"../../../../deploy/kubernetes/ai-gateway/semantic-router-values/responses-state.yaml",
	))
	data, err := os.ReadFile(valuesPath)
	if err != nil {
		t.Fatalf("read %s: %v", valuesPath, err)
	}

	var values struct {
		Config         interface{} `yaml:"config"`
		ConfigOverride interface{} `yaml:"configOverride"`
	}
	if err := yamlv3.Unmarshal(data, &values); err != nil {
		t.Fatalf("decode Helm values: %v", err)
	}
	routerConfig := values.ConfigOverride
	if routerConfig == nil {
		routerConfig = values.Config
	}
	configYAML, err := yamlv3.Marshal(routerConfig)
	if err != nil {
		t.Fatalf("encode embedded Router config: %v", err)
	}
	cfg, err := config.ParseYAMLBytes(configYAML)
	if err != nil {
		t.Fatalf("parse embedded Router config: %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("state-only profile requested local model downloads: %#v", specs)
	}
}
