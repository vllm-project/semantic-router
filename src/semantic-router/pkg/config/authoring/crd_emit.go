package authoring

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"

// EmitCRD compiles the canonical authoring contract into the SemanticRouter CRD
// envelope used by the operator path today.
func EmitCRD(cfg *Config, name, namespace string) ([]byte, error) {
	runtimeCfg, err := CompileRuntime(cfg)
	if err != nil {
		return nil, err
	}
	return dsl.EmitCRD(runtimeCfg, name, namespace)
}

// EmitCRDFile reads a canonical authoring config file and emits a
// SemanticRouter CRD envelope from it.
func EmitCRDFile(path, name, namespace string) ([]byte, error) {
	cfg, err := ParseFile(path)
	if err != nil {
		return nil, err
	}
	return EmitCRD(cfg, name, namespace)
}
