package routerauthoring

import (
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

// Public DSL aliases keep control-plane code on an explicit authoring seam.
type (
	Diagnostic = dsl.Diagnostic
	Program    = dsl.Program
	SignalDecl = dsl.SignalDecl
	ArrayValue = dsl.ArrayValue
)

// ValidateWithSymbols runs the public DSL validator used by control-plane
// authoring flows.
func ValidateWithSymbols(input string) ([]Diagnostic, *dsl.SymbolTable, []error) {
	return dsl.ValidateWithSymbols(input)
}

// Parse returns the parsed public DSL program.
func Parse(input string) (*Program, []error) {
	return dsl.Parse(input)
}

// Compile validates authoring DSL and returns the normalized canonical config
// contract instead of the internal runtime config.
func Compile(input string) (*routercontract.CanonicalConfig, []error) {
	cfg, errs := dsl.Compile(input)
	if len(errs) > 0 {
		return nil, errs
	}
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	return &canonical, nil
}

// Format canonicalizes DSL input for control-plane editing flows.
func Format(input string) (string, error) {
	return dsl.Format(input)
}

// DecompileRouting exports canonical routing config into public DSL text.
func DecompileRouting(canonical *routercontract.CanonicalConfig) (string, error) {
	if canonical == nil {
		return "", nil
	}
	data, err := yaml.Marshal(canonical)
	if err != nil {
		return "", err
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		return "", err
	}
	return dsl.DecompileRouting(cfg)
}

// EmitRoutingYAMLFromConfig emits the routing-only YAML fragment used by
// dashboard deploy flows from the public canonical contract.
func EmitRoutingYAMLFromConfig(canonical *routercontract.CanonicalConfig) ([]byte, error) {
	type routingFragmentDocument struct {
		Routing routercontract.CanonicalRouting `yaml:"routing"`
	}
	if canonical == nil {
		return yaml.Marshal(routingFragmentDocument{})
	}
	return yaml.Marshal(routingFragmentDocument{Routing: canonical.Routing})
}
