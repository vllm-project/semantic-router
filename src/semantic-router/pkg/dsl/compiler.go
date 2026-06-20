package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Compiler transforms a DSL AST into a RouterConfig.
type Compiler struct {
	prog            *Program
	config          *config.RouterConfig
	pluginTemplates map[string]*PluginDecl // name → template
	errors          []error
}

// Compile parses a DSL source string and compiles it to a RouterConfig.

func Compile(input string) (*config.RouterConfig, []error) {
	prog, parseErrors := Parse(input)
	if len(parseErrors) > 0 {
		return nil, parseErrors
	}
	return CompileAST(prog)
}

func CompileAST(prog *Program) (*config.RouterConfig, []error) {
	defaults := config.DefaultGlobalConfig()
	c := &Compiler{
		prog: prog,
		config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ModelSelection: defaults.ModelSelection,
			},
		},
		pluginTemplates: make(map[string]*PluginDecl),
	}
	c.compile()
	if len(c.errors) > 0 {
		return nil, c.errors
	}
	return c.config, nil
}

func (c *Compiler) compile() {
	// 1. Register plugin templates
	for _, p := range c.prog.Plugins {
		c.pluginTemplates[p.Name] = p
	}

	// 2. Compile signals
	c.compileSignals()

	// 3. Compile projection partitions
	c.compileProjectionPartitions()

	// 4. Compile projections
	c.compileProjectionScores()
	c.compileProjectionMappings()

	// 5. Compile top-level model catalog
	c.compileModels()

	// 6. Compile routes (decisions)
	c.compileRoutes()
}

func (c *Compiler) compileProjectionPartitions() {
	for _, partitionDecl := range c.prog.ProjectionPartitions {
		c.validateSoftmaxDomainProjectionPartition(partitionDecl)
		partition := config.ProjectionPartition{
			Name:        partitionDecl.Name,
			Semantics:   partitionDecl.Semantics,
			Temperature: partitionDecl.Temperature,
			Members:     partitionDecl.Members,
			Default:     partitionDecl.Default,
		}
		c.config.Projections.Partitions = append(c.config.Projections.Partitions, partition)
	}
}

func (c *Compiler) compileProjectionScores() {
	for _, scoreDecl := range c.prog.ProjectionScores {
		score := config.ProjectionScore{
			Name:   scoreDecl.Name,
			Method: scoreDecl.Method,
			Inputs: make([]config.ProjectionScoreInput, 0, len(scoreDecl.Inputs)),
		}
		for _, inputDecl := range scoreDecl.Inputs {
			score.Inputs = append(score.Inputs, config.ProjectionScoreInput{
				Type:        inputDecl.SignalType,
				Name:        inputDecl.SignalName,
				KB:          inputDecl.KB,
				Metric:      inputDecl.Metric,
				Weight:      inputDecl.Weight,
				ValueSource: inputDecl.ValueSource,
				Match:       inputDecl.Match,
				Miss:        inputDecl.Miss,
			})
		}
		c.config.Projections.Scores = append(c.config.Projections.Scores, score)
	}
}

func (c *Compiler) compileProjectionMappings() {
	for _, mappingDecl := range c.prog.ProjectionMappings {
		mapping := config.ProjectionMapping{
			Name:   mappingDecl.Name,
			Source: mappingDecl.Source,
			Method: mappingDecl.Method,
		}
		if mappingDecl.Calibration != nil {
			mapping.Calibration = &config.ProjectionMappingCalibration{
				Method: mappingDecl.Calibration.Method,
				Slope:  mappingDecl.Calibration.Slope,
			}
		}
		for _, outputDecl := range mappingDecl.Outputs {
			mapping.Outputs = append(mapping.Outputs, config.ProjectionMappingOutput{
				Name: outputDecl.Name,
				LT:   outputDecl.LT,
				LTE:  outputDecl.LTE,
				GT:   outputDecl.GT,
				GTE:  outputDecl.GTE,
			})
		}
		c.config.Projections.Mappings = append(c.config.Projections.Mappings, mapping)
	}
}

func (c *Compiler) addError(pos Position, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	c.errors = append(c.errors, fmt.Errorf("%s: %s", pos, msg))
}

func (c *Compiler) validateEmitForCompile(r *RouteDecl, e *EmitDecl) bool {
	if e == nil {
		return false
	}
	context := fmt.Sprintf("ROUTE %s EMIT", r.Name)
	if !supportedEmitKinds[e.Kind] {
		c.addError(e.Pos, "%s: unknown EMIT kind %q. Supported kinds: retention", context, e.Kind)
		return false
	}
	if e.Kind != emitKindRetention {
		return true
	}
	ok := true
	for _, issue := range retentionRawFieldDiagnostics(e, context) {
		c.addError(issue.pos, "%s", issue.message)
		ok = false
	}
	if e.Retention == nil {
		return ok
	}
	if e.Retention.TTLTurns != nil && *e.Retention.TTLTurns < 0 {
		c.addError(e.Pos, "%s retention: ttl_turns must be >= 0, got %d", context, *e.Retention.TTLTurns)
		ok = false
	}
	if e.Retention.Drop != nil && *e.Retention.Drop && e.Retention.TTLTurns != nil && *e.Retention.TTLTurns > 0 {
		c.addError(e.Pos, "%s retention: drop=true conflicts with ttl_turns=%d. Use one or the other", context, *e.Retention.TTLTurns)
		ok = false
	}
	return ok
}

func compileEmitDecl(e *EmitDecl) config.EmitDirective {
	if e == nil {
		return config.EmitDirective{}
	}
	out := config.EmitDirective{Kind: e.Kind}
	if e.Retention != nil {
		out.Retention = &config.RetentionDirective{
			Drop:                  clonePtrBool(e.Retention.Drop),
			TTLTurns:              clonePtrInt(e.Retention.TTLTurns),
			KeepCurrentModel:      clonePtrBool(e.Retention.KeepCurrentModel),
			PreferPrefixRetention: clonePtrBool(e.Retention.PreferPrefixRetention),
		}
	}
	return out
}

func clonePtrBool(p *bool) *bool {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func clonePtrInt(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
