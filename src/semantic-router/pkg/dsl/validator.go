package dsl

import (
	"fmt"
	"sort"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DiagLevel represents the severity level of a diagnostic.
type DiagLevel int

const (
	DiagError      DiagLevel = iota // Level 1: Syntax errors (red)
	DiagWarning                     // Level 2: Reference errors (yellow)
	DiagConstraint                  // Level 3: Constraint violations (orange)
)

// String returns the human-readable label for a DiagLevel.
func (d DiagLevel) String() string {
	switch d {
	case DiagError:
		return "error"
	case DiagWarning:
		return "warning"
	case DiagConstraint:
		return "constraint"
	default:
		return fmt.Sprintf("DiagLevel(%d)", int(d))
	}
}

// QuickFix suggests an automated repair.
type QuickFix struct {
	Description string // e.g. "Change to \"math\""
	NewText     string // replacement text
}

// Diagnostic represents a single validation finding.
type Diagnostic struct {
	Level   DiagLevel
	Message string
	Pos     Position
	Fix     *QuickFix // optional auto-fix suggestion
}

// String returns a formatted diagnostic message.
func (d Diagnostic) String() string {
	prefix := ""
	switch d.Level {
	case DiagError:
		prefix = "🔴 Error"
	case DiagWarning:
		prefix = "🟡 Warning"
	case DiagConstraint:
		prefix = "🟠 Constraint"
	}
	s := fmt.Sprintf("%s: %s (at %s)", prefix, d.Message, d.Pos)
	if d.Fix != nil {
		s += fmt.Sprintf(" [Fix: %s]", d.Fix.Description)
	}
	return s
}

// Validator performs 3-level validation on a DSL AST.
type Validator struct {
	prog        *Program
	diagnostics []Diagnostic

	// Symbol tables (built during validation)
	signalNames map[string]map[string]bool // signalType → {name → true}
	modelNames  map[string]bool            // top-level model catalog entries
	modelLoRAs  map[string]map[string]bool // model name → lora adapter names
	pluginNames map[string]bool            // template name → true
}

// SymbolInfo represents a named symbol extracted from the AST.
type SymbolInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// SymbolTable contains all declared symbols from a DSL source, for use by
// editor features such as context-aware completion.
type SymbolTable struct {
	Signals []SymbolInfo `json:"signals"`
	Models  []string     `json:"models"`
	Plugins []string     `json:"plugins"`
	Routes  []string     `json:"routes"`
}

// Validate performs 3-level validation on a DSL source string.
// It first parses the input; Level 1 errors come from the parser.
// Then it runs Level 2 (reference checks) and Level 3 (constraint checks) on the AST.
func Validate(input string) ([]Diagnostic, []error) {
	diags, _, errs := ValidateWithSymbols(input)
	return diags, errs
}

// ValidateWithSymbols performs 3-level validation and also returns the symbol
// table extracted from the (possibly partial) AST. The symbol table is always
// populated, even when there are parse errors, because the parser recovers and
// successfully parsed declarations still appear in the AST.
func ValidateWithSymbols(input string) ([]Diagnostic, *SymbolTable, []error) {
	prog, parseErrors := Parse(input)
	if len(parseErrors) > 0 {
		var diags []Diagnostic
		for _, e := range parseErrors {
			diags = append(diags, Diagnostic{
				Level:   DiagError,
				Message: e.Error(),
				Pos:     Position{},
			})
		}
		// Still try to build symbol table from partial AST if available
		if prog == nil {
			return diags, &SymbolTable{}, parseErrors
		}
	}

	v := &Validator{
		prog:        prog,
		signalNames: make(map[string]map[string]bool),
		modelNames:  make(map[string]bool),
		modelLoRAs:  make(map[string]map[string]bool),
		pluginNames: make(map[string]bool),
	}

	// Level 1: Parser errors become Error diagnostics
	for _, e := range parseErrors {
		v.diagnostics = append(v.diagnostics, Diagnostic{
			Level:   DiagError,
			Message: e.Error(),
			Pos:     Position{},
		})
	}

	// Build symbol tables
	v.buildSymbolTable()

	// Level 2: Reference checks
	v.checkReferences()

	// Level 3: Constraint checks
	v.checkConstraints()

	// Level 4: Conflict detection
	v.checkConflicts()

	// Extract symbol table for editor completions
	symbols := v.extractSymbolTable()

	return v.diagnostics, symbols, parseErrors
}

// ValidateAST performs Level 2 and Level 3 validation on an existing AST.
func ValidateAST(prog *Program) []Diagnostic {
	v := &Validator{
		prog:        prog,
		signalNames: make(map[string]map[string]bool),
		modelNames:  make(map[string]bool),
		modelLoRAs:  make(map[string]map[string]bool),
		pluginNames: make(map[string]bool),
	}
	v.buildSymbolTable()
	v.checkReferences()
	v.checkConstraints()
	v.checkConflicts()
	return v.diagnostics
}

// ---------- Symbol Table ----------

func (v *Validator) buildSymbolTable() {
	for _, s := range v.prog.Signals {
		if v.signalNames[s.SignalType] == nil {
			v.signalNames[s.SignalType] = make(map[string]bool)
		}
		v.signalNames[s.SignalType][s.Name] = true
	}
	if v.signalNames[config.SignalTypeProjection] == nil {
		v.signalNames[config.SignalTypeProjection] = make(map[string]bool)
	}
	for _, mapping := range v.prog.ProjectionMappings {
		for _, output := range mapping.Outputs {
			v.signalNames[config.SignalTypeProjection][output.Name] = true
		}
	}

	for _, m := range v.prog.Models {
		v.modelNames[m.Name] = true
		if loras := collectModelLoRANames(m.Fields); len(loras) > 0 {
			v.modelLoRAs[m.Name] = loras
		}
	}

	for _, p := range v.prog.Plugins {
		v.pluginNames[p.Name] = true
	}
}

// extractSymbolTable builds a SymbolTable from the validator's symbol maps and AST.
func (v *Validator) extractSymbolTable() *SymbolTable {
	st := &SymbolTable{}

	// Signals: flatten signalType → names into a list of SymbolInfo
	for sigType, names := range v.signalNames {
		for name := range names {
			st.Signals = append(st.Signals, SymbolInfo{Name: name, Type: sigType})
		}
	}
	sort.Slice(st.Signals, func(i, j int) bool {
		if st.Signals[i].Type != st.Signals[j].Type {
			return st.Signals[i].Type < st.Signals[j].Type
		}
		return st.Signals[i].Name < st.Signals[j].Name
	})

	// Plugins
	st.Plugins = keysOfBool(v.pluginNames)

	// Models: prefer explicit top-level model catalog, but keep route refs visible
	// so completions still work when editing legacy DSL without model declarations.
	modelSet := make(map[string]bool)
	for name := range v.modelNames {
		modelSet[name] = true
	}
	for _, route := range v.prog.Routes {
		for _, m := range route.Models {
			if m.Model != "" {
				modelSet[m.Model] = true
			}
		}
	}
	st.Models = keysOfBool(modelSet)

	// Routes: collect route names
	for _, route := range v.prog.Routes {
		if route.Name != "" {
			st.Routes = append(st.Routes, route.Name)
		}
	}
	sort.Strings(st.Routes)

	return st
}

// ---------- Level 2: Reference Checks ----------

func (v *Validator) checkReferences() {
	for _, route := range v.prog.Routes {
		v.checkRouteReferences(route)
	}
}

func (v *Validator) checkRouteReferences(route *RouteDecl) {
	if route.When != nil {
		v.walkBoolExpr(route.When)
	}

	for _, pr := range route.Plugins {
		if !v.pluginNames[pr.Name] && !isInlinePluginType(pr.Name) {
			fix := v.suggestPlugin(pr.Name)
			v.addDiag(DiagWarning, pr.Pos,
				fmt.Sprintf("Plugin %q is not defined as a template and is not a recognized inline plugin type. Supported inline types: system_prompt, semantic_cache, hallucination, memory, rag, tools, image_gen, fast_response, request_params, router_replay, header_mutation, response_jailbreak. Define a template with PLUGIN %s <type> { ... } or use a supported type", pr.Name, pr.Name),
				fix,
			)
		}
	}

	if len(route.Models) == 0 {
		v.addDiag(DiagWarning, route.Pos,
			fmt.Sprintf("Route %q has no MODEL specified. Add MODEL \"<model_name>\" inside the route body", route.Name),
			nil,
		)
		return
	}

	if len(v.modelNames) > 0 {
		v.checkRouteModelReferences(route)
	}
}

func (v *Validator) checkRouteModelReferences(route *RouteDecl) {
	for _, mr := range route.Models {
		if mr.Model == "" || v.modelNames[mr.Model] {
			v.checkRouteLoRAReference(route, mr)
			continue
		}
		v.addDiag(DiagWarning, mr.Pos,
			fmt.Sprintf("Model %q is not declared in the top-level model catalog", mr.Model),
			v.suggestModel(mr.Model),
		)
	}
}

func (v *Validator) checkRouteLoRAReference(route *RouteDecl, mr *ModelRef) {
	if mr == nil {
		return
	}

	if mr.LoRA == "" || !v.modelNames[mr.Model] {
		return
	}

	loras := v.modelLoRAs[mr.Model]
	switch {
	case len(loras) == 0:
		v.addDiag(DiagWarning, mr.Pos,
			fmt.Sprintf("Model %q declares no LoRA adapters, but route %q references LoRA %q", mr.Model, route.Name, mr.LoRA),
			nil,
		)
	case !loras[mr.LoRA]:
		v.addDiag(DiagWarning, mr.Pos,
			fmt.Sprintf("LoRA %q is not declared for model %q in the top-level model catalog", mr.LoRA, mr.Model),
			v.suggestLoRA(mr.Model, mr.LoRA),
		)
	}
}

func (v *Validator) walkBoolExpr(expr BoolExpr) {
	switch e := expr.(type) {
	case *BoolAnd:
		v.walkBoolExpr(e.Left)
		v.walkBoolExpr(e.Right)
	case *BoolOr:
		v.walkBoolExpr(e.Left)
		v.walkBoolExpr(e.Right)
	case *BoolNot:
		v.walkBoolExpr(e.Expr)
	case *SignalRefExpr:
		if !v.isSignalDefined(e.SignalType, e.SignalName) {
			fix := v.suggestSignal(e.SignalType, e.SignalName)
			v.addDiag(DiagWarning, e.Pos,
				fmt.Sprintf("Signal %s(\"%s\") referenced in WHEN clause is not defined. Add SIGNAL %s %s { ... } as a top-level declaration", e.SignalType, e.SignalName, e.SignalType, e.SignalName),
				fix,
			)
		}
	}
}

func signalReferenceDefined(names map[string]bool, signalType, name string) bool {
	if len(names) == 0 {
		return false
	}
	if names[name] {
		return true
	}
	if signalType == config.SignalTypeComplexity {
		// Complexity references may target derived levels like "math_task:hard".
		if idx := strings.Index(name, ":"); idx > 0 {
			_, exists := names[name[:idx]]
			return exists
		}
	}
	return false
}

func (v *Validator) isSignalDefined(signalType, name string) bool {
	if names, ok := v.signalNames[signalType]; ok {
		return signalReferenceDefined(names, signalType, name)
	}
	return false
}

// isInlinePluginType returns true if the name is a recognized inline plugin type.
func isInlinePluginType(name string) bool {
	return config.IsSupportedDecisionPluginType(name)
}

// ---------- Level 3: Constraint Checks ----------

// constraintRule defines a numeric range check for a named field.
type constraintRule struct {
	field string
	min   *float64
	max   *float64
}

var (
	floatZero    = 0.0
	floatOne     = 1.0
	floatMinPort = 1.0
	floatMaxPort = 65535.0
)

var constraintRules = []constraintRule{
	{field: "threshold", min: &floatZero, max: &floatOne},
	{field: "similarity_threshold", min: &floatZero, max: &floatOne},
	{field: "bm25_threshold", min: &floatZero, max: &floatOne},
	{field: "ngram_threshold", min: &floatZero, max: &floatOne},
	{field: "verification_threshold", min: &floatZero, max: &floatOne},
	{field: "exploration_rate", min: &floatZero, max: &floatOne},
	{field: "min_similarity", min: &floatZero, max: &floatOne},
	{field: "port", min: &floatMinPort, max: &floatMaxPort},
	{field: "fuzzy_threshold", min: &floatZero},
	{field: "ngram_arity", min: &floatOne},
}

func (v *Validator) checkConstraints() {
	// Check signals
	for _, s := range v.prog.Signals {
		v.checkSignalConstraints(s)
	}

	// Check routes
	for _, r := range v.prog.Routes {
		v.checkRouteConstraints(r)
	}
}

func (v *Validator) checkSignalConstraints(s *SignalDecl) {
	context := fmt.Sprintf("SIGNAL %s %s", s.SignalType, s.Name)

	// Check valid signal types
	if !config.IsSupportedSignalType(s.SignalType) {
		v.addDiag(DiagConstraint, s.Pos,
			fmt.Sprintf("Unknown signal type %q in %s. Supported signal types: keyword, embedding, domain, fact_check, user_feedback, preference, language, context, structure, complexity, modality, authz, jailbreak, pii, kb", s.SignalType, context),
			nil,
		)
	}

	// Check field constraints
	v.checkFieldConstraints(s.Fields, s.Pos, context)

	// Signal-type-specific required fields
	switch s.SignalType {
	case "keyword":
		if _, ok := s.Fields["keywords"]; !ok {
			v.addDiag(DiagConstraint, s.Pos,
				fmt.Sprintf("%s: 'keywords' field is recommended", context),
				nil,
			)
		}
	case "embedding":
		if _, ok := s.Fields["threshold"]; !ok {
			v.addDiag(DiagConstraint, s.Pos,
				fmt.Sprintf("%s: 'threshold' field is recommended", context),
				nil,
			)
		}
		if _, ok := s.Fields["candidates"]; !ok {
			v.addDiag(DiagConstraint, s.Pos,
				fmt.Sprintf("%s: 'candidates' field is recommended", context),
				nil,
			)
		}
	case "domain":
		v.checkDomainSignalConstraints(s, context)
	case "structure":
		v.checkStructureSignalConstraints(s)
	}
}

func (v *Validator) checkStructureSignalConstraints(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name

	raw, err := yaml.Marshal(payload)
	if err != nil {
		v.addDiag(DiagConstraint, s.Pos,
			fmt.Sprintf("failed to encode structure signal %q: %v", s.Name, err),
			nil,
		)
		return
	}

	var rule config.StructureRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		v.addDiag(DiagConstraint, s.Pos,
			fmt.Sprintf("failed to decode structure signal %q: %v", s.Name, err),
			nil,
		)
		return
	}
	rule.Name = s.Name
	if err := config.ValidateStructureRuleContract(rule); err != nil {
		v.addDiag(DiagConstraint, s.Pos, err.Error(), nil)
	}
}

func (v *Validator) checkRouteConstraints(r *RouteDecl) {
	context := fmt.Sprintf("ROUTE %s", r.Name)

	// Priority should be >= 0
	if r.Priority < 0 {
		v.addDiag(DiagConstraint, r.Pos,
			fmt.Sprintf("%s: priority must be >= 0, got %d", context, r.Priority),
			&QuickFix{Description: "Set priority to 0", NewText: "0"},
		)
	}

	// Check algorithm constraints
	if r.Algorithm != nil {
		v.checkAlgorithmConstraints(r.Algorithm, context)
	}
}

func (v *Validator) checkAlgorithmConstraints(algo *AlgoSpec, parentContext string) {
	validAlgoTypes := config.SupportedDecisionAlgorithmTypes()
	if !config.IsSupportedDecisionAlgorithmType(algo.AlgoType) {
		similar := suggestSimilar(algo.AlgoType, validAlgoTypes)
		fix := (*QuickFix)(nil)
		if similar != "" {
			fix = &QuickFix{Description: fmt.Sprintf("Change to %q", similar), NewText: similar}
		}
		v.addDiag(DiagConstraint, algo.Pos,
			fmt.Sprintf("%s: unknown algorithm type %q. Supported types: %s", parentContext, algo.AlgoType, strings.Join(validAlgoTypes, ", ")),
			fix,
		)
	}

	if algo.Fields != nil {
		v.checkFieldConstraints(algo.Fields, algo.Pos, parentContext+" ALGORITHM")
	}
}

// checkFieldConstraints recursively checks all numeric field values against the constraint rules.
//
//nolint:gocognit,cyclop // Recursive constraint walking stays centralized for DSL numeric validation.
func (v *Validator) checkFieldConstraints(fields map[string]Value, pos Position, context string) {
	for k, val := range fields {
		for _, rule := range constraintRules {
			if k == rule.field {
				var numVal float64
				switch vt := val.(type) {
				case FloatValue:
					numVal = vt.V
				case IntValue:
					numVal = float64(vt.V)
				default:
					continue
				}
				if rule.min != nil && numVal < *rule.min {
					v.addDiag(DiagConstraint, pos,
						fmt.Sprintf("%s: %s must be >= %v, got %v", context, k, *rule.min, numVal),
						&QuickFix{Description: fmt.Sprintf("Set to %v", *rule.min), NewText: fmt.Sprintf("%v", *rule.min)},
					)
				}
				if rule.max != nil && numVal > *rule.max {
					v.addDiag(DiagConstraint, pos,
						fmt.Sprintf("%s: %s must be <= %v, got %v", context, k, *rule.max, numVal),
						&QuickFix{Description: fmt.Sprintf("Set to %v", *rule.max), NewText: fmt.Sprintf("%v", *rule.max)},
					)
				}
			}
		}

		// Recurse into nested objects
		if ov, ok := val.(ObjectValue); ok {
			v.checkFieldConstraints(ov.Fields, pos, context+"."+k)
		}
		if av, ok := val.(ArrayValue); ok {
			for _, item := range av.Items {
				if ov, ok := item.(ObjectValue); ok {
					v.checkFieldConstraints(ov.Fields, pos, context+"."+k)
				}
			}
		}
	}
}

// ---------- Suggestion Helpers ----------

func (v *Validator) suggestSignal(signalType, name string) *QuickFix {
	names, ok := v.signalNames[signalType]
	if !ok || len(names) == 0 {
		return nil
	}
	candidates := keysOfBool(names)
	closest := suggestSimilar(name, candidates)
	if closest == "" {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", closest),
		NewText:     closest,
	}
}

func (v *Validator) suggestPlugin(name string) *QuickFix {
	candidates := keysOfBool(v.pluginNames)
	closest := suggestSimilar(name, candidates)
	if closest == "" {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", closest),
		NewText:     closest,
	}
}

func (v *Validator) suggestModel(name string) *QuickFix {
	candidates := keysOfBool(v.modelNames)
	closest := suggestSimilar(name, candidates)
	if closest == "" {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", closest),
		NewText:     closest,
	}
}

func (v *Validator) suggestLoRA(modelName, loraName string) *QuickFix {
	candidates := keysOfBool(v.modelLoRAs[modelName])
	closest := suggestSimilar(loraName, candidates)
	if closest == "" {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", closest),
		NewText:     closest,
	}
}

func collectModelLoRANames(fields map[string]Value) map[string]bool {
	if fields == nil {
		return nil
	}
	raw, ok := fields["loras"]
	if !ok {
		return nil
	}
	av, ok := raw.(ArrayValue)
	if !ok {
		return nil
	}
	loras := make(map[string]bool)
	for _, item := range av.Items {
		ov, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		nameValue, ok := ov.Fields["name"].(StringValue)
		if !ok || nameValue.V == "" {
			continue
		}
		loras[nameValue.V] = true
	}
	return loras
}

// suggestSimilar finds the closest match using Levenshtein distance.
func suggestSimilar(target string, candidates []string) string {
	if len(candidates) == 0 {
		return ""
	}
	target = strings.ToLower(target)
	bestDist := len(target) + 1
	bestMatch := ""
	for _, c := range candidates {
		d := levenshtein(target, strings.ToLower(c))
		if d < bestDist && d <= len(target)/2+1 {
			bestDist = d
			bestMatch = c
		}
	}
	return bestMatch
}

// levenshtein computes the edit distance between two strings.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min3(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// ---------- Diagnostic Helpers ----------

func (v *Validator) addDiag(level DiagLevel, pos Position, message string, fix *QuickFix) {
	v.diagnostics = append(v.diagnostics, Diagnostic{
		Level:   level,
		Message: message,
		Pos:     pos,
		Fix:     fix,
	})
}
