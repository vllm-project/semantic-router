package dsl

import (
	"fmt"
	"sort"
	"strings"
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
	signalNames  map[string]map[string]bool // signalType → {name → true}
	pluginNames  map[string]bool            // template name → true
	backendNames map[string]map[string]bool // backendType → {name → true}
}

// SymbolInfo represents a named symbol extracted from the AST.
type SymbolInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// SymbolTable contains all declared symbols from a DSL source, for use by
// editor features such as context-aware completion.
type SymbolTable struct {
	Signals  []SymbolInfo `json:"signals"`
	Models   []string     `json:"models"`
	Plugins  []string     `json:"plugins"`
	Backends []SymbolInfo `json:"backends"`
	Routes   []string     `json:"routes"`
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
		prog:         prog,
		signalNames:  make(map[string]map[string]bool),
		pluginNames:  make(map[string]bool),
		backendNames: make(map[string]map[string]bool),
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

	// Extract symbol table for editor completions
	symbols := v.extractSymbolTable()

	return v.diagnostics, symbols, parseErrors
}

// ValidateAST performs Level 2 and Level 3 validation on an existing AST.
func ValidateAST(prog *Program) []Diagnostic {
	v := &Validator{
		prog:         prog,
		signalNames:  make(map[string]map[string]bool),
		pluginNames:  make(map[string]bool),
		backendNames: make(map[string]map[string]bool),
	}
	v.buildSymbolTable()
	v.checkReferences()
	v.checkConstraints()
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

	for _, p := range v.prog.Plugins {
		v.pluginNames[p.Name] = true
	}

	for _, b := range v.prog.Backends {
		if v.backendNames[b.BackendType] == nil {
			v.backendNames[b.BackendType] = make(map[string]bool)
		}
		v.backendNames[b.BackendType][b.Name] = true
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

	// Backends
	for bType, names := range v.backendNames {
		for name := range names {
			st.Backends = append(st.Backends, SymbolInfo{Name: name, Type: bType})
		}
	}
	sort.Slice(st.Backends, func(i, j int) bool {
		if st.Backends[i].Type != st.Backends[j].Type {
			return st.Backends[i].Type < st.Backends[j].Type
		}
		return st.Backends[i].Name < st.Backends[j].Name
	})

	// Models: collect unique model names from all ROUTE declarations
	modelSet := make(map[string]bool)
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
		// Check WHEN expression signal references
		if route.When != nil {
			v.walkBoolExpr(route.When)
		}

		// Check PLUGIN references
		for _, pr := range route.Plugins {
			if !v.pluginNames[pr.Name] && !isInlinePluginType(pr.Name) {
				fix := v.suggestPlugin(pr.Name)
				v.addDiag(DiagWarning, pr.Pos,
					fmt.Sprintf("Plugin %q is not defined as a template and is not a recognized inline plugin type", pr.Name),
					fix,
				)
			}
		}

		// Check MODEL — at least one model should be specified
		if len(route.Models) == 0 {
			v.addDiag(DiagWarning, route.Pos,
				fmt.Sprintf("Route %q has no MODEL specified", route.Name),
				nil,
			)
		}
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
				fmt.Sprintf("Signal '%s(\"%s\")' is not defined", e.SignalType, e.SignalName),
				fix,
			)
		}
	}
}

func (v *Validator) isSignalDefined(signalType, name string) bool {
	if names, ok := v.signalNames[signalType]; ok {
		if names[name] {
			return true
		}
		// Support sub-level references like complexity("math_problem:hard")
		// where "math_problem" is the defined signal and "hard" is a sub-level.
		if idx := strings.Index(name, ":"); idx > 0 {
			baseName := name[:idx]
			return names[baseName]
		}
	}
	return false
}

// isInlinePluginType returns true if the name is a recognized inline plugin type.
func isInlinePluginType(name string) bool {
	switch name {
	case "semantic_cache", "memory", "system_prompt",
		"header_mutation", "hallucination", "router_replay", "rag", "image_gen",
		"fast_response":
		return true
	default:
		return false
	}
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

	// Check backends
	for _, b := range v.prog.Backends {
		v.checkFieldConstraints(b.Fields, b.Pos, fmt.Sprintf("BACKEND %s %s", b.BackendType, b.Name))
	}

	// Check global
	if v.prog.Global != nil {
		v.checkGlobalConstraints(v.prog.Global)
	}
}

func (v *Validator) checkSignalConstraints(s *SignalDecl) {
	context := fmt.Sprintf("SIGNAL %s %s", s.SignalType, s.Name)

	// Check valid signal types
	validSignalTypes := map[string]bool{
		"keyword": true, "embedding": true, "domain": true, "fact_check": true,
		"user_feedback": true, "preference": true, "language": true,
		"context": true, "complexity": true, "modality": true, "authz": true,
		"jailbreak": true, "pii": true,
	}
	if !validSignalTypes[s.SignalType] {
		v.addDiag(DiagConstraint, s.Pos,
			fmt.Sprintf("Unknown signal type %q in %s", s.SignalType, context),
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
	validAlgoTypes := map[string]bool{
		"confidence": true, "ratings": true, "remom": true, "static": true,
		"elo": true, "router_dc": true, "automix": true, "hybrid": true,
		"rl_driven": true, "gmtrouter": true, "latency_aware": true,
		"knn": true, "kmeans": true, "svm": true,
	}
	if !validAlgoTypes[algo.AlgoType] {
		similar := suggestSimilar(algo.AlgoType, keysOf(validAlgoTypes))
		fix := (*QuickFix)(nil)
		if similar != "" {
			fix = &QuickFix{Description: fmt.Sprintf("Change to %q", similar), NewText: similar}
		}
		v.addDiag(DiagConstraint, algo.Pos,
			fmt.Sprintf("%s: unknown algorithm type %q", parentContext, algo.AlgoType),
			fix,
		)
	}

	if algo.Fields != nil {
		v.checkFieldConstraints(algo.Fields, algo.Pos, parentContext+" ALGORITHM")
	}
}

func (v *Validator) checkGlobalConstraints(g *GlobalDecl) {
	// Check prompt_guard threshold
	if obj, ok := g.Fields["prompt_guard"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			v.checkFieldConstraints(ov.Fields, g.Pos, "GLOBAL prompt_guard")
		}
	}

	// Check observability nested fields
	if obj, ok := g.Fields["observability"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			if tracing, ok := ov.Fields["tracing"]; ok {
				if tv, ok := tracing.(ObjectValue); ok {
					if sampling, ok := tv.Fields["sampling"]; ok {
						if sv, ok := sampling.(ObjectValue); ok {
							v.checkFieldConstraints(sv.Fields, g.Pos, "GLOBAL observability.tracing.sampling")
						}
					}
				}
			}
		}
	}
}

// checkFieldConstraints recursively checks all numeric field values against the constraint rules.
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

// ---------- Utility ----------

func keysOfBool(m map[string]bool) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	sort.Strings(result)
	return result
}

func keysOf(m map[string]bool) []string {
	return keysOfBool(m)
}
