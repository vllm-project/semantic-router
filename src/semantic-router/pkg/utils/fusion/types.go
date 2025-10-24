package fusion

// ActionType represents the type of action to take when a rule matches
type ActionType string

const (
	ActionBlock         ActionType = "block"
	ActionRoute         ActionType = "route"
	ActionBoostCategory ActionType = "boost_category"
	ActionFallthrough   ActionType = "fallthrough"
)

// Signal represents a single signal result from a provider
type Signal struct {
	// Provider is the source of the signal (e.g., "keyword", "regex", "similarity", "bert")
	Provider string
	// Name is the specific signal identifier (e.g., rule name, pattern name)
	Name string
	// Matched indicates if the signal matched
	Matched bool
	// Score is an optional numeric value (e.g., similarity score, confidence)
	Score float64
	// Value is an optional string value (e.g., category name)
	Value string
}

// SignalContext holds all available signals for evaluation
type SignalContext struct {
	Signals map[string]Signal
}

// Rule represents a fusion policy rule
type Rule struct {
	// Name is a unique identifier for the rule
	Name string
	// Condition is the boolean expression to evaluate
	Condition string
	// Action specifies what to do when the condition matches
	Action ActionType
	// Priority determines evaluation order (higher = evaluated first)
	Priority int
	// Models is the list of target models for route actions
	Models []string
	// Category is the target category for boost actions
	Category string
	// BoostWeight is the multiplier for boost actions
	BoostWeight float64
	// Message is the response message for block actions
	Message string
}

// Policy represents a complete fusion policy with multiple rules
type Policy struct {
	// Rules are evaluated in priority order
	Rules []Rule
}

// EvaluationResult represents the result of evaluating a policy
type EvaluationResult struct {
	// Matched indicates if any rule matched
	Matched bool
	// MatchedRule is the name of the first matching rule (if any)
	MatchedRule string
	// Action is the action to take
	Action ActionType
	// Models is the list of candidate models for route actions
	Models []string
	// Category is the target category for boost actions
	Category string
	// BoostWeight is the multiplier for boost actions
	BoostWeight float64
	// Message is the response message for block actions
	Message string
}
