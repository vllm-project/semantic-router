/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fusion

// Action represents the type of action a fusion policy can take
type Action string

const (
	// ActionBlock blocks the request with a custom message
	ActionBlock Action = "block"
	// ActionRoute routes the request to specific model candidates
	ActionRoute Action = "route"
	// ActionBoost boosts the weight of a specific category
	ActionBoost Action = "boost_category"
	// ActionFallthrough falls through to the existing decision engine
	ActionFallthrough Action = "fallthrough"
)

// FusionPolicy represents a single routing policy with a boolean expression condition
type FusionPolicy struct {
	// Name is the unique identifier for this policy
	Name string `yaml:"name" json:"name"`

	// Description provides information about what this policy does
	Description string `yaml:"description,omitempty" json:"description,omitempty"`

	// Priority determines evaluation order (higher = evaluated first)
	// Recommended ranges:
	//   200: Safety blocks (SSN, credit cards, PII)
	//   150: High-confidence routing (keyword + BERT matches)
	//   100: Category boosting (embedding similarity)
	//   50:  Consensus requirements (multiple signals)
	//   0:   Default fallthrough
	Priority int `yaml:"priority" json:"priority"`

	// Condition is a boolean expression that references signals
	// Examples:
	//   "keyword.k8s.matched && bert.category == 'computer_science'"
	//   "regex.ssn.matched"
	//   "similarity.reasoning.score > 0.75"
	//   "(keyword.security.matched || regex.cve.matched) && bert.confidence > 0.8"
	Condition string `yaml:"condition" json:"condition"`

	// Action specifies what to do when the condition matches
	Action Action `yaml:"action" json:"action"`

	// Models specifies target models (used with ActionRoute)
	Models []string `yaml:"models,omitempty" json:"models,omitempty"`

	// Category specifies the category to boost (used with ActionBoost)
	Category string `yaml:"category,omitempty" json:"category,omitempty"`

	// BoostWeight is the multiplier for category boosting (used with ActionBoost)
	// Example: 1.5 means boost category weight by 50%
	BoostWeight float64 `yaml:"boost_weight,omitempty" json:"boost_weight,omitempty"`

	// Message is the error message to return (used with ActionBlock)
	Message string `yaml:"message,omitempty" json:"message,omitempty"`
}

// SignalContext holds all signal values collected from various providers
type SignalContext struct {
	// KeywordMatches maps keyword rule names to match status
	// Example: {"k8s": true, "database": false}
	KeywordMatches map[string]bool

	// RegexMatches maps regex pattern names to match status
	// Example: {"ssn": false, "cve": true}
	RegexMatches map[string]bool

	// SimilarityScores maps similarity concept names to similarity scores [0.0, 1.0]
	// Example: {"reasoning": 0.82, "sentiment": 0.23}
	SimilarityScores map[string]float64

	// BERTCategory is the category from BERT classification
	// Example: "computer_science", "math", "biology"
	BERTCategory string

	// BERTConfidence is the confidence score from BERT classification [0.0, 1.0]
	BERTConfidence float64
}

// NewSignalContext creates a new SignalContext with initialized maps
func NewSignalContext() *SignalContext {
	return &SignalContext{
		KeywordMatches:   make(map[string]bool),
		RegexMatches:     make(map[string]bool),
		SimilarityScores: make(map[string]float64),
	}
}

// FusionResult represents the output of fusion policy evaluation
type FusionResult struct {
	// Action is the action to take
	Action Action

	// Models are the target models (for ActionRoute)
	Models []string

	// Category is the category to boost (for ActionBoost)
	Category string

	// BoostWeight is the boost multiplier (for ActionBoost)
	BoostWeight float64

	// Message is the block message (for ActionBlock)
	Message string

	// MatchedPolicy is the name of the policy that matched
	MatchedPolicy string

	// Priority is the priority of the matched policy
	Priority int
}

// TokenType represents the type of token in the expression
type TokenType int

const (
	// Token types for lexer
	TokenEOF TokenType = iota
	TokenError

	// Literals
	TokenIdentifier // keyword.k8s.matched, bert.category
	TokenString     // "computer_science"
	TokenNumber     // 0.75
	TokenTrue       // true
	TokenFalse      // false

	// Operators
	TokenAnd // &&
	TokenOr  // ||
	TokenNot // !

	// Comparison operators
	TokenEQ // ==
	TokenNE // !=
	TokenGT // >
	TokenLT // <
	TokenGE // >=
	TokenLE // <=

	// Delimiters
	TokenLParen // (
	TokenRParen // )
)

// Token represents a lexical token
type Token struct {
	Type  TokenType
	Value string
}

// NodeType represents the type of AST node
type NodeType int

const (
	// Node types for AST
	NodeAnd        NodeType = iota // &&
	NodeOr                          // ||
	NodeNot                         // !
	NodeComparison                  // ==, !=, >, <, >=, <=
	NodeIdentifier                  // signal path
	NodeLiteral                     // string, number, boolean
)

// ASTNode represents a node in the Abstract Syntax Tree
type ASTNode struct {
	// Type of this node
	Type NodeType

	// Operator (for comparison nodes)
	Operator string

	// Left and right children (for binary operators)
	Left  *ASTNode
	Right *ASTNode

	// Value (for literals and identifiers)
	Value interface{}
}
