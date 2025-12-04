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

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestFusion(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Fusion Suite")
}

var _ = Describe("Expression Parser", func() {
	var parser *Parser

	BeforeEach(func() {
		parser = NewParser()
	})

	Context("Simple expressions", func() {
		It("should parse AND expression", func() {
			ast, err := parser.Parse("keyword.k8s.matched && bert.category == \"computer_science\"")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeAnd))
		})

		It("should parse OR expression", func() {
			ast, err := parser.Parse("keyword.k8s.matched || keyword.kubernetes.matched")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeOr))
		})

		It("should parse NOT expression", func() {
			ast, err := parser.Parse("!regex.ssn.matched")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeNot))
		})

		It("should parse boolean literal true", func() {
			ast, err := parser.Parse("true")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeLiteral))
			Expect(ast.Value).To(Equal(true))
		})

		It("should parse boolean literal false", func() {
			ast, err := parser.Parse("false")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeLiteral))
			Expect(ast.Value).To(Equal(false))
		})

		It("should parse identifier", func() {
			ast, err := parser.Parse("keyword.k8s.matched")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeIdentifier))
			Expect(ast.Value).To(Equal("keyword.k8s.matched"))
		})
	})

	Context("Comparison expressions", func() {
		It("should parse equals comparison", func() {
			ast, err := parser.Parse("bert.category == \"math\"")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal("=="))
		})

		It("should parse not equals comparison", func() {
			ast, err := parser.Parse("bert.category != \"casual\"")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal("!="))
		})

		It("should parse greater than comparison", func() {
			ast, err := parser.Parse("similarity.reasoning.score > 0.75")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal(">"))
		})

		It("should parse less than comparison", func() {
			ast, err := parser.Parse("bert.confidence < 0.5")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal("<"))
		})

		It("should parse greater than or equal comparison", func() {
			ast, err := parser.Parse("similarity.reasoning.score >= 0.8")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal(">="))
		})

		It("should parse less than or equal comparison", func() {
			ast, err := parser.Parse("bert.confidence <= 0.9")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeComparison))
			Expect(ast.Operator).To(Equal("<="))
		})
	})

	Context("Complex expressions with parentheses", func() {
		It("should parse nested parentheses", func() {
			ast, err := parser.Parse("(keyword.k8s.matched || keyword.kubernetes.matched) && bert.category == \"computer_science\"")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeAnd))
			Expect(ast.Left.Type).To(Equal(NodeOr))
		})

		It("should parse deeply nested expression", func() {
			ast, err := parser.Parse("((keyword.k8s.matched && bert.confidence > 0.9) || regex.cve.matched) && !regex.ssn.matched")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			Expect(ast.Type).To(Equal(NodeAnd))
		})

		It("should respect operator precedence", func() {
			// AND has higher precedence than OR
			ast, err := parser.Parse("keyword.a.matched || keyword.b.matched && keyword.c.matched")
			Expect(err).ToNot(HaveOccurred())
			Expect(ast).ToNot(BeNil())
			// Should parse as: keyword.a.matched || (keyword.b.matched && keyword.c.matched)
			Expect(ast.Type).To(Equal(NodeOr))
			Expect(ast.Right.Type).To(Equal(NodeAnd))
		})
	})

	Context("Error handling", func() {
		It("should error on unterminated string", func() {
			_, err := parser.Parse("bert.category == \"computer_science")
			Expect(err).To(HaveOccurred())
		})

		It("should error on unexpected token", func() {
			_, err := parser.Parse("keyword.k8s.matched &&")
			Expect(err).To(HaveOccurred())
		})

		It("should error on mismatched parentheses", func() {
			_, err := parser.Parse("(keyword.k8s.matched")
			Expect(err).To(HaveOccurred())
		})

		It("should error on invalid number", func() {
			_, err := parser.Parse("similarity.score > 0.75.5")
			Expect(err).To(HaveOccurred())
		})
	})
})

var _ = Describe("Expression Evaluator", func() {
	var ctx *SignalContext

	BeforeEach(func() {
		ctx = NewSignalContext()
		ctx.KeywordMatches["k8s"] = true
		ctx.KeywordMatches["database"] = false
		ctx.RegexMatches["ssn"] = false
		ctx.RegexMatches["cve"] = true
		ctx.SimilarityScores["reasoning"] = 0.82
		ctx.SimilarityScores["sentiment"] = 0.23
		ctx.BERTCategory = "computer_science"
		ctx.BERTConfidence = 0.95
	})

	Context("Boolean operations", func() {
		It("should evaluate AND expression (both true)", func() {
			parser := NewParser()
			ast, err := parser.Parse("keyword.k8s.matched && regex.cve.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate AND expression (one false)", func() {
			parser := NewParser()
			ast, err := parser.Parse("keyword.k8s.matched && keyword.database.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse())
		})

		It("should evaluate OR expression (one true)", func() {
			parser := NewParser()
			ast, err := parser.Parse("keyword.database.matched || keyword.k8s.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate OR expression (both false)", func() {
			parser := NewParser()
			ast, err := parser.Parse("keyword.database.matched || regex.ssn.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse())
		})

		It("should evaluate NOT expression (true)", func() {
			parser := NewParser()
			ast, err := parser.Parse("!regex.ssn.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate NOT expression (false)", func() {
			parser := NewParser()
			ast, err := parser.Parse("!keyword.k8s.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse())
		})
	})

	Context("Comparison operations", func() {
		It("should evaluate string equals (true)", func() {
			parser := NewParser()
			ast, err := parser.Parse("bert.category == \"computer_science\"")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate string equals (false)", func() {
			parser := NewParser()
			ast, err := parser.Parse("bert.category == \"math\"")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse())
		})

		It("should evaluate string not equals", func() {
			parser := NewParser()
			ast, err := parser.Parse("bert.category != \"math\"")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate numeric greater than (true)", func() {
			parser := NewParser()
			ast, err := parser.Parse("similarity.reasoning.score > 0.75")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate numeric greater than (false)", func() {
			parser := NewParser()
			ast, err := parser.Parse("similarity.sentiment.score > 0.5")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse())
		})

		It("should evaluate numeric less than", func() {
			parser := NewParser()
			ast, err := parser.Parse("similarity.sentiment.score < 0.5")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate numeric greater than or equal", func() {
			parser := NewParser()
			ast, err := parser.Parse("bert.confidence >= 0.95")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate numeric less than or equal", func() {
			parser := NewParser()
			ast, err := parser.Parse("similarity.sentiment.score <= 0.25")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})
	})

	Context("Complex expressions", func() {
		It("should evaluate complex AND/OR expression", func() {
			parser := NewParser()
			ast, err := parser.Parse("(keyword.k8s.matched || keyword.database.matched) && bert.category == \"computer_science\"")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate expression with NOT and comparison", func() {
			parser := NewParser()
			ast, err := parser.Parse("!regex.ssn.matched && similarity.reasoning.score > 0.7")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})

		It("should evaluate deeply nested expression", func() {
			parser := NewParser()
			ast, err := parser.Parse("((keyword.k8s.matched && bert.confidence > 0.9) || regex.cve.matched) && !regex.ssn.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeTrue())
		})
	})

	Context("Signal lookup", func() {
		It("should handle non-existent keyword rule", func() {
			parser := NewParser()
			ast, err := parser.Parse("keyword.nonexistent.matched")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse()) // Should default to false
		})

		It("should handle non-existent similarity concept", func() {
			parser := NewParser()
			ast, err := parser.Parse("similarity.nonexistent.score > 0.5")
			Expect(err).ToNot(HaveOccurred())

			evaluator := NewEvaluator(ast, ctx)
			result, err := evaluator.Evaluate()
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(BeFalse()) // 0.0 > 0.5 = false
		})
	})
})

var _ = Describe("Fusion Engine", func() {
	Context("Policy evaluation", func() {
		It("should evaluate policies in priority order", func() {
			policies := []FusionPolicy{
				{
					Name:      "low-priority",
					Priority:  50,
					Condition: "true",
					Action:    ActionRoute,
					Models:    []string{"model-low"},
				},
				{
					Name:      "high-priority",
					Priority:  150,
					Condition: "keyword.k8s.matched",
					Action:    ActionRoute,
					Models:    []string{"model-high"},
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()
			ctx.KeywordMatches["k8s"] = true

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.MatchedPolicy).To(Equal("high-priority"))
			Expect(result.Models).To(Equal([]string{"model-high"}))
		})

		It("should short-circuit on first match", func() {
			policies := []FusionPolicy{
				{
					Name:      "first",
					Priority:  200,
					Condition: "regex.ssn.matched",
					Action:    ActionBlock,
					Message:   "SSN detected",
				},
				{
					Name:      "second",
					Priority:  150,
					Condition: "keyword.k8s.matched",
					Action:    ActionRoute,
					Models:    []string{"k8s-model"},
				},
				{
					Name:      "third",
					Priority:  100,
					Condition: "true",
					Action:    ActionFallthrough,
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()
			ctx.RegexMatches["ssn"] = false
			ctx.KeywordMatches["k8s"] = true

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			// Should match second policy (priority 150) and short-circuit
			Expect(result.MatchedPolicy).To(Equal("second"))
			Expect(result.Action).To(Equal(ActionRoute))
		})

		It("should return error when no policy matches", func() {
			policies := []FusionPolicy{
				{
					Name:      "only-policy",
					Priority:  100,
					Condition: "keyword.nonexistent.matched",
					Action:    ActionRoute,
					Models:    []string{"model"},
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()

			result, err := engine.Evaluate(ctx)
			Expect(err).To(HaveOccurred())
			Expect(result).To(BeNil())
		})
	})

	Context("Action types", func() {
		It("should create block result", func() {
			policies := []FusionPolicy{
				{
					Name:      "block-ssn",
					Priority:  200,
					Condition: "regex.ssn.matched",
					Action:    ActionBlock,
					Message:   "SSN pattern detected",
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()
			ctx.RegexMatches["ssn"] = true

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Action).To(Equal(ActionBlock))
			Expect(result.Message).To(Equal("SSN pattern detected"))
		})

		It("should create route result", func() {
			policies := []FusionPolicy{
				{
					Name:      "route-k8s",
					Priority:  150,
					Condition: "keyword.k8s.matched",
					Action:    ActionRoute,
					Models:    []string{"k8s-expert", "devops-model"},
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()
			ctx.KeywordMatches["k8s"] = true

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Action).To(Equal(ActionRoute))
			Expect(result.Models).To(Equal([]string{"k8s-expert", "devops-model"}))
		})

		It("should create boost result", func() {
			policies := []FusionPolicy{
				{
					Name:        "boost-reasoning",
					Priority:    100,
					Condition:   "similarity.reasoning.score > 0.75",
					Action:      ActionBoost,
					Category:    "reasoning",
					BoostWeight: 1.5,
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()
			ctx.SimilarityScores["reasoning"] = 0.85

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Action).To(Equal(ActionBoost))
			Expect(result.Category).To(Equal("reasoning"))
			Expect(result.BoostWeight).To(Equal(1.5))
		})

		It("should create fallthrough result", func() {
			policies := []FusionPolicy{
				{
					Name:      "default",
					Priority:  0,
					Condition: "true",
					Action:    ActionFallthrough,
				},
			}

			engine, err := NewFusionEngine(policies)
			Expect(err).ToNot(HaveOccurred())

			ctx := NewSignalContext()

			result, err := engine.Evaluate(ctx)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Action).To(Equal(ActionFallthrough))
		})
	})

	Context("Policy validation", func() {
		It("should validate correct block policy", func() {
			policy := FusionPolicy{
				Name:      "block-test",
				Priority:  200,
				Condition: "regex.ssn.matched",
				Action:    ActionBlock,
				Message:   "SSN detected",
			}

			err := ValidatePolicy(&policy)
			Expect(err).ToNot(HaveOccurred())
		})

		It("should error on block policy without message", func() {
			policy := FusionPolicy{
				Name:      "block-test",
				Priority:  200,
				Condition: "regex.ssn.matched",
				Action:    ActionBlock,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on route policy without models", func() {
			policy := FusionPolicy{
				Name:      "route-test",
				Priority:  150,
				Condition: "keyword.k8s.matched",
				Action:    ActionRoute,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on boost policy without category", func() {
			policy := FusionPolicy{
				Name:        "boost-test",
				Priority:    100,
				Condition:   "similarity.score > 0.5",
				Action:      ActionBoost,
				BoostWeight: 1.5,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on boost policy with invalid boost weight", func() {
			policy := FusionPolicy{
				Name:        "boost-test",
				Priority:    100,
				Condition:   "similarity.score > 0.5",
				Action:      ActionBoost,
				Category:    "reasoning",
				BoostWeight: 0,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on invalid condition", func() {
			policy := FusionPolicy{
				Name:      "invalid-test",
				Priority:  100,
				Condition: "invalid syntax &&",
				Action:    ActionFallthrough,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on empty policy name", func() {
			policy := FusionPolicy{
				Priority:  100,
				Condition: "true",
				Action:    ActionFallthrough,
			}

			err := ValidatePolicy(&policy)
			Expect(err).To(HaveOccurred())
		})

		It("should error on duplicate policy names", func() {
			policies := []FusionPolicy{
				{
					Name:      "duplicate",
					Priority:  100,
					Condition: "true",
					Action:    ActionFallthrough,
				},
				{
					Name:      "duplicate",
					Priority:  50,
					Condition: "false",
					Action:    ActionFallthrough,
				},
			}

			err := ValidatePolicies(policies)
			Expect(err).To(HaveOccurred())
		})
	})
})
