package fusion

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestFusion(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Signal Fusion Suite")
}

var _ = Describe("Signal Fusion Engine", func() {
	Describe("Expression Evaluator", func() {
		var (
			context   *SignalContext
			evaluator *ExpressionEvaluator
		)

		BeforeEach(func() {
			context = NewSignalContext()
			evaluator = NewExpressionEvaluator(context)
		})

		Context("when evaluating simple signal references", func() {
			It("should return true for matched signals", func() {
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})

				result, err := evaluator.Evaluate("keyword.kubernetes.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())
			})

			It("should return false for unmatched signals", func() {
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  false,
				})

				result, err := evaluator.Evaluate("keyword.kubernetes.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeFalse())
			})

			It("should return false for non-existent signals", func() {
				result, err := evaluator.Evaluate("keyword.nonexistent.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeFalse())
			})
		})

		Context("when evaluating boolean operators", func() {
			BeforeEach(func() {
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "security",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "docker",
					Matched:  false,
				})
			})

			It("should evaluate AND correctly", func() {
				result, err := evaluator.Evaluate("keyword.kubernetes.matched && keyword.security.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("keyword.kubernetes.matched && keyword.docker.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeFalse())
			})

			It("should evaluate OR correctly", func() {
				result, err := evaluator.Evaluate("keyword.kubernetes.matched || keyword.docker.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("keyword.docker.matched || keyword.nonexistent.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeFalse())
			})

			It("should evaluate NOT correctly", func() {
				result, err := evaluator.Evaluate("!keyword.docker.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("!keyword.kubernetes.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeFalse())
			})

			It("should handle complex boolean expressions", func() {
				result, err := evaluator.Evaluate("keyword.kubernetes.matched && (keyword.security.matched || keyword.docker.matched)")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("(keyword.kubernetes.matched || keyword.docker.matched) && keyword.security.matched")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())
			})
		})

		Context("when evaluating comparisons", func() {
			BeforeEach(func() {
				context.AddSignal(Signal{
					Provider: "similarity",
					Name:     "reasoning",
					Score:    0.85,
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "bert",
					Name:     "category",
					Value:    "computer science",
					Matched:  true,
				})
			})

			It("should evaluate numeric comparisons", func() {
				result, err := evaluator.Evaluate("similarity.reasoning.score > 0.75")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score >= 0.85")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score < 0.9")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score <= 0.85")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score == 0.85")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score != 0.75")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())
			})

			It("should evaluate string comparisons", func() {
				result, err := evaluator.Evaluate("bert.category.value == 'computer science'")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("bert.category.value != 'biology'")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())
			})

			It("should combine comparisons with boolean operators", func() {
				result, err := evaluator.Evaluate("similarity.reasoning.score > 0.75 && bert.category.value == 'computer science'")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())

				result, err = evaluator.Evaluate("similarity.reasoning.score > 0.9 || bert.category.value == 'computer science'")
				Expect(err).ToNot(HaveOccurred())
				Expect(result).To(BeTrue())
			})
		})

		Context("when handling edge cases", func() {
			It("should return error for empty expressions", func() {
				_, err := evaluator.Evaluate("")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("empty expression"))
			})

			It("should return error for invalid expressions", func() {
				_, err := evaluator.Evaluate("invalid expression &&")
				Expect(err).To(HaveOccurred())
			})
		})
	})

	Describe("Policy Engine", func() {
		var (
			context *SignalContext
		)

		BeforeEach(func() {
			context = NewSignalContext()
		})

		Context("when evaluating priority-based rules", func() {
			It("should evaluate rules in priority order", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "low-priority",
							Condition: "keyword.test.matched",
							Action:    ActionRoute,
							Priority:  10,
							Models:    []string{"model-a"},
						},
						{
							Name:      "high-priority",
							Condition: "keyword.test.matched",
							Action:    ActionBlock,
							Priority:  100,
							Message:   "Blocked by high priority rule",
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "test",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.MatchedRule).To(Equal("high-priority"))
				Expect(result.Action).To(Equal(ActionBlock))
				Expect(result.Message).To(Equal("Blocked by high priority rule"))
			})

			It("should sort rules by priority descending", func() {
				policy := &Policy{
					Rules: []Rule{
						{Name: "rule1", Condition: "keyword.a.matched", Priority: 50},
						{Name: "rule2", Condition: "keyword.b.matched", Priority: 200},
						{Name: "rule3", Condition: "keyword.c.matched", Priority: 100},
					},
				}

				engine := NewEngine(policy)

				// Verify rules are sorted by priority
				Expect(engine.policy.Rules[0].Name).To(Equal("rule2"))
				Expect(engine.policy.Rules[0].Priority).To(Equal(200))
				Expect(engine.policy.Rules[1].Name).To(Equal("rule3"))
				Expect(engine.policy.Rules[1].Priority).To(Equal(100))
				Expect(engine.policy.Rules[2].Name).To(Equal("rule1"))
				Expect(engine.policy.Rules[2].Priority).To(Equal(50))
			})
		})

		Context("when using short-circuit evaluation", func() {
			It("should return first matching rule", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "first-match",
							Condition: "keyword.kubernetes.matched",
							Action:    ActionRoute,
							Priority:  100,
							Models:    []string{"k8s-expert"},
						},
						{
							Name:      "second-match",
							Condition: "keyword.kubernetes.matched",
							Action:    ActionBlock,
							Priority:  50,
							Message:   "Should not reach here",
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.MatchedRule).To(Equal("first-match"))
				Expect(result.Action).To(Equal(ActionRoute))
				Expect(result.Models).To(Equal([]string{"k8s-expert"}))
			})

			It("should skip non-matching rules", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "no-match",
							Condition: "keyword.docker.matched",
							Action:    ActionBlock,
							Priority:  200,
						},
						{
							Name:      "match",
							Condition: "keyword.kubernetes.matched",
							Action:    ActionRoute,
							Priority:  100,
							Models:    []string{"k8s-expert"},
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "docker",
					Matched:  false,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.MatchedRule).To(Equal("match"))
			})
		})

		Context("when testing different action types", func() {
			It("should handle block actions", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "block-rule",
							Condition: "regex.ssn.matched",
							Action:    ActionBlock,
							Priority:  200,
							Message:   "SSN detected",
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "regex",
					Name:     "ssn",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.Action).To(Equal(ActionBlock))
				Expect(result.Message).To(Equal("SSN detected"))
			})

			It("should handle route actions", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "route-rule",
							Condition: "keyword.kubernetes.matched && keyword.security.matched",
							Action:    ActionRoute,
							Priority:  150,
							Models:    []string{"k8s-security-expert", "devops-model"},
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "security",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.Action).To(Equal(ActionRoute))
				Expect(result.Models).To(Equal([]string{"k8s-security-expert", "devops-model"}))
			})

			It("should handle boost_category actions", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:        "boost-rule",
							Condition:   "similarity.reasoning.score > 0.75",
							Action:      ActionBoostCategory,
							Priority:    100,
							Category:    "reasoning",
							BoostWeight: 1.5,
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "similarity",
					Name:     "reasoning",
					Score:    0.85,
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.Action).To(Equal(ActionBoostCategory))
				Expect(result.Category).To(Equal("reasoning"))
				Expect(result.BoostWeight).To(Equal(1.5))
			})

			It("should handle fallthrough when no rules match", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "no-match",
							Condition: "keyword.docker.matched",
							Action:    ActionBlock,
							Priority:  100,
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "docker",
					Matched:  false,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeFalse())
				Expect(result.Action).To(Equal(ActionFallthrough))
			})
		})

		Context("when handling complex real-world scenarios", func() {
			It("should handle multi-signal consensus requirements", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "consensus-route",
							Condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.8 && bert.category.value == 'computer science'",
							Action:    ActionRoute,
							Priority:  50,
							Models:    []string{"k8s-expert"},
						},
					},
				}

				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "similarity",
					Name:     "infrastructure",
					Score:    0.85,
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "bert",
					Name:     "category",
					Value:    "computer science",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.Action).To(Equal(ActionRoute))
				Expect(result.Models).To(Equal([]string{"k8s-expert"}))
			})

			It("should prioritize safety blocks over routing", func() {
				policy := &Policy{
					Rules: []Rule{
						{
							Name:      "safety-block",
							Condition: "regex.ssn.matched || regex.credit-card.matched",
							Action:    ActionBlock,
							Priority:  200,
							Message:   "PII detected",
						},
						{
							Name:      "route-to-model",
							Condition: "keyword.kubernetes.matched",
							Action:    ActionRoute,
							Priority:  150,
							Models:    []string{"k8s-expert"},
						},
					},
				}

				// Both rules would match
				context.AddSignal(Signal{
					Provider: "regex",
					Name:     "ssn",
					Matched:  true,
				})
				context.AddSignal(Signal{
					Provider: "keyword",
					Name:     "kubernetes",
					Matched:  true,
				})

				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				// Safety block should win due to higher priority
				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeTrue())
				Expect(result.MatchedRule).To(Equal("safety-block"))
				Expect(result.Action).To(Equal(ActionBlock))
			})
		})

		Context("when handling empty policies", func() {
			It("should return fallthrough for nil policy", func() {
				engine := NewEngine(&Policy{})
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeFalse())
				Expect(result.Action).To(Equal(ActionFallthrough))
			})

			It("should return fallthrough for empty rules", func() {
				policy := &Policy{Rules: []Rule{}}
				engine := NewEngine(policy)
				result, err := engine.Evaluate(context)

				Expect(err).ToNot(HaveOccurred())
				Expect(result.Matched).To(BeFalse())
				Expect(result.Action).To(Equal(ActionFallthrough))
			})
		})
	})

	Describe("SignalContext", func() {
		var context *SignalContext

		BeforeEach(func() {
			context = NewSignalContext()
		})

		It("should add and retrieve signals", func() {
			signal := Signal{
				Provider: "keyword",
				Name:     "kubernetes",
				Matched:  true,
			}

			context.AddSignal(signal)

			retrieved, exists := context.GetSignal("keyword", "kubernetes")
			Expect(exists).To(BeTrue())
			Expect(retrieved.Matched).To(BeTrue())
		})

		It("should return false for non-existent signals", func() {
			_, exists := context.GetSignal("keyword", "nonexistent")
			Expect(exists).To(BeFalse())
		})

		It("should overwrite existing signals", func() {
			signal1 := Signal{
				Provider: "keyword",
				Name:     "test",
				Matched:  false,
			}
			signal2 := Signal{
				Provider: "keyword",
				Name:     "test",
				Matched:  true,
			}

			context.AddSignal(signal1)
			context.AddSignal(signal2)

			retrieved, exists := context.GetSignal("keyword", "test")
			Expect(exists).To(BeTrue())
			Expect(retrieved.Matched).To(BeTrue())
		})
	})
})
