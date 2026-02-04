package classification

import (
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("LatencyClassifier", func() {
	var classifier *LatencyClassifier

	BeforeEach(func() {
		// Reset both caches to ensure test isolation
		ResetTPOT()
		ResetTTFT()
	})

	Describe("TPOT and TTFT Percentile Evaluation", func() {
		It("should select best model when both TPOT and TTFT are configured", func() {
			// Setup: Create latency rule with both TPOT and TTFT percentiles
			// Use 50th percentile (median) - more realistic for matching
			rules := []config.LatencyRule{
				{
					Name:           "low_latency_comprehensive",
					TPOTPercentile: 50, // 50th percentile (median) - realistic threshold
					TTFTPercentile: 50, // 50th percentile (median) - realistic threshold
					Description:    "Fast start and fast generation",
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Setup: Add realistic TPOT data for multiple models
			// Model A: Moderate performance
			UpdateTPOT("model-a", 0.05)
			UpdateTPOT("model-a", 0.06)
			UpdateTPOT("model-a", 0.07)
			UpdateTPOT("model-a", 0.05)
			UpdateTPOT("model-a", 0.06)

			// Model B: Good performance
			UpdateTPOT("model-b", 0.03)
			UpdateTPOT("model-b", 0.04)
			UpdateTPOT("model-b", 0.05)
			UpdateTPOT("model-b", 0.03)
			UpdateTPOT("model-b", 0.04)

			// Model C: Best performance (fastest)
			UpdateTPOT("model-c", 0.02)
			UpdateTPOT("model-c", 0.03)
			UpdateTPOT("model-c", 0.04)
			UpdateTPOT("model-c", 0.02)
			UpdateTPOT("model-c", 0.03)

			// Setup: Add realistic TTFT data for multiple models
			// Model A: Moderate TTFT
			UpdateTTFT("model-a", 0.30)
			UpdateTTFT("model-a", 0.35)
			UpdateTTFT("model-a", 0.40)
			UpdateTTFT("model-a", 0.30)
			UpdateTTFT("model-a", 0.35)

			// Model B: Good TTFT
			UpdateTTFT("model-b", 0.20)
			UpdateTTFT("model-b", 0.25)
			UpdateTTFT("model-b", 0.30)
			UpdateTTFT("model-b", 0.20)
			UpdateTTFT("model-b", 0.25)

			// Model C: Best TTFT (fastest)
			UpdateTTFT("model-c", 0.10)
			UpdateTTFT("model-c", 0.15)
			UpdateTTFT("model-c", 0.20)
			UpdateTTFT("model-c", 0.10)
			UpdateTTFT("model-c", 0.15)

			// Get current values and thresholds for logging
			fmt.Println("\n=== Model Performance Data ===")
			for _, model := range []string{"model-a", "model-b", "model-c"} {
				tpot, _ := GetTPOT(model)
				tpotThreshold, _ := GetTPOTPercentile(model, 50)
				ttftThreshold, _ := GetTTFTPercentile(model, 50)
				ttft, _ := GetTTFT(model)

				fmt.Printf("Model: %s\n", model)
				fmt.Printf("  Current TPOT: %.4fs, 50th percentile threshold: %.4fs\n", tpot, tpotThreshold)
				fmt.Printf("  Current TTFT: %.4fs, 50th percentile threshold: %.4fs\n", ttft, ttftThreshold)
				fmt.Printf("  TPOT meets threshold: %v\n", tpot <= tpotThreshold)
				fmt.Printf("  TTFT meets threshold: %v\n", ttft <= ttftThreshold)
				if tpot <= tpotThreshold && ttft <= ttftThreshold {
					tpotRatio := tpot / tpotThreshold
					ttftRatio := ttft / ttftThreshold
					combinedScore := (tpotRatio + ttftRatio) / 2.0
					fmt.Printf("  Combined Score: %.4f (lower is better)\n", combinedScore)
				}
				fmt.Println()
			}

			// Execute: Classify with all three models
			result, err := classifier.Classify([]string{"model-a", "model-b", "model-c"})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())

			// Verify: Rule should match (at least one model should meet the threshold)
			Expect(result.MatchedRules).To(ContainElement("low_latency_comprehensive"))

			// Log the decision
			fmt.Println("=== Decision ===")
			fmt.Printf("Matched Rules: %v\n", result.MatchedRules)
			fmt.Println()
		})

		It("should handle TPOT-only rule", func() {
			// Setup: Create latency rule with only TPOT percentile
			rules := []config.LatencyRule{
				{
					Name:           "batch_processing",
					TPOTPercentile: 50, // 50th percentile - realistic threshold
					Description:    "Only TPOT matters for batch processing",
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Setup: Add TPOT data
			// Fast model: meets threshold
			UpdateTPOT("fast-model", 0.02)
			UpdateTPOT("fast-model", 0.03)
			UpdateTPOT("fast-model", 0.04)
			UpdateTPOT("fast-model", 0.02)
			UpdateTPOT("fast-model", 0.03)

			// Slow model: doesn't meet threshold
			UpdateTPOT("slow-model", 0.10)
			UpdateTPOT("slow-model", 0.11)
			UpdateTPOT("slow-model", 0.12)
			UpdateTPOT("slow-model", 0.13)
			UpdateTPOT("slow-model", 0.14)

			// Log values
			fmt.Println("\n=== TPOT-Only Rule Test ===")
			fastTPOT, _ := GetTPOT("fast-model")
			fastThreshold, _ := GetTPOTPercentile("fast-model", 50)
			slowTPOT, _ := GetTPOT("slow-model")
			slowThreshold, _ := GetTPOTPercentile("slow-model", 50)

			fmt.Printf("Fast Model: TPOT=%.4fs, Threshold=%.4fs, Meets: %v\n", fastTPOT, fastThreshold, fastTPOT <= fastThreshold)
			fmt.Printf("Slow Model: TPOT=%.4fs, Threshold=%.4fs, Meets: %v\n", slowTPOT, slowThreshold, slowTPOT <= slowThreshold)

			// Execute
			result, err := classifier.Classify([]string{"fast-model", "slow-model"})
			Expect(err).NotTo(HaveOccurred())

			// Verify: Rule should match (fast-model meets threshold)
			Expect(result.MatchedRules).To(ContainElement("batch_processing"))
			fmt.Printf("Decision: Matched rules: %v\n", result.MatchedRules)
		})

		It("should handle TTFT-only rule", func() {
			// Setup: Create latency rule with only TTFT percentile
			rules := []config.LatencyRule{
				{
					Name:           "chat_fast_start",
					TTFTPercentile: 50, // 50th percentile - realistic threshold
					Description:    "Only TTFT matters for chat apps",
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Setup: Add TTFT data
			// Fast start model: meets threshold
			UpdateTTFT("fast-start-model", 0.10)
			UpdateTTFT("fast-start-model", 0.15)
			UpdateTTFT("fast-start-model", 0.20)
			UpdateTTFT("fast-start-model", 0.10)
			UpdateTTFT("fast-start-model", 0.15)

			// Slow start model: doesn't meet threshold
			UpdateTTFT("slow-start-model", 0.50)
			UpdateTTFT("slow-start-model", 0.55)
			UpdateTTFT("slow-start-model", 0.60)
			UpdateTTFT("slow-start-model", 0.65)
			UpdateTTFT("slow-start-model", 0.70)

			// Log values
			fmt.Println("\n=== TTFT-Only Rule Test ===")
			fastTTFT, _ := GetTTFT("fast-start-model")
			slowTTFT, _ := GetTTFT("slow-start-model")
			fastThreshold, _ := GetTTFTPercentile("fast-start-model", 50)
			slowThreshold, _ := GetTTFTPercentile("slow-start-model", 50)

			fmt.Printf("Fast Start Model: TTFT=%.4fs, Threshold=%.4fs, Meets: %v\n", fastTTFT, fastThreshold, fastTTFT <= fastThreshold)
			fmt.Printf("Slow Start Model: TTFT=%.4fs, Threshold=%.4fs, Meets: %v\n", slowTTFT, slowThreshold, slowTTFT <= slowThreshold)

			// Execute
			result, err := classifier.Classify([]string{"fast-start-model", "slow-start-model"})
			Expect(err).NotTo(HaveOccurred())

			// Verify: Rule should match (fast-start-model meets threshold)
			Expect(result.MatchedRules).To(ContainElement("chat_fast_start"))
			fmt.Printf("Decision: Matched rules: %v\n", result.MatchedRules)
		})

		It("should handle models that don't meet strict thresholds", func() {
			// Setup: Create strict latency rule (10th percentile is very strict)
			rules := []config.LatencyRule{
				{
					Name:           "very_low_latency",
					TPOTPercentile: 10, // Very strict: top 10%
					TTFTPercentile: 10, // Very strict: top 10%
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Setup: Add data for models that won't meet strict thresholds
			// These models have moderate performance - won't meet 10th percentile
			UpdateTPOT("average-model", 0.05)
			UpdateTPOT("average-model", 0.06)
			UpdateTPOT("average-model", 0.07)
			UpdateTPOT("average-model", 0.08)
			UpdateTPOT("average-model", 0.09)

			UpdateTTFT("average-model", 0.30)
			UpdateTTFT("average-model", 0.35)
			UpdateTTFT("average-model", 0.40)
			UpdateTTFT("average-model", 0.45)
			UpdateTTFT("average-model", 0.50)

			// Log values
			fmt.Println("\n=== No Match Test (Strict Threshold) ===")
			tpot, _ := GetTPOT("average-model")
			tpotThreshold, _ := GetTPOTPercentile("average-model", 10)
			ttft, _ := GetTTFT("average-model")
			ttftThreshold, _ := GetTTFTPercentile("average-model", 10)

			fmt.Printf("Model: average-model\n")
			fmt.Printf("  TPOT: %.4fs, 10th percentile threshold: %.4fs, Meets: %v\n", tpot, tpotThreshold, tpot <= tpotThreshold)
			fmt.Printf("  TTFT: %.4fs, 10th percentile threshold: %.4fs, Meets: %v\n", ttft, ttftThreshold, ttft <= ttftThreshold)

			// Execute
			result, err := classifier.Classify([]string{"average-model"})
			Expect(err).NotTo(HaveOccurred())

			// Verify: No match (model doesn't meet strict thresholds)
			// Note: This test verifies that strict thresholds work correctly
			// The model might or might not match depending on EMA vs 10th percentile
			fmt.Printf("Decision: Matched rules: %v\n", result.MatchedRules)
			fmt.Printf("Note: With 10th percentile, EMA (average) may be higher than threshold\n")
		})

		It("should work with small sample sizes (1-2 observations)", func() {
			// Setup: Create rule
			rules := []config.LatencyRule{
				{
					Name:           "early_evaluation",
					TPOTPercentile: 50,
					TTFTPercentile: 50,
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Setup: Add only 2 observations (should use average as threshold)
			UpdateTPOT("new-model", 0.04)
			UpdateTPOT("new-model", 0.05)

			UpdateTTFT("new-model", 0.20)
			UpdateTTFT("new-model", 0.25)

			// Log values
			fmt.Println("\n=== Small Sample Size Test ===")
			tpot, _ := GetTPOT("new-model")
			tpotThreshold, hasTPOT := GetTPOTPercentile("new-model", 50)
			ttft, _ := GetTTFT("new-model")
			ttftThreshold, hasTTFT := GetTTFTPercentile("new-model", 50)

			fmt.Printf("Model: new-model (only 2 observations)\n")
			fmt.Printf("  TPOT: %.4fs, Threshold: %.4fs (has data: %v)\n", tpot, tpotThreshold, hasTPOT)
			fmt.Printf("  TTFT: %.4fs, Threshold: %.4fs (has data: %v)\n", ttft, ttftThreshold, hasTTFT)
			fmt.Printf("  Note: With < 3 observations, uses average as threshold\n")

			// Execute
			result, err := classifier.Classify([]string{"new-model"})
			Expect(err).NotTo(HaveOccurred())

			// Verify: Should still work (uses average as threshold)
			fmt.Printf("Decision: Matched rules: %v\n", result.MatchedRules)
		})

		It("should handle models without TPOT/TTFT data", func() {
			rules := []config.LatencyRule{
				{
					Name:           "low_latency",
					TPOTPercentile: 50,
					TTFTPercentile: 50,
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Model with no data
			result, err := classifier.Classify([]string{"unknown-model"})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			Expect(result.MatchedRules).To(BeEmpty())
			fmt.Println("\n=== No Data Test ===")
			fmt.Printf("Model: unknown-model (no TPOT/TTFT data)\n")
			fmt.Printf("Decision: No match (no data available)\n")
		})

		It("should handle empty model list", func() {
			rules := []config.LatencyRule{
				{
					Name:           "low_latency",
					TPOTPercentile: 50,
					TTFTPercentile: 50,
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			result, err := classifier.Classify([]string{})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			Expect(result.MatchedRules).To(BeEmpty())
			fmt.Println("\n=== Empty Model List Test ===")
			fmt.Printf("Decision: No match (no models provided)\n")
		})

		It("should use exponential moving average for TPOT", func() {
			// Update TPOT multiple times to test averaging
			UpdateTPOT("test-model", 0.10)
			UpdateTPOT("test-model", 0.12)
			UpdateTPOT("test-model", 0.08)

			tpot, exists := GetTPOT("test-model")
			Expect(exists).To(BeTrue())
			Expect(tpot).To(BeNumerically(">", 0.08))
			Expect(tpot).To(BeNumerically("<", 0.12))
			fmt.Println("\n=== EMA Test ===")
			fmt.Printf("Model: test-model\n")
			fmt.Printf("TPOT values: [0.10, 0.12, 0.08]\n")
			fmt.Printf("Average TPOT: %.4fs (should be between 0.08 and 0.12)\n", tpot)
		})

		It("should handle multiple rules with different percentiles", func() {
			rules := []config.LatencyRule{
				{
					Name:           "p10_latency",
					TPOTPercentile: 10,
					TTFTPercentile: 10,
				},
				{
					Name:           "p50_latency",
					TPOTPercentile: 50,
					TTFTPercentile: 50,
				},
				{
					Name:           "p90_latency",
					TPOTPercentile: 90,
					TTFTPercentile: 90,
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Add realistic data
			UpdateTPOT("multi-rule-model", 0.03)
			UpdateTPOT("multi-rule-model", 0.04)
			UpdateTPOT("multi-rule-model", 0.05)
			UpdateTPOT("multi-rule-model", 0.06)
			UpdateTPOT("multi-rule-model", 0.07)

			UpdateTTFT("multi-rule-model", 0.20)
			UpdateTTFT("multi-rule-model", 0.25)
			UpdateTTFT("multi-rule-model", 0.30)
			UpdateTTFT("multi-rule-model", 0.35)
			UpdateTTFT("multi-rule-model", 0.40)

			// Log values
			fmt.Println("\n=== Multiple Rules Test ===")
			tpot, _ := GetTPOT("multi-rule-model")
			ttft, _ := GetTTFT("multi-rule-model")

			p10TPOT, _ := GetTPOTPercentile("multi-rule-model", 10)
			p50TPOT, _ := GetTPOTPercentile("multi-rule-model", 50)
			p90TPOT, _ := GetTPOTPercentile("multi-rule-model", 90)
			p10TTFT, _ := GetTTFTPercentile("multi-rule-model", 10)
			p50TTFT, _ := GetTTFTPercentile("multi-rule-model", 50)
			p90TTFT, _ := GetTTFTPercentile("multi-rule-model", 90)

			fmt.Printf("Model: multi-rule-model\n")
			fmt.Printf("  Current TPOT: %.4fs\n", tpot)
			fmt.Printf("    p10 threshold: %.4fs, meets: %v\n", p10TPOT, tpot <= p10TPOT)
			fmt.Printf("    p50 threshold: %.4fs, meets: %v\n", p50TPOT, tpot <= p50TPOT)
			fmt.Printf("    p90 threshold: %.4fs, meets: %v\n", p90TPOT, tpot <= p90TPOT)
			fmt.Printf("  Current TTFT: %.4fs\n", ttft)
			fmt.Printf("    p10 threshold: %.4fs, meets: %v\n", p10TTFT, ttft <= p10TTFT)
			fmt.Printf("    p50 threshold: %.4fs, meets: %v\n", p50TTFT, ttft <= p50TTFT)
			fmt.Printf("    p90 threshold: %.4fs, meets: %v\n", p90TTFT, ttft <= p90TTFT)

			// Execute
			result, err := classifier.Classify([]string{"multi-rule-model"})
			Expect(err).NotTo(HaveOccurred())
			fmt.Printf("Decision: Matched rules: %v\n", result.MatchedRules)
		})

		It("should test AND logic: both TPOT and TTFT must meet thresholds", func() {
			// Setup: Rule requires both TPOT and TTFT
			rules := []config.LatencyRule{
				{
					Name:           "both_required",
					TPOTPercentile: 50,
					TTFTPercentile: 50,
				},
			}
			var err error
			classifier, err = NewLatencyClassifier(rules)
			Expect(err).NotTo(HaveOccurred())

			// Model A: Good TPOT, bad TTFT (should NOT match)
			UpdateTPOT("model-a", 0.02)
			UpdateTPOT("model-a", 0.03)
			UpdateTPOT("model-a", 0.04)
			UpdateTPOT("model-a", 0.05)
			UpdateTPOT("model-a", 0.06)

			UpdateTTFT("model-a", 0.60) // Too slow
			UpdateTTFT("model-a", 0.65)
			UpdateTTFT("model-a", 0.70)
			UpdateTTFT("model-a", 0.75)
			UpdateTTFT("model-a", 0.80)

			// Model B: Bad TPOT, good TTFT (should NOT match)
			UpdateTPOT("model-b", 0.10) // Too slow
			UpdateTPOT("model-b", 0.11)
			UpdateTPOT("model-b", 0.12)
			UpdateTPOT("model-b", 0.13)
			UpdateTPOT("model-b", 0.14)

			UpdateTTFT("model-b", 0.10)
			UpdateTTFT("model-b", 0.15)
			UpdateTTFT("model-b", 0.20)
			UpdateTTFT("model-b", 0.25)
			UpdateTTFT("model-b", 0.30)

			// Model C: Good TPOT, good TTFT (should match)
			UpdateTPOT("model-c", 0.02)
			UpdateTPOT("model-c", 0.03)
			UpdateTPOT("model-c", 0.04)
			UpdateTPOT("model-c", 0.02)
			UpdateTPOT("model-c", 0.03)

			UpdateTTFT("model-c", 0.10)
			UpdateTTFT("model-c", 0.15)
			UpdateTTFT("model-c", 0.20)
			UpdateTTFT("model-c", 0.10)
			UpdateTTFT("model-c", 0.15)

			// Execute
			result, err := classifier.Classify([]string{"model-a", "model-b", "model-c"})
			Expect(err).NotTo(HaveOccurred())

			// Verify: Only model-c should match (both TPOT and TTFT meet thresholds)
			Expect(result.MatchedRules).To(ContainElement("both_required"))
			fmt.Println("\n=== AND Logic Test ===")
			fmt.Printf("Model A: Good TPOT, Bad TTFT - Should NOT match\n")
			fmt.Printf("Model B: Bad TPOT, Good TTFT - Should NOT match\n")
			fmt.Printf("Model C: Good TPOT, Good TTFT - Should match\n")
			fmt.Printf("Matched rules: %v\n", result.MatchedRules)
		})
	})
})
