package testcases

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Keyword Routing", func() {
	var (
		classifier      *classification.KeywordClassifier
		rules           []config.KeywordRule
		rulesWithoutNOR []config.KeywordRule
	)

	BeforeEach(func() {
		// Get all rules including NOR
		allRules := CreateKeywordTestRules()

		// Create version without NOR for tests that expect empty results
		rulesWithoutNOR = []config.KeywordRule{}
		for _, rule := range allRules {
			if rule.Operator != "NOR" {
				rulesWithoutNOR = append(rulesWithoutNOR, rule)
			}
		}

		// By default, use rules without NOR
		// Tests that specifically test NOR will create their own classifier
		rules = rulesWithoutNOR
		var err error
		classifier, err = CreateTestKeywordClassifier(rules)
		Expect(err).NotTo(HaveOccurred())
		Expect(classifier).NotTo(BeNil())
	})

	Context("OR Operator", func() {
		It("should match when any keyword is present", func() {
			testCases := []struct {
				query            string
				expectedCategory string
			}{
				{"I need urgent help", "urgent_request"},
				{"This is an immediate issue", "urgent_request"},
				{"Please respond asap", "urgent_request"},
				{"This is an emergency situation", "urgent_request"},
			}

			for _, tc := range testCases {
				category, confidence, err := classifier.Classify(tc.query)
				Expect(err).NotTo(HaveOccurred(), "Query: %s", tc.query)
				Expect(category).To(Equal(tc.expectedCategory),
					"Query '%s' should match category %s", tc.query, tc.expectedCategory)
				Expect(confidence).To(Equal(1.0), "Keyword matches should have 100%% confidence")
			}
		})

		It("should not match when no keywords are present", func() {
			category, _, err := classifier.Classify("Just a normal query")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(BeEmpty())
		})

		It("should be case-insensitive when configured", func() {
			testCases := []string{
				"This is URGENT",
				"This is Urgent",
				"This is urgent",
				"This is UrGeNt",
			}

			for _, query := range testCases {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("urgent_request"),
					"Query '%s' should match case-insensitively", query)
			}
		})

		It("should match keyword at beginning of text", func() {
			category, _, err := classifier.Classify("Urgent: please help")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})

		It("should match keyword at end of text", func() {
			category, _, err := classifier.Classify("Please help, this is urgent")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})

		It("should match keyword in middle of text", func() {
			category, _, err := classifier.Classify("This is an urgent matter that needs attention")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})
	})

	Context("AND Operator", func() {
		It("should match when all keywords are present", func() {
			query := "My SSN and credit card were stolen"
			category, confidence, err := classifier.Classify(query)

			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("sensitive_data"))
			Expect(confidence).To(Equal(1.0))
		})

		It("should not match when only some keywords are present", func() {
			queries := []string{
				"My SSN was stolen",         // Only SSN
				"My credit card was stolen", // Only credit card
				"Something else entirely",   // Neither
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).NotTo(Equal("sensitive_data"),
					"Query '%s' should not match AND rule", query)
			}
		})

		It("should match regardless of keyword order", func() {
			queries := []string{
				"My SSN and credit card",
				"My credit card and SSN",
				"SSN credit card stolen",
				"credit card and SSN compromised",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("sensitive_data"))
			}
		})

		It("should match with keywords far apart in text", func() {
			query := "My SSN was compromised yesterday, and today I noticed my credit card was also affected"
			category, _, err := classifier.Classify(query)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("sensitive_data"))
		})

		It("should match with repeated keywords", func() {
			query := "SSN SSN credit card credit card"
			category, _, err := classifier.Classify(query)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("sensitive_data"))
		})
	})

	Context("NOR Operator", func() {
		var norClassifier *classification.KeywordClassifier

		BeforeEach(func() {
			// Create classifier with ALL rules including NOR for these tests
			allRules := CreateKeywordTestRules()
			var err error
			norClassifier, err = CreateTestKeywordClassifier(allRules)
			Expect(err).NotTo(HaveOccurred())
			Expect(norClassifier).NotTo(BeNil())
		})

		It("should match spam when no forbidden keywords are present", func() {
			// NOR matches when NONE of the keywords are found
			queries := []string{
				"How do I reset my password?",
				"What is the capital of France?",
				"Can you help me with my account?",
			}

			for _, query := range queries {
				category, confidence, err := norClassifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("spam"),
					"Query '%s' should match spam via NOR (no spam keywords present)", query)
				Expect(confidence).To(Equal(1.0))
			}
		})

		It("should not match spam when any forbidden keyword is present", func() {
			// NOR does NOT match when any keyword is found
			queries := []string{
				"Buy now and save!",
				"Click here for free money",
				"Free money available now",
				"Buy now, click here for free money",
			}

			for _, query := range queries {
				category, _, err := norClassifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).NotTo(Equal("spam"),
					"Query '%s' should NOT match spam via NOR (spam keywords present)", query)
			}
		})
	})

	Context("Case Sensitivity", func() {
		It("should match exact case when case-sensitive enabled", func() {
			category, _, err := classifier.Classify("This is SECRET")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("case_sensitive_test"))
		})

		It("should not match different case when case-sensitive enabled", func() {
			queries := []string{
				"This is secret",
				"This is Secret",
				"This is sEcReT",
				"This is seCRet",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).NotTo(Equal("case_sensitive_test"),
					"Query '%s' should not match case-sensitive rule", query)
			}
		})

		It("should handle case-insensitive rules correctly", func() {
			// secret_detection has case_sensitive: false
			// Use lowercase to avoid matching case_sensitive_test first
			queries := []string{
				"This is secret",
				"This is Secret",
				"This is sEcReT",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("secret_detection"),
					"Query '%s' should match case-insensitive secret_detection", query)
			}
		})
	})

	Context("Word Boundaries", func() {
		It("should respect word boundaries - positive case", func() {
			queries := []string{
				"This is a secret",
				"The secret is safe",
				"secret meeting",
				"A secret!",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("secret_detection"),
					"Query '%s' should match secret as whole word", query)
			}
		})

		It("should respect word boundaries - negative case", func() {
			queries := []string{
				"Talk to my secretary",
				"The secretariat is here",
				"Secretive behavior",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).NotTo(Equal("secret_detection"),
					"Query '%s' should not match secret in partial word", query)
			}
		})

		It("should handle word boundaries with punctuation", func() {
			queries := []string{
				"secret.",
				"secret!",
				"secret?",
				"secret,",
				"(secret)",
				"\"secret\"",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("secret_detection"),
					"Query '%s' should match secret with punctuation", query)
			}
		})
	})

	Context("Regex Special Characters", func() {
		It("should handle dots literally", func() {
			queries := []string{
				"Version 1.0 released",
				"Using 2.0 now",
				"3.0 is coming",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("version_check"),
					"Query '%s' should match version with literal dot", query)
			}
		})

		It("should not match dots as wildcard", func() {
			// 1.0 should match literally, not 1X0
			category, _, err := classifier.Classify("Version 1X0")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).NotTo(Equal("version_check"))
		})

		It("should handle asterisks literally", func() {
			queries := []string{
				"The symbol * is here",
				"Use * wildcard",
				"asterisk *",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("wildcard_test"),
					"Query '%s' should match asterisk literally", query)
			}
		})
	})

	Context("Edge Cases", func() {
		It("should handle empty text", func() {
			category, _, err := classifier.Classify("")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(BeEmpty())
		})

		It("should handle whitespace-only text", func() {
			queries := []string{
				"   ",
				"\t\t",
				"\n\n",
				"   \t\n  ",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(BeEmpty())
			}
		})

		It("should handle very long text", func() {
			longPrefix := "This is normal text that goes on and on. "
			longSuffix := "More normal text. "
			var longText string
			for i := 0; i < 100; i++ {
				longText += longPrefix
			}
			longText += "urgent "
			for i := 0; i < 100; i++ {
				longText += longSuffix
			}

			category, _, err := classifier.Classify(longText)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})

		It("should handle Unicode characters", func() {
			queries := []string{
				"需要 urgent 帮助",
				"緊急 urgent 事項",
				"срочно urgent помощь",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("urgent_request"),
					"Query '%s' should match with Unicode", query)
			}
		})

		It("should handle emoji", func() {
			queries := []string{
				"🚨 urgent 🚨",
				"😱 urgent help 😱",
				"⚠️ urgent ⚠️",
			}

			for _, query := range queries {
				category, _, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(category).To(Equal("urgent_request"),
					"Query '%s' should match with emoji", query)
			}
		})

		It("should handle newlines in text", func() {
			query := "This is\nurgent\nhelp"
			category, _, err := classifier.Classify(query)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})

		It("should handle tabs in text", func() {
			query := "This is\turgent\thelp"
			category, _, err := classifier.Classify(query)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})
	})

	Context("Multiple Rule Matching", func() {
		It("should use first matching rule when multiple rules match", func() {
			// Add overlapping rules
			overlappingRules := []config.KeywordRule{
				{Name: "rule1", Operator: "OR", Keywords: []string{"urgent"}, CaseSensitive: false},
				{Name: "rule2", Operator: "OR", Keywords: []string{"urgent"}, CaseSensitive: false},
			}
			newClassifier, err := CreateTestKeywordClassifier(overlappingRules)
			Expect(err).NotTo(HaveOccurred())

			category, _, err := newClassifier.Classify("urgent request")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("rule1"), "Should match first rule")
		})

		It("should handle multiple different keywords matching", func() {
			query := "This is urgent and also an emergency"
			category, _, err := classifier.Classify(query)
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(Equal("urgent_request"))
		})
	})

	Context("Confidence Scores", func() {
		It("should always return confidence 1.0 for keyword matches", func() {
			testCases := []string{
				"urgent",
				"This is urgent",
				"URGENT",
				"My SSN and credit card",
			}

			for _, query := range testCases {
				_, confidence, err := classifier.Classify(query)
				Expect(err).NotTo(HaveOccurred())
				Expect(confidence).To(Equal(1.0),
					"Query '%s' should have confidence 1.0", query)
			}
		})
	})

	Context("Loading from JSON test data", func() {
		It("should pass all test cases from JSON file", func() {
			testCases, err := LoadKeywordTestCases("testdata/keyword_routing_cases.json")
			if err != nil {
				Skip("Test data file not found: " + err.Error())
				return
			}

			for _, tc := range testCases {
				category, confidence, err := classifier.Classify(tc.Query)
				Expect(err).NotTo(HaveOccurred(), "Test: %s - %s", tc.Name, tc.Description)

				if tc.ExpectedCategory != "" {
					Expect(category).To(Equal(tc.ExpectedCategory),
						"Test: %s - Query: %s", tc.Name, tc.Query)
				}

				if tc.ExpectedConfidence > 0 {
					Expect(confidence).To(Equal(tc.ExpectedConfidence),
						"Test: %s - Query: %s", tc.Name, tc.Query)
				}
			}
		})
	})

	Context("Error Handling", func() {
		It("should handle invalid operator gracefully", func() {
			invalidRules := []config.KeywordRule{
				{Name: "invalid", Operator: "INVALID", Keywords: []string{"test"}, CaseSensitive: false},
			}
			_, err := CreateTestKeywordClassifier(invalidRules)
			Expect(err).To(HaveOccurred())
		})

		It("should handle empty keywords array", func() {
			emptyRules := []config.KeywordRule{
				{Name: "empty", Operator: "OR", Keywords: []string{}, CaseSensitive: false},
			}
			newClassifier, err := CreateTestKeywordClassifier(emptyRules)
			Expect(err).NotTo(HaveOccurred())

			category, _, err := newClassifier.Classify("any text")
			Expect(err).NotTo(HaveOccurred())
			Expect(category).To(BeEmpty())
		})
	})
})
