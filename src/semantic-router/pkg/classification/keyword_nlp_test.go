package classification

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBM25KeywordClassifier tests BM25-based keyword classification via nlp-binding.
func TestBM25KeywordClassifier(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name:     "code_keywords",
			Operator: "OR",
			Method:   "bm25",
			Keywords: []string{
				"code", "function", "implement", "debug",
				"algorithm", "compile", "syntax", "variable",
			},
			BM25Threshold: 0.1,
			CaseSensitive: false,
		},
		{
			Name:     "medical_keywords",
			Operator: "OR",
			Method:   "bm25",
			Keywords: []string{
				"diagnosis", "treatment", "symptoms", "prescription",
				"surgery", "patient", "medication",
			},
			BM25Threshold: 0.1,
			CaseSensitive: false,
		},
	}

	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("Failed to create BM25 classifier: %v", err)
	}
	defer classifier.Free()

	testCases := []struct {
		query       string
		expectRule  string
		expectMatch bool
		description string
	}{
		// Code matches
		{"Help me debug this function", "code_keywords", true, "BM25: debug + function"},
		{"I need to implement an algorithm", "code_keywords", true, "BM25: implement + algorithm"},
		{"Fix the syntax error in my code", "code_keywords", true, "BM25: syntax + code"},

		// Medical matches
		{"What are the symptoms of flu?", "medical_keywords", true, "BM25: symptoms"},
		{"The patient needs surgery", "medical_keywords", true, "BM25: patient + surgery"},
		{"Discuss treatment options for diagnosis", "medical_keywords", true, "BM25: treatment + diagnosis"},

		// No match
		{"What is the weather like today?", "", false, "BM25: no match - weather"},
		{"How do I cook pasta?", "", false, "BM25: no match - cooking"},

		// Multi-rule priority (first-match: code before medical)
		{"Help me code a medical app", "code_keywords", true, "BM25: first-match on code"},
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("BM25 KEYWORD CLASSIFIER TEST")
	fmt.Println(strings.Repeat("=", 100))

	passed := 0
	for _, tc := range testCases {
		category, keywords, err := classifier.ClassifyWithKeywords(tc.query)
		if err != nil {
			t.Errorf("[FAIL] %s: error: %v", tc.description, err)
			continue
		}

		matched := category != ""
		ok := matched == tc.expectMatch && (category == tc.expectRule || !tc.expectMatch)

		status := "PASS"
		if !ok {
			status = "FAIL"
			t.Errorf("[FAIL] %s: expected rule=%q match=%v, got rule=%q match=%v keywords=%v",
				tc.description, tc.expectRule, tc.expectMatch, category, matched, keywords)
		} else {
			passed++
		}

		fmt.Printf("  [%s] %s\n", status, tc.description)
		fmt.Printf("         Query: %q\n", tc.query)
		if matched {
			fmt.Printf("         Rule: %s, Keywords: %v\n", category, keywords)
		}
	}

	fmt.Printf("\n  Results: %d/%d passed\n", passed, len(testCases))
	fmt.Println(strings.Repeat("=", 100))
}

// TestNgramKeywordClassifier tests N-gram based keyword classification via nlp-binding.
func TestNgramKeywordClassifier(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name:     "urgent_request",
			Operator: "OR",
			Method:   "ngram",
			Keywords: []string{
				"urgent", "immediate", "asap", "emergency", "critical",
			},
			NgramThreshold: 0.4,
			NgramArity:     3,
			CaseSensitive:  false,
		},
	}

	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("Failed to create N-gram classifier: %v", err)
	}
	defer classifier.Free()

	testCases := []struct {
		query       string
		expectMatch bool
		description string
	}{
		// Exact matches
		{"This is urgent please help", true, "N-gram: exact 'urgent'"},
		{"I need immediate assistance", true, "N-gram: exact 'immediate'"},
		{"This is an emergency", true, "N-gram: exact 'emergency'"},

		// Typo tolerance (the killer feature of n-gram)
		{"This is urgnet please help", true, "N-gram: typo 'urgnet' -> 'urgent'"},
		{"I need immedaite help", true, "N-gram: typo 'immedaite' -> 'immediate'"},
		{"This is an emergncy situation", true, "N-gram: typo 'emergncy' -> 'emergency'"},
		{"The situation is critcal", true, "N-gram: typo 'critcal' -> 'critical'"},

		// No match
		{"What is the weather like?", false, "N-gram: no match - weather"},
		{"How do I cook pasta?", false, "N-gram: no match - cooking"},
		{"Tell me about history", false, "N-gram: no match - history"},
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("N-GRAM KEYWORD CLASSIFIER TEST (with typo tolerance)")
	fmt.Println(strings.Repeat("=", 100))

	passed := 0
	for _, tc := range testCases {
		category, keywords, err := classifier.ClassifyWithKeywords(tc.query)
		if err != nil {
			t.Errorf("[FAIL] %s: error: %v", tc.description, err)
			continue
		}

		matched := category != ""
		ok := matched == tc.expectMatch

		status := "PASS"
		if !ok {
			status = "FAIL"
			t.Errorf("[FAIL] %s: expected match=%v, got match=%v (rule=%q keywords=%v)",
				tc.description, tc.expectMatch, matched, category, keywords)
		} else {
			passed++
		}

		fmt.Printf("  [%s] %s\n", status, tc.description)
		fmt.Printf("         Query: %q\n", tc.query)
		if matched {
			fmt.Printf("         Rule: %s, Keywords: %v\n", category, keywords)
		}
	}

	fmt.Printf("\n  Results: %d/%d passed\n", passed, len(testCases))
	fmt.Println(strings.Repeat("=", 100))
}

// TestNgramANDOperator tests the AND operator with N-gram matching.
func TestNgramANDOperator(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name:           "sensitive_data",
			Operator:       "AND",
			Method:         "ngram",
			Keywords:       []string{"password", "credentials"},
			NgramThreshold: 0.5,
			NgramArity:     3,
			CaseSensitive:  false,
		},
	}

	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("Failed to create N-gram AND classifier: %v", err)
	}
	defer classifier.Free()

	testCases := []struct {
		query       string
		expectMatch bool
		description string
	}{
		{"My password and credentials were leaked", true, "N-gram AND: both present"},
		{"Reset my pasword and credentials now", true, "N-gram AND: typo 'pasword' + exact 'credentials'"},
		{"My password was stolen", false, "N-gram AND: only password"},
		{"The credentials are expired", false, "N-gram AND: only credentials"},
		{"How is the weather?", false, "N-gram AND: neither present"},
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("N-GRAM AND OPERATOR TEST")
	fmt.Println(strings.Repeat("=", 100))

	passed := 0
	for _, tc := range testCases {
		category, keywords, err := classifier.ClassifyWithKeywords(tc.query)
		if err != nil {
			t.Errorf("[FAIL] %s: error: %v", tc.description, err)
			continue
		}

		matched := category != ""
		ok := matched == tc.expectMatch

		status := "PASS"
		if !ok {
			status = "FAIL"
			t.Errorf("[FAIL] %s: expected match=%v, got match=%v (keywords=%v)",
				tc.description, tc.expectMatch, matched, keywords)
		} else {
			passed++
		}

		fmt.Printf("  [%s] %s\n", status, tc.description)
		if matched {
			fmt.Printf("         Keywords: %v\n", keywords)
		}
	}

	fmt.Printf("\n  Results: %d/%d passed\n", passed, len(testCases))
	fmt.Println(strings.Repeat("=", 100))
}

// TestMixedMethodClassifier tests a classifier with all three methods mixed together.
func TestMixedMethodClassifier(t *testing.T) {
	rules := []config.KeywordRule{
		// BM25 rule
		{
			Name:          "code_keywords",
			Operator:      "OR",
			Method:        "bm25",
			Keywords:      []string{"code", "function", "debug", "compile"},
			BM25Threshold: 0.1,
			CaseSensitive: false,
		},
		// N-gram rule (with typo tolerance)
		{
			Name:           "urgent_request",
			Operator:       "OR",
			Method:         "ngram",
			Keywords:       []string{"urgent", "emergency", "critical"},
			NgramThreshold: 0.4,
			NgramArity:     3,
			CaseSensitive:  false,
		},
		// Regex rule (exact match)
		{
			Name:          "exclude_spam",
			Operator:      "NOR",
			Method:        "regex",
			Keywords:      []string{"buy now", "free money"},
			CaseSensitive: false,
		},
		// Default (regex) when method is empty
		{
			Name:          "exact_match",
			Operator:      "OR",
			Keywords:      []string{"hello world"},
			CaseSensitive: false,
		},
	}

	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("Failed to create mixed classifier: %v", err)
	}
	defer classifier.Free()

	testCases := []struct {
		query       string
		expectRule  string
		expectMatch bool
		description string
	}{
		// BM25 matches
		{"Help me debug this code", "code_keywords", true, "Mixed: BM25 code match"},

		// N-gram matches (including typos!)
		{"This is urgnet help me", "urgent_request", true, "Mixed: N-gram typo match"},
		{"There is an emergency", "urgent_request", true, "Mixed: N-gram exact match"},

		// Regex NOR match (no spam = matches)
		{"Tell me about the weather", "exclude_spam", true, "Mixed: Regex NOR passes (clean text)"},

		// Regex NOR fails (spam detected)
		{"Buy now and get free money", "", false, "Mixed: Regex NOR blocks spam"},

		// Default regex match — NOR rule fires first since clean text has no spam
		// "hello world" in clean text still hits exclude_spam NOR first
		{"Just say hello world please", "exclude_spam", true, "Mixed: NOR fires before exact_match (clean text)"},
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("MIXED METHOD CLASSIFIER TEST (BM25 + N-gram + Regex)")
	fmt.Println(strings.Repeat("=", 100))

	passed := 0
	for _, tc := range testCases {
		category, confidence, err := classifier.Classify(tc.query)
		if err != nil {
			t.Errorf("[FAIL] %s: error: %v", tc.description, err)
			continue
		}

		matched := category != ""
		ok := matched == tc.expectMatch && (category == tc.expectRule || !tc.expectMatch)

		status := "PASS"
		if !ok {
			status = "FAIL"
			t.Errorf("[FAIL] %s: expected rule=%q match=%v, got rule=%q match=%v",
				tc.description, tc.expectRule, tc.expectMatch, category, matched)
		} else {
			passed++
		}

		fmt.Printf("  [%s] %s\n", status, tc.description)
		fmt.Printf("         Query: %q\n", tc.query)
		if matched {
			fmt.Printf("         Rule: %s, Confidence: %.4f\n", category, confidence)
		}
	}

	fmt.Printf("\n  Results: %d/%d passed\n", passed, len(testCases))
	fmt.Println(strings.Repeat("=", 100))
}

// TestBM25vsRegexComparison demonstrates BM25's advantage over regex for morphological variants.
func TestBM25vsRegexComparison(t *testing.T) {
	bm25Rules := []config.KeywordRule{
		{
			Name:          "urgent_request",
			Operator:      "OR",
			Method:        "bm25",
			Keywords:      []string{"urgent", "emergency"},
			BM25Threshold: 0.1,
			CaseSensitive: false,
		},
	}
	regexRules := []config.KeywordRule{
		{
			Name:          "urgent_request",
			Operator:      "OR",
			Method:        "regex",
			Keywords:      []string{"urgent", "emergency"},
			CaseSensitive: false,
		},
	}

	bm25Classifier, _ := NewKeywordClassifier(bm25Rules)
	defer bm25Classifier.Free()
	regexClassifier, _ := NewKeywordClassifier(regexRules)
	defer regexClassifier.Free()

	queries := []string{
		"This is urgent",       // exact - both match
		"This is an emergency", // exact - both match
		"Handle this urgently", // morphological variant - BM25 may match via stemming
		"What is the weather?", // no match - neither
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("BM25 vs REGEX COMPARISON")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Printf("  %-45s %-20s %-20s\n", "Query", "BM25", "Regex")
	fmt.Println("  " + strings.Repeat("-", 85))

	for _, q := range queries {
		bm25Cat, _, _ := bm25Classifier.ClassifyWithKeywords(q)
		regexCat, _, _ := regexClassifier.ClassifyWithKeywords(q)

		bm25Status := "no match"
		if bm25Cat != "" {
			bm25Status = bm25Cat
		}
		regexStatus := "no match"
		if regexCat != "" {
			regexStatus = regexCat
		}

		fmt.Printf("  %-45s %-20s %-20s\n", q, bm25Status, regexStatus)
	}

	fmt.Println(strings.Repeat("=", 100))
}

// TestNgramvsRegexFuzzyComparison demonstrates N-gram's advantage for typo tolerance vs regex+Levenshtein.
func TestNgramvsRegexFuzzyComparison(t *testing.T) {
	ngramRules := []config.KeywordRule{
		{
			Name:           "urgent_request",
			Operator:       "OR",
			Method:         "ngram",
			Keywords:       []string{"urgent", "emergency", "immediate"},
			NgramThreshold: 0.4,
			NgramArity:     3,
			CaseSensitive:  false,
		},
	}
	regexFuzzyRules := []config.KeywordRule{
		{
			Name:           "urgent_request",
			Operator:       "OR",
			Method:         "regex",
			Keywords:       []string{"urgent", "emergency", "immediate"},
			CaseSensitive:  false,
			FuzzyMatch:     true,
			FuzzyThreshold: 2,
		},
	}

	ngramClassifier, _ := NewKeywordClassifier(ngramRules)
	defer ngramClassifier.Free()
	regexFuzzyClassifier, _ := NewKeywordClassifier(regexFuzzyRules)
	defer regexFuzzyClassifier.Free()

	queries := []struct {
		text string
		desc string
	}{
		{"This is urgent", "exact match"},
		{"This is urgnet", "transposition"},
		{"This is an emergncy", "1 deletion"},
		{"I need immedaite help", "transposition"},
		{"This is urgentt help", "1 insertion"},
		{"What is the weather?", "no match"},
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("N-GRAM vs REGEX+LEVENSHTEIN COMPARISON (typo tolerance)")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Printf("  %-35s %-15s %-25s %-25s\n", "Query", "Type", "N-gram", "Regex+Levenshtein")
	fmt.Println("  " + strings.Repeat("-", 95))

	for _, q := range queries {
		nCat, nKw, _ := ngramClassifier.ClassifyWithKeywords(q.text)
		rCat, rKw, _ := regexFuzzyClassifier.ClassifyWithKeywords(q.text)

		nStatus := "no match"
		if nCat != "" {
			nStatus = fmt.Sprintf("match: %v", nKw)
		}
		rStatus := "no match"
		if rCat != "" {
			rStatus = fmt.Sprintf("match: %v", rKw)
		}

		fmt.Printf("  %-35s %-15s %-25s %-25s\n", q.text, q.desc, nStatus, rStatus)
	}

	fmt.Println(strings.Repeat("=", 100))
}
