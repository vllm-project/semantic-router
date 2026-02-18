package nlp_binding

import (
	"testing"
)

// ---------------------------------------------------------------------------
// BM25 Classifier Tests
// ---------------------------------------------------------------------------

func TestBM25ClassifierORMatch(t *testing.T) {
	c := NewBM25Classifier()
	defer c.Free()

	err := c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "asap", "emergency"},
		0.1, false)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	result := c.Classify("This is an urgent matter that needs attention")
	if !result.Matched {
		t.Fatal("Expected BM25 OR match, got no match")
	}
	if result.RuleName != "urgent_request" {
		t.Fatalf("Expected rule name 'urgent_request', got %q", result.RuleName)
	}
	if len(result.MatchedKeywords) == 0 {
		t.Fatal("Expected matched keywords, got none")
	}
	t.Logf("BM25 OR match: rule=%s keywords=%v scores=%v", result.RuleName, result.MatchedKeywords, result.Scores)
}

func TestBM25ClassifierORNoMatch(t *testing.T) {
	c := NewBM25Classifier()
	defer c.Free()

	err := c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "asap", "emergency"},
		0.1, false)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	result := c.Classify("The weather is nice today")
	if result.Matched {
		t.Fatalf("Expected no match, got rule=%s keywords=%v", result.RuleName, result.MatchedKeywords)
	}
}

func TestBM25ClassifierAND(t *testing.T) {
	c := NewBM25Classifier()
	defer c.Free()

	err := c.AddRule("code_request", "AND",
		[]string{"code", "debug"},
		0.1, false)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	// Both keywords present
	result := c.Classify("I need to debug this code")
	if !result.Matched {
		t.Fatal("Expected AND match with both keywords present")
	}
	t.Logf("BM25 AND match: rule=%s keywords=%v", result.RuleName, result.MatchedKeywords)

	// Only one keyword present
	result = c.Classify("I need help with my code")
	if result.Matched {
		t.Fatal("Expected no AND match with only one keyword")
	}
}

func TestBM25ClassifierNOR(t *testing.T) {
	c := NewBM25Classifier()
	defer c.Free()

	err := c.AddRule("not_spam", "NOR",
		[]string{"buy now", "free money"},
		0.1, false)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	// Clean text
	result := c.Classify("What is the weather like today")
	if !result.Matched {
		t.Fatal("Expected NOR match (no forbidden keywords found)")
	}

	// Spam text
	result = c.Classify("Buy now and get free money today")
	if result.Matched {
		t.Fatal("Expected NOR to NOT match when forbidden keywords present")
	}
}

func TestBM25MultipleRules(t *testing.T) {
	c := NewBM25Classifier()
	defer c.Free()

	_ = c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "emergency"},
		0.1, false)
	_ = c.AddRule("code_request", "OR",
		[]string{"code", "function", "debug", "implement"},
		0.1, false)

	result := c.Classify("I need to implement a function")
	if !result.Matched {
		t.Fatal("Expected match on code_request")
	}
	if result.RuleName != "code_request" {
		t.Fatalf("Expected 'code_request', got %q", result.RuleName)
	}
	t.Logf("BM25 multi-rule: rule=%s keywords=%v", result.RuleName, result.MatchedKeywords)
}

// ---------------------------------------------------------------------------
// N-gram Classifier Tests
// ---------------------------------------------------------------------------

func TestNgramClassifierORMatch(t *testing.T) {
	c := NewNgramClassifier()
	defer c.Free()

	err := c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "asap", "emergency"},
		0.4, false, 3)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	result := c.Classify("This is an urgent matter")
	if !result.Matched {
		t.Fatal("Expected N-gram OR match")
	}
	t.Logf("N-gram OR match: rule=%s keywords=%v similarities=%v", result.RuleName, result.MatchedKeywords, result.Scores)
}

func TestNgramClassifierFuzzyMatch(t *testing.T) {
	c := NewNgramClassifier()
	defer c.Free()

	err := c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "emergency"},
		0.4, false, 3)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	// Typo: "urgnet" should fuzzy-match "urgent"
	result := c.Classify("This is urgnet please help")
	if !result.Matched {
		t.Fatal("Expected N-gram fuzzy match for 'urgnet' -> 'urgent'")
	}
	t.Logf("N-gram fuzzy match: rule=%s keywords=%v similarities=%v", result.RuleName, result.MatchedKeywords, result.Scores)
}

func TestNgramClassifierNoMatch(t *testing.T) {
	c := NewNgramClassifier()
	defer c.Free()

	err := c.AddRule("urgent_request", "OR",
		[]string{"urgent", "immediate", "emergency"},
		0.5, false, 3)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	result := c.Classify("The weather is nice today")
	if result.Matched {
		t.Fatalf("Expected no match, got rule=%s keywords=%v", result.RuleName, result.MatchedKeywords)
	}
}

func TestNgramClassifierAND(t *testing.T) {
	c := NewNgramClassifier()
	defer c.Free()

	err := c.AddRule("code_request", "AND",
		[]string{"code", "debug"},
		0.5, false, 3)
	if err != nil {
		t.Fatalf("AddRule failed: %v", err)
	}

	result := c.Classify("I need to debug this code")
	if !result.Matched {
		t.Fatal("Expected N-gram AND match")
	}
	t.Logf("N-gram AND match: rule=%s keywords=%v", result.RuleName, result.MatchedKeywords)
}
