package classification

import (
	"fmt"
	"math"
	"testing"

	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type mockTokenCounter struct {
	count int
	err   error
}

func (m *mockTokenCounter) CountTokens(text string) (int, error) {
	return m.count, m.err
}

func TestContextClassifier(t *testing.T) {
	RegisterTestingT(t)

	rules := []config.ContextRule{
		{Name: "low", MinTokens: "0", MaxTokens: "1K"},
		{Name: "high", MinTokens: "4K", MaxTokens: "128K"},
	}

	t.Run("Classify low token count", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{count: 500}, rules)
		matched, count, err := classifier.Classify("some text")
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(500))
		Expect(matched).To(ConsistOf("low"))
	})

	t.Run("Classify high token count", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{count: 5000}, rules)
		matched, count, err := classifier.Classify("some text")
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(5000))
		Expect(matched).To(ConsistOf("high"))
	})

	t.Run("Classify no match", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{count: 2000}, rules)
		matched, count, err := classifier.Classify("some text")
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(2000))
		Expect(matched).To(BeEmpty())
	})

	t.Run("Token counter error", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{err: fmt.Errorf("error")}, rules)
		_, _, err := classifier.Classify("some text")
		Expect(err).To(HaveOccurred())
	})

	t.Run("Request context floor selects high band", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{count: 2}, rules)
		matched, count, err := classifier.ClassifyWithTokenFloor("ok", 5000)
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(5000))
		Expect(matched).To(ConsistOf("high"))
	})

	t.Run("Calibrated text estimate remains authoritative above floor", func(t *testing.T) {
		classifier := NewContextClassifier(&mockTokenCounter{count: 5000}, rules)
		matched, count, err := classifier.ClassifyWithTokenFloor("some text", 2)
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(5000))
		Expect(matched).To(ConsistOf("high"))
	})
}

// TestContextClassifierBandSemantics pins the boundary contract: bounds are
// inclusive, an equal min/max band matches exactly one count, and a band
// without max_tokens is open-ended.
func TestContextClassifierBandSemantics(t *testing.T) {
	rules := []config.ContextRule{
		{Name: "exact", MinTokens: "1K", MaxTokens: "1K"},
		{Name: "low", MinTokens: "0", MaxTokens: "1K"},
		{Name: "mid", MinTokens: "2K", MaxTokens: "4K"},
		{Name: "overflow", MinTokens: "8K"},
	}

	cases := []struct {
		name  string
		count int
		want  []string
	}{
		{name: "zero on low min", count: 0, want: []string{"low"}},
		{name: "inside low", count: 500, want: []string{"low"}},
		{name: "exactly on shared boundary", count: 1000, want: []string{"exact", "low"}},
		{name: "one above exact band", count: 1001, want: nil},
		{name: "between low and mid", count: 1500, want: nil},
		{name: "on mid min", count: 2000, want: []string{"mid"}},
		{name: "inside mid", count: 3000, want: []string{"mid"}},
		{name: "on mid max", count: 4000, want: []string{"mid"}},
		{name: "one above mid max", count: 4001, want: nil},
		{name: "between mid and overflow", count: 6000, want: nil},
		{name: "on overflow min", count: 8000, want: []string{"overflow"}},
		{name: "far above overflow min", count: 5_000_000, want: []string{"overflow"}},
		{name: "max int hits open-ended band", count: math.MaxInt, want: []string{"overflow"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			classifier := NewContextClassifier(&mockTokenCounter{count: tc.count}, rules)
			matched, count, err := classifier.Classify("text")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if count != tc.count {
				t.Fatalf("count = %d, want %d", count, tc.count)
			}
			if fmt.Sprint(matched) != fmt.Sprint(tc.want) {
				t.Fatalf("matched = %v, want %v", matched, tc.want)
			}
		})
	}
}

func TestContextClassifierOverlappingBandsReportInConfigOrder(t *testing.T) {
	rules := []config.ContextRule{
		{Name: "wide", MinTokens: "0", MaxTokens: "10K"},
		{Name: "narrow", MinTokens: "2K", MaxTokens: "3K"},
		{Name: "tail", MinTokens: "2.5K"},
	}
	classifier := NewContextClassifier(&mockTokenCounter{count: 2600}, rules)
	matched, _, err := classifier.Classify("text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"wide", "narrow", "tail"}
	if fmt.Sprint(matched) != fmt.Sprint(want) {
		t.Fatalf("matched = %v, want %v", matched, want)
	}
}

func TestContextClassifierSkipsInvalidRulesWithoutLosingOthers(t *testing.T) {
	rules := []config.ContextRule{
		{Name: "broken", MinTokens: "abc", MaxTokens: "1K"},
		{Name: "inverted", MinTokens: "5K", MaxTokens: "1K"},
		{Name: "no_limits"},
		{Name: "ok", MinTokens: "0", MaxTokens: "1K"},
	}
	classifier := NewContextClassifier(&mockTokenCounter{count: 500}, rules)
	matched, _, err := classifier.Classify("text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fmt.Sprint(matched) != fmt.Sprint([]string{"ok"}) {
		t.Fatalf("matched = %v, want [ok]", matched)
	}
}
