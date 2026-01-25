package classification

import (
	"fmt"
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
}
