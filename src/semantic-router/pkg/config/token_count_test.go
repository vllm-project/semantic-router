package config

import (
	"testing"

	. "github.com/onsi/gomega"
)

func TestTokenCount(t *testing.T) {
	RegisterTestingT(t)

	tests := []struct {
		input    TokenCount
		expected int
		hasError bool
	}{
		{"1000", 1000, false},
		{"1K", 1000, false},
		{"1.5K", 1500, false},
		{"1M", 1000000, false},
		{"128K", 128000, false},
		{"0", 0, false},
		{"", 0, false},
		{"invalid", 0, true},
	}

	for _, tt := range tests {
		val, err := tt.input.Value()
		if tt.hasError {
			Expect(err).To(HaveOccurred())
		} else {
			Expect(err).NotTo(HaveOccurred())
			Expect(val).To(Equal(tt.expected))
		}
	}
}
