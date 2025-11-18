package testcases

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// TestE2ETestcases is the entry point for the Ginkgo test suite
func TestE2ETestcases(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "E2E Testcases Suite")
}

// Suite-level setup
var _ = BeforeSuite(func() {
	// Initialize any suite-level resources
	// e.g., models, databases, etc.
})

// Suite-level cleanup
var _ = AfterSuite(func() {
	// Cleanup suite-level resources
})
