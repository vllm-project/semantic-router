package classification

import (
	"context"
	"errors"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// These tests prove that an unavailable model backend cannot produce a
// successful routing or safety decision (issue #2620, follow-up to #2491).
//
// The model runtime fails closed with modelservice.ErrUnavailable. The batch
// analyzers tolerate per-item inference failures by design, but they must not
// report a benign "nothing detected" verdict when *no* content could be
// classified at all — that is indistinguishable from a clean scan to callers.

var _ = Describe("Consumer fail-closed behavior with an unavailable backend", func() {
	Context("jailbreak analysis", func() {
		var (
			classifier *Classifier
			mockModel  *MockJailbreakInference
		)

		BeforeEach(func() {
			classifier, _, mockModel = newTestJailbreakClassifier()
		})

		It("should not report a benign verdict when every item fails with an unavailable backend", func() {
			mockModel.classifyError = fmt.Errorf("jailbreak classification failed: %w", modelservice.ErrUnavailable)

			hasJailbreak, detections, err := classifier.AnalyzeContentForJailbreak(context.Background(), []string{"prompt one", "prompt two"})

			Expect(err).To(HaveOccurred(), "an unavailable backend must not yield a successful safety decision")
			Expect(errors.Is(err, modelservice.ErrUnavailable)).To(BeTrue(),
				"the backend-unavailable cause must be preserved for callers")
			Expect(hasJailbreak).To(BeFalse())
			Expect(detections).To(BeEmpty())
		})

		It("should still tolerate partial failures when at least one item is classified", func() {
			mockModel.setMockResponse("bad", 0, 0.9, modelservice.ErrUnavailable)
			mockModel.setMockResponse("good", 1, 0.9, nil)

			hasJailbreak, detections, err := classifier.AnalyzeContentForJailbreak(context.Background(), []string{"bad", "good"})

			Expect(err).ToNot(HaveOccurred())
			Expect(hasJailbreak).To(BeFalse())
			Expect(detections).To(HaveLen(1))
			Expect(detections[0].Content).To(Equal("good"))
		})

		It("should remain successful when only empty content is supplied", func() {
			hasJailbreak, detections, err := classifier.AnalyzeContentForJailbreak(context.Background(), []string{"", ""})

			Expect(err).ToNot(HaveOccurred())
			Expect(hasJailbreak).To(BeFalse())
			Expect(detections).To(BeEmpty())
		})
	})

	Context("PII analysis", func() {
		var (
			classifier *Classifier
			mockModel  *MockPIIInference
		)

		BeforeEach(func() {
			classifier, _, mockModel = newTestPIIClassifier()
		})

		It("should not report a clean verdict when every item fails with an unavailable backend", func() {
			mockModel.classifyTokensError = fmt.Errorf("pii classification failed: %w", modelservice.ErrUnavailable)

			hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{"my ssn is 123-45-6789", "call 555-1234"})

			Expect(err).To(HaveOccurred(), "an unavailable backend must not yield a successful PII decision")
			Expect(errors.Is(err, modelservice.ErrUnavailable)).To(BeTrue(),
				"the backend-unavailable cause must be preserved for callers")
			Expect(hasPII).To(BeFalse())
			Expect(results).To(BeEmpty())
		})

		It("should still tolerate partial failures when at least one item is classified", func() {
			mockModel.setMockResponse("bad", nil, modelservice.ErrUnavailable)
			mockModel.setMockResponse("clean", nil, nil)

			hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{"bad", "clean"})

			Expect(err).ToNot(HaveOccurred())
			Expect(hasPII).To(BeFalse())
			Expect(results).To(HaveLen(1))
			Expect(results[0].Content).To(Equal("clean"))
		})

		It("should remain successful when only empty content is supplied", func() {
			hasPII, results, err := classifier.AnalyzeContentForPII(context.Background(), []string{"", ""})

			Expect(err).ToNot(HaveOccurred())
			Expect(hasPII).To(BeFalse())
			Expect(results).To(BeEmpty())
		})
	})
})
