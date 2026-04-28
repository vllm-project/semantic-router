package tools_test

import (
	"context"
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// fakeRetriever is a test double for the Retriever interface.
type fakeRetriever struct {
	id     string
	result tools.RetrievalResult
	err    error
}

func (f *fakeRetriever) Retrieve(_ context.Context, _ string, _ int) (tools.RetrievalResult, error) {
	return f.result, f.err
}

var _ = Describe("Registry", func() {
	var reg *tools.Registry

	BeforeEach(func() {
		reg = tools.NewRegistry()
	})

	Describe("Get", func() {
		It("returns the named strategy when registered", func() {
			r := &fakeRetriever{id: "named"}
			reg.Register("named", r)

			got, ok := reg.Get("named")
			Expect(ok).To(BeTrue())
			Expect(got).To(Equal(r))
		})

		It("returns false when named strategy is absent, even if default exists", func() {
			def := &fakeRetriever{id: "default"}
			reg.Register("default", def)

			_, ok := reg.Get("nonexistent")
			Expect(ok).To(BeFalse())
		})

		It("returns false when neither named nor default is registered", func() {
			_, ok := reg.Get("anything")
			Expect(ok).To(BeFalse())
		})

		It("prefers the exact name over the default fallback", func() {
			exact := &fakeRetriever{id: "exact"}
			def := &fakeRetriever{id: "default"}
			reg.Register("exact", exact)
			reg.Register("default", def)

			got, ok := reg.Get("exact")
			Expect(ok).To(BeTrue())
			Expect(got).To(Equal(exact))
		})
	})

	Describe("Register", func() {
		It("panics when given a nil Retriever", func() {
			Expect(func() { reg.Register("bad", nil) }).To(Panic())
		})
	})
})

var _ = Describe("EmbeddingRetriever", func() {
	It("panics when constructed with a nil ToolsDatabase", func() {
		Expect(func() { tools.NewEmbeddingRetriever(nil) }).To(Panic())
	})

	It("returns zero confidence when the database returns no results", func() {
		db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
			Enabled:             false,
			SimilarityThreshold: 0.5,
		})
		r := tools.NewEmbeddingRetriever(db)
		result, err := r.Retrieve(context.Background(), "weather", 3)
		// disabled DB returns empty slice, not an error
		Expect(err).NotTo(HaveOccurred())
		Expect(result.Confidence).To(Equal(float32(0)))
		Expect(result.StrategyID).To(Equal("default"))
	})
})

var _ = Describe("fakeRetriever propagates errors", func() {
	It("surfaces the error returned by the underlying retriever", func() {
		boom := errors.New("retrieval failed")
		r := &fakeRetriever{err: boom}
		_, err := r.Retrieve(context.Background(), "query", 3)
		Expect(err).To(MatchError(boom))
	})
})
