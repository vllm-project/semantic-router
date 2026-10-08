package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
)

// A dispatch error or panic after model selection returns from Process
// without running the receive-error cleanup. The top-level defer must still
// release the in-flight admission, against the bucket the request was
// admitted to, even when later routing rewrote the request model.
func TestReleaseInflightAdmissionReleasesLeakedToken(t *testing.T) {
	inflight.Reset()
	t.Cleanup(inflight.Reset)

	const primaryModel = "primary-model"
	const candidateModel = "candidate-model"

	token := inflight.Begin(primaryModel)
	if got := inflight.Get(primaryModel); got != 1 {
		t.Fatalf("admission not tracked after Begin: %d", got)
	}

	// Simulate a fallback hand-off that rewrote the routing model while the
	// original admission was still open.
	ctx := &RequestContext{
		RequestModel:  candidateModel,
		InflightModel: primaryModel,
		InflightToken: token,
	}

	releaseInflightAdmission(ctx)

	if got := inflight.Get(primaryModel); got != 0 {
		t.Fatalf("leaked admission still counted for %q: %d", primaryModel, got)
	}
	if ctx.InflightToken != 0 {
		t.Fatalf("token not cleared after release: %d", ctx.InflightToken)
	}
}

func TestReleaseInflightAdmissionIsIdempotentAndNilSafe(t *testing.T) {
	inflight.Reset()
	t.Cleanup(inflight.Reset)

	releaseInflightAdmission(nil)

	const model = "model-x"
	token := inflight.Begin(model)
	ctx := &RequestContext{InflightModel: model, InflightToken: token}
	releaseInflightAdmission(ctx)
	// A second release must not panic or corrupt the tracker (Begin issues
	// strictly increasing token ids, so End on a stale token is a no-op).
	releaseInflightAdmission(ctx)

	if got := inflight.Get(model); got != 0 {
		t.Fatalf("model count must stay zero after double release: %d", got)
	}
}

func TestReleaseInflightAdmissionOnlyTouchesAdmittedBucket(t *testing.T) {
	inflight.Reset()
	t.Cleanup(inflight.Reset)

	otherToken := inflight.Begin("other-model")
	defer func() {
		inflight.End("other-model", otherToken)
	}()

	token := inflight.Begin("admitted-model")
	ctx := &RequestContext{InflightModel: "admitted-model", InflightToken: token}
	releaseInflightAdmission(ctx)

	if got := inflight.Get("other-model"); got != 1 {
		t.Fatalf("unrelated model bucket must be untouched: %d", got)
	}
	if got := inflight.Get("admitted-model"); got != 0 {
		t.Fatalf("admitted bucket must be released: %d", got)
	}
}
