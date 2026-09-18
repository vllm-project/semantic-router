package looper

import (
	"context"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"net/http"
	"strings"
	"testing"
)

func TestLooperChildCarriesParentConfigBinding(t *testing.T) {
	hash := strings.Repeat("a", 64)
	ctx := WithExpectedConfigHash(context.Background(), hash)
	h := http.Header{}
	setInternalRequestHeaders(h, ctx, CallOptions{})
	if h.Get(headers.SRBenchExpectedConfigHash) != hash {
		t.Fatal("looper child lost parent config binding")
	}
}
