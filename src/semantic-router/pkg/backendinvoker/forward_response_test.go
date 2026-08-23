package backendinvoker

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestForwardResponseObserverReturnsCodecEncodedBody(t *testing.T) {
	body := io.NopCloser(strings.NewReader("encoded"))
	observed, err := (ForwardResponseObserver{}).Observe(
		context.Background(), Plan{}, AttemptResult{}, &http.Response{Body: body},
	)
	if err != nil || observed != body {
		t.Fatalf("Observe() = %v, %v", observed, err)
	}
	if _, err := (ForwardResponseObserver{}).Observe(context.Background(), Plan{}, AttemptResult{}, nil); err == nil {
		t.Fatal("empty response was accepted")
	}
}
