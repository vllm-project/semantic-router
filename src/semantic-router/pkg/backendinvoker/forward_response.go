package backendinvoker

import (
	"context"
	"fmt"
	"io"
	"net/http"
)

// ForwardResponseObserver preserves the protocol-adapted body for the outer
// ExtProc stream, which owns authoritative usage extraction and settlement.
// It exists as an explicit accounting hand-off rather than a silent nil/no-op
// observer at the private dispatch boundary.
type ForwardResponseObserver struct{}

var _ ResponseObserver = ForwardResponseObserver{}

func (ForwardResponseObserver) Observe(
	_ context.Context,
	_ Plan,
	_ AttemptResult,
	response *http.Response,
) (io.ReadCloser, error) {
	if response == nil || response.Body == nil {
		return nil, fmt.Errorf("backend response body is unavailable")
	}
	return response.Body, nil
}
