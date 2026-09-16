package extproc

import (
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// processBodyRoutingError turns any ProtocolError it can see into HTTP 400, so
// only client-owned categories may survive errors.As. A server fault that
// leaked through would be reported to the caller as their mistake.
func TestDispatchWireErrorKeepsServerCategoriesInternal(t *testing.T) {
	clientCategories := []llmprotocol.ErrorCategory{
		llmprotocol.ErrorInvalidRequest,
		llmprotocol.ErrorUnsupportedFeature,
	}
	serverCategories := []llmprotocol.ErrorCategory{
		llmprotocol.ErrorInternal,
		llmprotocol.ErrorUpstreamUnavailable,
		llmprotocol.ErrorUpstreamTimeout,
		llmprotocol.ErrorAuthentication,
		llmprotocol.ErrorPermission,
		llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict,
		llmprotocol.ErrorRateLimited,
	}

	for _, category := range clientCategories {
		t.Run("client/"+string(category), func(t *testing.T) {
			ctx := &RequestContext{}
			cause := llmprotocol.NewError(category, "code", "message", nil)

			got := dispatchWireError(cause, ctx, "encode provider request")

			var protocolError *llmprotocol.ProtocolError
			if !errors.As(got, &protocolError) {
				t.Fatalf("errors.As failed, so the clean 400 path cannot see it: %v", got)
			}
			if ctx.ImmediateProtocolError == nil {
				t.Fatal("ImmediateProtocolError was not recorded for a client error")
			}
		})
	}

	for _, category := range serverCategories {
		t.Run("server/"+string(category), func(t *testing.T) {
			ctx := &RequestContext{}
			cause := llmprotocol.NewError(category, "code", "message", nil)

			got := dispatchWireError(cause, ctx, "encode provider request")

			var protocolError *llmprotocol.ProtocolError
			if errors.As(got, &protocolError) {
				t.Fatalf("%s reached the HTTP 400 path", category)
			}
			if status.Code(got) != codes.Internal {
				t.Fatalf("gRPC code = %s, want Internal", status.Code(got))
			}
			if ctx.ImmediateProtocolError != nil {
				t.Fatalf("server fault was recorded as a client protocol error")
			}
		})
	}
}

// A plain error carries no category and must stay on the internal path.
func TestDispatchWireErrorWrapsNonProtocolError(t *testing.T) {
	ctx := &RequestContext{}

	got := dispatchWireError(fmt.Errorf("marshal failed"), ctx, "encode provider request")

	if status.Code(got) != codes.Internal {
		t.Fatalf("gRPC code = %s, want Internal", status.Code(got))
	}
	if ctx.ImmediateProtocolError != nil {
		t.Fatal("a plain error must not be recorded as a client protocol error")
	}
}
