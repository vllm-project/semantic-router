package managementserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

func TestImmutableRoutingRecipeReturnsProductiveConflict(t *testing.T) {
	response := httptest.NewRecorder()
	writeRoutingDomainError(response, routingmanagement.ErrImmutable, "request-id", true)
	if response.Code != http.StatusConflict {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusConflict)
	}
	if body := response.Body.String(); !strings.Contains(body, `"code":"immutable_resource"`) ||
		!strings.Contains(body, "Create a custom Recipe") {
		t.Fatalf("body = %s", body)
	}
}
