package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// TestRoutingSessionMatchesExtProcAdapter runs the corpus through the routing
// engine twice: over in-process routing sessions, as the native gateway does,
// and over the ext_proc gRPC adapter, as Envoy does. Every phase effect, the
// upstream request, its route and the client response must be identical.
func TestRoutingSessionMatchesExtProcAdapter(t *testing.T) {
	corpus, cfg := loadParityCorpus(t)
	viaSession := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	viaExtProc := parity.NewRecorder(&extprocStreamProcessor{router: newParityRouter(t, cfg)}, routing.DefaultOptions)
	for _, c := range corpus.Cases {
		t.Run(c.Name, func(t *testing.T) {
			got := viaSession.Run(context.Background(), c)
			want := viaExtProc.Run(context.Background(), c)
			if got.Error != "" || want.Error != "" {
				t.Fatalf("run errors: session %q, ext_proc %q", got.Error, want.Error)
			}
			encoded, err := got.Encode()
			if err != nil {
				t.Fatal(err)
			}
			checkParityGolden(t, parityRecordsGoldenDir, c.Name, encoded)
			// The ext_proc side cannot observe evidence; it is checked against
			// the response headers instead.
			assertEvidenceMatchesResponseHeaders(t, got)
			got.Evidence = routing.Evidence{}
			if diff := parity.Diff(want, got); diff != "" {
				t.Fatalf("routing session diverges from the ext_proc adapter:\n%s", diff)
			}
		})
	}
}

func assertEvidenceMatchesResponseHeaders(t *testing.T, record *parity.Record) {
	t.Helper()
	if record.Response == nil {
		return
	}
	checks := map[string]string{
		"x-vsr-selected-decision": record.Evidence.Decision,
		"x-vsr-selected-model":    record.Evidence.Model,
	}
	for header, evidence := range checks {
		if value := record.Response.Header.Get(header); value != "" && value != evidence {
			t.Fatalf("%s is %q but the evidence says %q", header, value, evidence)
		}
	}
}
