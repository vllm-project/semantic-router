package gatewayparity

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const (
	parityCorpusDir  = "../../pkg/routing/parity/testdata/corpus"
	parityRecordsDir = "../../pkg/extproc/testdata/parity/records"
)

// TestNativeGatewayServesTheParityCorpus sends the parity corpus through the
// native gateway composed as the Router composes it (routing core, upstream
// layer, frontend) against a recording fake backend. What the backend
// receives and what the client gets must match the records of the same
// corpus through the routing core, apart from the route-level and transport
// differences listed below.
func TestNativeGatewayServesTheParityCorpus(t *testing.T) {
	corpus, err := parity.LoadCorpus(parityCorpusDir)
	if err != nil {
		t.Fatal(err)
	}
	backend := parity.NewBackend(corpus.Cases)
	backendServer := httptest.NewServer(backend)
	defer backendServer.Close()
	configYAML, err := corpus.ConfigWithBackend(backendServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err = os.WriteFile(configPath, configYAML, 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := extproc.NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	}()
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: router.Config,
	})
	if err != nil {
		t.Fatal(err)
	}
	handler, err := gateway.NewHandler(gateway.Options{
		Serving: gateway.Static(gateway.Serving{
			Engine:   routing.NewEngine(extproc.NewRouterServiceForSnapshot(router, snapshot), routing.DefaultOptions),
			Upstream: set,
		}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	frontend := httptest.NewServer(handler)
	defer frontend.Close()

	records := parity.RecordWire(context.Background(), frontend.Client(), frontend.URL, backend, corpus.Cases)
	for _, got := range records {
		t.Run(got.Case, func(t *testing.T) {
			want := loadParityRecord(t, got.Case)
			if got.Error != "" {
				t.Fatalf("wire run failed: %s", got.Error)
			}
			assertSameClientResponse(t, want.Response, got.Response)
			assertSameUpstreamRequest(t, want, got)
			if code, ok := routingFailureCodes[got.Case]; ok {
				assertRoutingFailureCode(t, got.Response, code)
			}
		})
	}
}

// routingFailureCodes are the corpus cases the Router cannot route, with the
// reason code both modes must return for them.
var routingFailureCodes = map[string]string{
	"unknown-model":          "model_not_found",
	"unpublished-flow-model": "model_not_found",
}

func assertRoutingFailureCode(t *testing.T, response *parity.Message, code string) {
	t.Helper()
	if response.Status != 400 {
		t.Fatalf("status = %d, want 400", response.Status)
	}
	var body struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.Body, &body); err != nil {
		t.Fatalf("the error body is not JSON: %v: %s", err, response.Body)
	}
	if body.Error.Type != "invalid_request_error" || body.Error.Code != code {
		t.Fatalf("error = %+v, want an invalid_request_error with code %q", body.Error, code)
	}
}

func loadParityRecord(t *testing.T, name string) *parity.Record {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(parityRecordsDir, name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var record parity.Record
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	return &record
}

// Transport framing and the clock differ between an in-process record and
// the wire: the native server adds its date and may frame a buffered body
// with its own length.
var wireOnlyResponseHeaders = map[string]bool{"date": true, "content-length": true}

func assertSameClientResponse(t *testing.T, want, got *parity.Message) {
	t.Helper()
	if want.Status != got.Status {
		t.Fatalf("status = %d, want %d", got.Status, want.Status)
	}
	wantBody := string(want.Body)
	for _, chunk := range want.Chunks {
		wantBody += string(chunk)
	}
	if string(got.Body) != wantBody {
		t.Fatalf("client body differs:\nwant %s\ngot  %s", wantBody, got.Body)
	}
	for _, field := range want.Header {
		if field.Name == "content-length" {
			continue
		}
		if values := got.Header.Values(field.Name); len(values) == 0 || !contains(values, field.Value) {
			t.Fatalf("client header %s = %v, want %q", field.Name, values, field.Value)
		}
	}
	for _, field := range got.Header {
		if !want.Header.Has(field.Name) && !wireOnlyResponseHeaders[field.Name] {
			t.Fatalf("unexpected client header %s: %q", field.Name, field.Value)
		}
	}
}

// The upstream layer rewrites the authority to the backend's, as the
// template's host rewrite does; the scheme is not visible on the wire.
var routeRewrittenHeaders = map[string]bool{":authority": true, ":scheme": true}

func assertSameUpstreamRequest(t *testing.T, want, got *parity.Record) {
	t.Helper()
	if want.Upstream == nil {
		if received := got.Upstream; received != nil {
			t.Fatalf("the core answered this case itself, but the backend received %v", received.Header)
		}
		return
	}
	if got.Upstream == nil {
		t.Fatal("the backend received nothing")
	}
	if got.Route != want.Route {
		t.Fatalf("route = %q, want %q", got.Route, want.Route)
	}
	if string(got.Upstream.Body) != string(want.Upstream.Body) {
		t.Fatalf("upstream body differs:\nwant %s\ngot  %s", want.Upstream.Body, got.Upstream.Body)
	}
	for _, field := range want.Upstream.Header {
		if routeRewrittenHeaders[field.Name] {
			continue
		}
		if values := got.Upstream.Header.Values(field.Name); !contains(values, field.Value) {
			t.Fatalf("upstream header %s = %v, want %q", field.Name, values, field.Value)
		}
	}
	for _, field := range got.Upstream.Header {
		if !want.Upstream.Header.Has(field.Name) {
			t.Fatalf("unexpected upstream header %s: %q", field.Name, field.Value)
		}
	}
}

func contains(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}
