package parity

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestWireRecordsThroughAPassThroughGateway(t *testing.T) {
	cases := []Case{
		{
			Name:     "buffered",
			Request:  CaseRequest{Method: "POST", Path: "/v1/chat/completions?x=1", Headers: [][]string{{"Content-Type", "application/json"}}, Body: `{"model":"m"}`},
			Upstream: &Upstream{Status: 200, Headers: [][]string{{"content-type", "application/json"}}, Body: `{"created":99,"ok":true}`},
		},
		{
			Name:     "streamed",
			Request:  CaseRequest{Method: "POST", Path: "/v1/chat/completions", Body: `{"stream":true}`},
			Upstream: &Upstream{Status: 200, Headers: [][]string{{"content-type", "text/event-stream"}}, Chunks: []string{"data: a\n\n", "data: [DONE]\n\n"}},
		},
		{Name: "no-fixture", Request: CaseRequest{Method: "GET", Path: "/v1/models"}},
	}
	backend := NewBackend(cases)
	backendServer := httptest.NewServer(backend)
	defer backendServer.Close()
	target, _ := url.Parse(backendServer.URL)
	proxy := httputil.NewSingleHostReverseProxy(target)
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Header.Set("x-selected-model", "model-a")
		proxy.ServeHTTP(w, r)
	}))
	defer gateway.Close()

	records := RecordWire(context.Background(), gateway.Client(), gateway.URL, backend, cases)
	buffered, streamed, missing := records[0], records[1], records[2]
	if buffered.Error != "" || buffered.Response.Status != 200 || string(buffered.Response.Body) != `{"created":0,"ok":true}` {
		t.Fatalf("buffered record = %+v", buffered)
	}
	if buffered.Route != "model-a" || buffered.Upstream.Header.Get(":path") != "/v1/chat/completions?x=1" ||
		string(buffered.Upstream.Body) != `{"model":"m"}` || buffered.Upstream.Header.Get("x-request-id") != "parity-buffered" {
		t.Fatalf("buffered upstream = %+v", buffered.Upstream)
	}
	if string(streamed.Response.Body) != "data: a\n\ndata: [DONE]\n\n" {
		t.Fatalf("streamed body = %q", streamed.Response.Body)
	}
	if missing.Response.Status != http.StatusNotFound || missing.Upstream == nil {
		t.Fatalf("a case without a fixture is answered 404 and still recorded: %+v", missing)
	}
	headers := buffered.Response.Header
	for i := 1; i < len(headers); i++ {
		if headers[i-1].Name > headers[i].Name {
			t.Fatalf("wire headers must be sorted: %v", headers)
		}
	}
	stripped := buffered.WithoutHeaders("date", "content-length")
	if stripped.Response.Header.Has("date") || !buffered.Response.Header.Has("date") {
		t.Fatal("WithoutHeaders must strip a copy only")
	}
	backend.Reset()
	if len(backend.Received("buffered")) != 0 {
		t.Fatal("Reset must forget received requests")
	}
}

func TestConfigWithBackendPointsEveryProviderAtTheFake(t *testing.T) {
	corpus, err := LoadCorpus("testdata/corpus")
	if err != nil {
		t.Fatal(err)
	}
	rewritten, err := corpus.ConfigWithBackend("http://127.0.0.1:19999")
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Providers struct {
			Models []struct {
				BackendRefs []map[string]interface{} `yaml:"backend_refs"`
			} `yaml:"models"`
		} `yaml:"providers"`
	}
	if err := yaml.Unmarshal(rewritten, &document); err != nil {
		t.Fatal(err)
	}
	refs := 0
	for _, model := range document.Providers.Models {
		for _, ref := range model.BackendRefs {
			refs++
			endpoint, _ := ref["endpoint"].(string)
			baseURL, _ := ref["base_url"].(string)
			if endpoint != "127.0.0.1:19999" && !strings.HasPrefix(baseURL, "http://127.0.0.1:19999/") {
				t.Fatalf("backend ref not rewritten: %v", ref)
			}
		}
	}
	if refs == 0 {
		t.Fatal("the corpus has no backends")
	}
	if _, err := corpus.ConfigWithBackend("not a url"); err == nil {
		t.Fatal("an invalid backend URL must be rejected")
	}
}
