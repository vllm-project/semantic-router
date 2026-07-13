package embedding

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestOpenAICompatibleProviderClosesResponseBody(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		payload string
		wantErr bool
	}{
		{name: "success", status: http.StatusOK, payload: `{"data":[{"index":0,"embedding":[0.5]}]}`},
		{name: "invalid JSON", status: http.StatusOK, payload: `{"data":`, wantErr: true},
		{name: "provider error", status: http.StatusBadGateway, payload: "private upstream body", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := &closeTrackingBody{Reader: strings.NewReader(test.payload)}
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: test.status,
					Header:     make(http.Header),
					Body:       body,
				}, nil
			})}
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL: "https://embedding.example/v1", Model: "embedding-model",
				ExpectedDimension: 1, HTTPClient: client,
			})

			_, err := provider.Embed(context.Background(), "hello")
			if (err != nil) != test.wantErr {
				t.Fatalf("Embed() error = %v, wantErr=%v", err, test.wantErr)
			}
			if !body.closed {
				t.Fatal("Embed() did not close the response body")
			}
		})
	}
}

func TestOpenAICompatibleProviderReusesConnectionAfterErrorBodyDrain(t *testing.T) {
	var connections atomic.Int32
	var requests atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if requests.Add(1) == 1 {
			w.WriteHeader(http.StatusBadRequest)
			w.(http.Flusher).Flush()
			_, _ = io.WriteString(w, "private chunked upstream detail")
			return
		}
		writeEmbeddingResponse(t, w, [][]float64{{0.5}})
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	server.Start()
	defer server.Close()

	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConnsPerHost = 1
	defer transport.CloseIdleConnections()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
		HTTPClient: &http.Client{Transport: transport},
	})

	if _, err := provider.Embed(context.Background(), "first"); err == nil {
		t.Fatal("first Embed() returned nil error, want provider error")
	}
	if _, err := provider.Embed(context.Background(), "second"); err != nil {
		t.Fatalf("second Embed() error = %v", err)
	}
	if got := connections.Load(); got != 1 {
		t.Fatalf("provider opened %d connections, want one reused connection", got)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (roundTrip roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return roundTrip(request)
}

type closeTrackingBody struct {
	io.Reader
	closed bool
}

func (body *closeTrackingBody) Close() error {
	body.closed = true
	return nil
}
