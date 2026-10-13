package upstream

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
)

// backend starts an HTTP test server and stops it with the test.
func backend(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func endpointOf(t *testing.T, name string, server *httptest.Server) EndpointSpec {
	t.Helper()
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	return EndpointSpec{Name: name, Scheme: u.Scheme, Host: u.Hostname(), Port: port, Weight: 1}
}

func clusterOf(name string, endpoints ...EndpointSpec) ClusterSpec {
	return ClusterSpec{Name: name, Endpoints: endpoints, LBPolicy: LBRoundRobin}
}

// newSet builds a Set over the clusters, the first serving the default
// route, and closes it with the test.
func newSet(t *testing.T, opts Options, clusters ...ClusterSpec) *Set {
	t.Helper()
	topology := Topology{Clusters: clusters}
	if len(clusters) > 0 {
		topology.DefaultCluster = clusters[0].Name
	}
	set, err := New(topology, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = set.Close(context.Background()) })
	return set
}

func post(routeKey string) *Request {
	return &Request{
		Method:   http.MethodPost,
		Path:     "/v1/chat/completions",
		Header:   http.Header{"Content-Type": {"application/json"}},
		Body:     []byte(`{"model":"m","messages":[]}`),
		RouteKey: routeKey,
	}
}

func withTimeouts(req *Request, timeouts Timeouts) *Request {
	req.Policy = &Policy{Timeouts: timeouts}
	return req
}

// readAll drains and closes a response body.
func readAll(t *testing.T, resp *Response) (string, error) {
	t.Helper()
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return string(data), err
}

// closedPort returns a loopback port with nothing listening on it.
func closedPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	return port
}

// awaitDisconnect blocks until the client goes away. The server only notices
// a closed connection once the request body is consumed.
func awaitDisconnect(r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	<-r.Context().Done()
}

// hijackAndClose answers by dropping the connection without a response.
func hijackAndClose(w http.ResponseWriter) {
	conn, _, err := http.NewResponseController(w).Hijack()
	if err == nil {
		_ = conn.Close()
	}
}
