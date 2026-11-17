package upstream

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// The backend closes a connection after 5s idle, as uvicorn (vLLM's server)
// does by default, and the link delivers what the backend sends 300ms late,
// as a tunnel or a remote network does. A call sent in the last 300ms before
// the Router sees that close would go out on a connection the backend has
// already closed, and get no response.
func TestDoNeverSendsOnAConnectionTheBackendIsClosing(t *testing.T) {
	const keepAlive, lag = 5 * time.Second, 300 * time.Millisecond
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	server.Config.IdleTimeout = keepAlive
	server.Start()
	t.Cleanup(server.Close)
	dial := func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := (&net.Dialer{}).DialContext(ctx, network, address)
		if err != nil {
			return nil, err
		}
		return lagConn{Conn: conn, lag: lag}, nil
	}
	set := newSet(t, Options{Dial: dial}, clusterOf("keep-alive", endpointOf(t, "e", server)))

	for _, idle := range []time.Duration{0, keepAlive - lag/2} {
		time.Sleep(idle)
		resp, err := set.Do(t.Context(), post("keep-alive"))
		if err != nil {
			t.Fatalf("after %v idle: Do: %v", idle, err)
		}
		if body, err := readAll(t, resp); err != nil || resp.Local != nil || body != "ok" {
			t.Fatalf("after %v idle: status %d, body %q, failure %v", idle, resp.StatusCode, body, resp.Local)
		}
	}
}

// lagConn hands over what the backend sent lag after it arrived.
type lagConn struct {
	net.Conn
	lag time.Duration
}

func (c lagConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	time.Sleep(c.lag)
	return n, err
}
