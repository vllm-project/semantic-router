//go:build !windows

package main

import (
	"context"
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/http2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// heldBackend signals each chat completion on arrived and answers it once
// release closes.
func heldBackend(arrived chan<- struct{}, release <-chan struct{}) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		arrived <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"whole"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
}

// startDrainTestGateway serves the standalone listener of a one-model config
// on backendURL whose document ends with global, and returns the listener's
// URL, a stop that begins the Router's shutdown, and a channel closed when
// the gateway has drained.
func startDrainTestGateway(t *testing.T, backendURL, global string) (string, context.CancelFunc, <-chan struct{}) {
	t.Helper()
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	reservation, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(reservation.Addr().(*net.TCPAddr).Port)
	_ = reservation.Close()
	document := strings.Replace(backendDocument(backendURL, ""), "port: 8899", "port: "+port, 1) + global
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err = os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg),
		extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		if serveErr := startNativeGateway(ctx, server, &startupStateLog{}, "127.0.0.1"); serveErr != nil {
			t.Errorf("startNativeGateway() = %v", serveErr)
		}
	}()
	t.Cleanup(func() { cancel(); <-drained })
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	url := "http://127.0.0.1:" + port
	for deadline := time.Now().Add(readinessWait); ; time.Sleep(20 * time.Millisecond) {
		if resp, getErr := http.Get(url + "/ready"); getErr == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("listener /ready never answered 200")
		}
	}
	return url, cancel, drained
}

func postChatForDrain(client *http.Client, url string) string {
	resp, err := client.Post(url+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		return "error: " + err.Error()
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "read error: " + err.Error()
	}
	return strconv.Itoa(resp.StatusCode) + " " + string(body)
}

// An h2c request in flight when the Router begins to shut down is drained:
// the gateway's Shutdown alone returned at once for it, and the process then
// cut it. The drain ends as soon as the request does.
func TestStandaloneDrainWaitsForAnH2CRequestInFlight(t *testing.T) {
	arrived, release := make(chan struct{}, 1), make(chan struct{})
	backend := heldBackend(arrived, release)
	defer backend.Close()
	url, stop, drained := startDrainTestGateway(t, backend.URL, "")
	h2c := &http.Client{Transport: &http2.Transport{AllowHTTP: true, DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}}
	answered := make(chan string, 1)
	go func() { answered <- postChatForDrain(h2c, url) }()
	waitForTestSignal(t, arrived, "the request never reached the backend")

	stop()
	select {
	case <-drained:
		t.Fatal("the gateway stopped with an h2c request in flight")
	case <-time.After(500 * time.Millisecond):
	}
	finished := time.Now()
	close(release)
	select {
	case <-drained:
		if waited := time.Since(finished); waited > 5*time.Second {
			t.Fatalf("the drain ended %v after the request finished", waited)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("the drain did not end when the request finished")
	}
	if got := <-answered; !strings.HasPrefix(got, "200 ") || !strings.Contains(got, `"whole"`) {
		t.Fatalf("the h2c client got %q, want the whole answer", got)
	}
}

// global.router.drain_timeout_seconds bounds the drain: a request still in
// flight then is left to the process's exit.
func TestStandaloneDrainEndsAtTheConfiguredTimeout(t *testing.T) {
	arrived, release := make(chan struct{}, 1), make(chan struct{})
	backend := heldBackend(arrived, release)
	defer backend.Close()
	defer close(release)
	url, stop, drained := startDrainTestGateway(t, backend.URL, "global:\n  router:\n    drain_timeout_seconds: 1\n")
	go postChatForDrain(&http.Client{}, url)
	waitForTestSignal(t, arrived, "the request never reached the backend")

	begun := time.Now()
	stop()
	select {
	case <-drained:
		if waited := time.Since(begun); waited < time.Second || waited > 5*time.Second {
			t.Fatalf("the drain ended after %v, want its configured timeout of 1s", waited)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("the drain outlasted its configured timeout")
	}
}

// The process's shutdown budget is the drain and the resources' reserve after
// it: 30 s by default, as before the drain could be set, and a longer drain
// keeps its whole time.
func TestProcessShutdownBudgetCoversTheDrain(t *testing.T) {
	if got := processShutdownTimeout(nil); got != 30*time.Second {
		t.Fatalf("the default shutdown budget is %v, want 30s", got)
	}
	cfg := &config.RouterConfig{}
	cfg.DrainTimeoutSeconds = 180
	budget := processShutdownTimeout(cfg)
	if budget != 185*time.Second {
		t.Fatalf("a 180 s drain gets a %v shutdown budget, want 185s", budget)
	}
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	servingCtx, cancelServing := processServingDrainContext(ctx)
	defer cancelServing()
	deadline, _ := servingCtx.Deadline()
	if serving := time.Until(deadline); serving < cfg.DrainTimeout()-time.Second {
		t.Fatalf("a %v budget leaves the serving shutdown %v, less than the %v drain", budget, serving, cfg.DrainTimeout())
	}
}
