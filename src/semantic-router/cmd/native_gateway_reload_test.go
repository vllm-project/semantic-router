//go:build !windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func namedBackend(name string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Backend", name)
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
}

type servedResponse struct {
	status  int
	version uint64
	backend string
}

// Each activation alternates the backend, so a response whose configuration
// version and backend disagree was routed by one generation and sent by
// another.
func TestNativeGatewayHotReloadUnderLoadNeverSplitsARequest(t *testing.T) {
	one, two := namedBackend("one"), namedBackend("two")
	defer one.Close()
	defer two.Close()
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	// Other tests in this package leave a shut-down process default behind.
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(backendDocument(one.URL, "")), 0o600); err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(backendConfig(t, one.URL)),
		extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	engine := routing.DefaultOptions
	engine.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: engine}, Listener: "http-8899"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	var (
		stop      atomic.Bool
		mu        sync.Mutex
		responses []servedResponse
		workers   sync.WaitGroup
	)
	for range 6 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for !stop.Load() {
				resp, postErr := http.Post(front.URL+"/v1/chat/completions", "application/json",
					strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`))
				if postErr != nil {
					t.Error(postErr)
					return
				}
				_, _ = io.Copy(io.Discard, resp.Body)
				_ = resp.Body.Close()
				version, _ := strconv.ParseUint(resp.Header.Get("x-vsr-config-version"), 10, 64)
				mu.Lock()
				responses = append(responses, servedResponse{resp.StatusCode, version, resp.Header.Get("X-Backend")})
				mu.Unlock()
			}
		}()
	}
	for i := range 8 {
		time.Sleep(30 * time.Millisecond)
		backend := two.URL
		if i%2 == 1 {
			backend = one.URL
		}
		if err = server.ActivateKubernetesConfig(ctx, backendConfig(t, backend)); err != nil {
			t.Fatalf("reload %d: %v", i+1, err)
		}
	}
	time.Sleep(30 * time.Millisecond)
	stop.Store(true)
	workers.Wait()

	versions := map[uint64]int{}
	for _, r := range responses {
		want := "one"
		if r.version%2 == 0 {
			want = "two"
		}
		if r.status != http.StatusOK || r.version == 0 || r.backend != want {
			t.Fatalf("response %+v: want 200 from %s at a configuration version", r, want)
		}
		versions[r.version]++
	}
	if len(versions) < 3 {
		t.Fatalf("traffic crossed too few reloads to prove anything: %v", versions)
	}
	t.Logf("%d requests across versions %v", len(responses), versions)

	lease, err := server.Pin()
	if err != nil {
		t.Fatal(err)
	}
	last := upstreamOf(t, lease.Snapshot)
	lease.Release()
	cancel()
	<-served
	if err = server.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err = callUpstream(last); err == nil {
		t.Fatal("the serving upstream set outlived the Router's shutdown")
	}
}

// A key rotated through the management API reaches the running standalone
// listener without a restart: the old key gets 401 and the new one 200.
func TestManagementAPIRotatesAStandaloneListenersKeys(t *testing.T) {
	backend := namedBackend("one")
	defer backend.Close()
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	withKey := func(key string) string {
		return strings.Replace(backendDocument(backend.URL, ""), "    port: 8899\n", "    port: 8899\n    api_keys: ["+key+"]\n", 1)
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(withKey("old-key")), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(withKey("old-key")))
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(cfg)
	server, err := extproc.NewServer(path, 0, false, "", registry,
		extproc.WithGatewayMode(config.GatewayStandalone), extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
	defer func() { cancel(); <-served; _ = server.Shutdown(context.Background()) }()
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	free, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := free.Addr().(*net.TCPAddr).Port
	_ = free.Close()
	api, err := apiserver.StartWithOptions(apiserver.InitOptions{
		ConfigPath: path, Port: port, BindAddress: "127.0.0.1", RuntimeRegistry: registry, GatewayMode: config.GatewayStandalone,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = api.Shutdown(context.Background()) }()
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: routing.DefaultOptions}, Listener: "http-8899"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()
	chat := func(key string) int {
		request, _ := http.NewRequest(http.MethodPost, front.URL+"/v1/chat/completions",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`))
		request.Header.Set("Authorization", "Bearer "+key)
		response, chatErr := http.DefaultClient.Do(request)
		if chatErr != nil {
			t.Fatal(chatErr)
		}
		_, _ = io.Copy(io.Discard, response.Body)
		_ = response.Body.Close()
		return response.StatusCode
	}
	if status := chat("old-key"); status != http.StatusOK {
		t.Fatalf("the old key before the rotation got %d", status)
	}

	configURL := "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) + "/api/v1/config"
	read, _ := http.NewRequest(http.MethodGet, configURL, nil)
	current, err := http.DefaultClient.Do(read)
	if err != nil {
		t.Fatal(err)
	}
	_ = current.Body.Close()
	body, _ := json.Marshal(map[string]string{"yaml": withKey("new-key")})
	request, _ := http.NewRequest(http.MethodPut, configURL, bytes.NewReader(body))
	request.Header.Set("If-Match", current.Header.Get("ETag"))
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	var update struct {
		ActivationStatus string `json:"activation_status"`
	}
	_ = json.NewDecoder(response.Body).Decode(&update)
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK || update.ActivationStatus != "active" {
		t.Fatalf("the key rotation got %d, activation %q", response.StatusCode, update.ActivationStatus)
	}
	if old, rotated := chat("old-key"), chat("new-key"); old != http.StatusUnauthorized || rotated != http.StatusOK {
		t.Fatalf("after the rotation the old key got %d and the new key %d, want 401 and 200", old, rotated)
	}
}
