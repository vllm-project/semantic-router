package gatewayparity

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// BenchmarkNativeGatewayClients sends the parity corpus's default-route
// request through the native gateway, composed as the Router composes it,
// from 1, 8, 32 and 64 closed-loop clients: the levels of the standalone
// latency record. It logs through a production-style sampled JSON logger.
// A contended lock or an allocation regression on the request path shows up
// as throughput that stops growing with clients.
func BenchmarkNativeGatewayClients(b *testing.B) {
	restore := zap.ReplaceGlobals(zap.New(zapcore.NewSamplerWithOptions(
		zapcore.NewCore(zapcore.NewJSONEncoder(zap.NewProductionEncoderConfig()), zapcore.AddSync(io.Discard), zapcore.InfoLevel),
		time.Second, 100, 100)))
	defer restore()
	corpus, err := parity.LoadCorpus(parityCorpusDir)
	if err != nil {
		b.Fatal(err)
	}
	var request parity.Case
	for _, c := range corpus.Cases {
		if c.Name == "auto-no-match-default-model" {
			request = c
		}
	}
	if request.Upstream == nil {
		b.Fatal("the corpus has no default-route case")
	}
	backend := httptest.NewServer(fixtureBackend(request.Upstream))
	defer backend.Close()
	frontend := nativeGateway(b, corpus, backend.URL)
	defer frontend.Close()
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 64}}
	defer client.CloseIdleConnections()

	for _, clients := range []int{1, 8, 32, 64} {
		b.Run(fmt.Sprintf("clients=%d", clients), func(b *testing.B) {
			latencies := make([]time.Duration, b.N)
			var next atomic.Int64
			var wg sync.WaitGroup
			b.ResetTimer()
			for range clients {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := next.Add(1) - 1; i < int64(b.N); i = next.Add(1) - 1 {
						start := time.Now()
						got, err := parity.Send(context.Background(), client, frontend.URL, request)
						if err != nil || got.Status != http.StatusOK {
							b.Errorf("request %d: %v, %v", i, got, err)
							return
						}
						latencies[i] = time.Since(start)
					}
				}()
			}
			wg.Wait()
			b.StopTimer()
			slices.Sort(latencies)
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "req/s")
			b.ReportMetric(float64(latencies[len(latencies)/2].Microseconds())/1000, "p50-ms")
			b.ReportMetric(float64(latencies[len(latencies)*99/100].Microseconds())/1000, "p99-ms")
		})
	}
}

// fixtureBackend answers every request with one canned upstream response and
// keeps nothing, so a long run holds no memory.
func fixtureBackend(fixture *parity.Upstream) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		for _, field := range fixture.Header() {
			w.Header().Add(field.Name, field.Value)
		}
		w.WriteHeader(fixture.Status)
		for _, part := range fixture.Parts() {
			_, _ = io.WriteString(w, part)
		}
	})
}

// nativeGateway composes the routing core, the upstream layer and the
// frontend as the Router's standalone mode does, over the corpus
// configuration pointed at backendURL.
func nativeGateway(b *testing.B, corpus *parity.Corpus, backendURL string) *httptest.Server {
	b.Helper()
	configYAML, err := corpus.ConfigWithBackend(backendURL)
	if err != nil {
		b.Fatal(err)
	}
	configPath := filepath.Join(b.TempDir(), "config.yaml")
	if err = os.WriteFile(configPath, configYAML, 0o600); err != nil {
		b.Fatal(err)
	}
	router, err := extproc.NewOpenAIRouter(configPath)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = router.Close() })
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	})
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: router.Config,
	})
	if err != nil {
		b.Fatal(err)
	}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{
		Serving: gateway.Static(gateway.Serving{
			Engine:   routing.NewEngine(extproc.NewRouterServiceForSnapshot(router, snapshot), opts),
			Upstream: set,
		}),
		Listener: "http-8899",
	})
	if err != nil {
		b.Fatal(err)
	}
	return httptest.NewServer(handler)
}
