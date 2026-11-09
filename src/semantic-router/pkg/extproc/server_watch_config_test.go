package extproc

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestShouldReloadForConfigEvent(t *testing.T) {
	cfgDir := "/app/config"
	cfgFile := filepath.Join(cfgDir, "config.yaml")

	tests := []struct {
		name      string
		eventPath string
		want      bool
	}{
		{
			name:      "exact config file path",
			eventPath: cfgFile,
			want:      true,
		},
		{
			name:      "config basename in watched dir",
			eventPath: filepath.Join(cfgDir, "config.yaml"),
			want:      true,
		},
		{
			name:      "configmap data symlink swap",
			eventPath: filepath.Join(cfgDir, "..data_tmp"),
			want:      true,
		},
		{
			name:      "startup status writability probe",
			eventPath: filepath.Join(cfgDir, ".vllm-sr-write-check-123"),
			want:      false,
		},
		{
			name:      "unrelated sibling file",
			eventPath: filepath.Join(cfgDir, "router-runtime.json"),
			want:      false,
		},
		{
			name:      "event in other directory",
			eventPath: "/tmp/config.yaml",
			want:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := shouldReloadForConfigEvent(cfgFile, cfgDir, tc.eventPath)
			if got != tc.want {
				t.Fatalf("shouldReloadForConfigEvent(%q) = %t, want %t", tc.eventPath, got, tc.want)
			}
		})
	}
}

// A configuration written the moment the server reports serving is reloaded
// in both gateway modes.
func TestConfigWrittenAsTheServerReportsServingIsReloaded(t *testing.T) {
	starts := map[string]func(*Server, context.Context, func()) error{
		"standalone": (*Server).StartContextWithoutExtProc,
		"extproc":    (*Server).StartContextWithReady,
	}
	for name, start := range starts {
		t.Run(name, func(t *testing.T) {
			defer stubReloadSeams(t)()
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			reloaded := make(chan struct{}, 1)
			parseReloadConfig = func(string) (*config.RouterConfig, error) {
				select {
				case reloaded <- struct{}{}:
				default:
				}
				return nil, errors.New("the reload began")
			}
			initial := &config.RouterConfig{}
			server := &Server{configPath: configPath, service: NewRouterService(&OpenAIRouter{Config: initial}), runtime: routerruntime.NewRegistry(initial)}

			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan error, 1)
			go func() {
				done <- start(server, ctx, func() {
					if err := os.WriteFile(configPath, []byte("version: v0.3\n# edited\n"), 0o600); err != nil {
						t.Error(err)
					}
				})
			}()
			select {
			case <-reloaded:
			case <-time.After(10 * time.Second):
				t.Error("the configuration written as the server reported serving was never reloaded")
			}
			cancel()
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			if err := server.lifecycle.stopAndWaitForBackgroundWork(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}
