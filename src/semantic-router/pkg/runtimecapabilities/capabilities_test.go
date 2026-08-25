package runtimecapabilities

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDeriveCapabilityMatrix(t *testing.T) {
	managementStore := func() *config.AccessStoreConfig { return &config.AccessStoreConfig{} }
	runtimeStore := func() *config.AccessRuntimeStoreConfig { return &config.AccessRuntimeStoreConfig{} }

	tests := []struct {
		name      string
		configure func(*config.RouterConfig)
		want      RuntimeCapabilities
		wantErr   string
	}{
		{
			name: "file authority",
			want: RuntimeCapabilities{FileRouting: true},
		},
		{
			name: "durable routing authority",
			configure: func(cfg *config.RouterConfig) {
				cfg.AccessStore = managementStore()
			},
			want: RuntimeCapabilities{DurableRouting: true},
		},
		{
			name: "management API",
			configure: func(cfg *config.RouterConfig) {
				cfg.AccessStore = managementStore()
				cfg.ManagementAPI.Enabled = true
			},
			want: RuntimeCapabilities{DurableRouting: true, ManagementAPI: true},
		},
		{
			name: "native access",
			configure: func(cfg *config.RouterConfig) {
				cfg.AccessStore = managementStore()
				cfg.AccessRuntimeStore = runtimeStore()
				cfg.Access.Enabled = true
			},
			want: RuntimeCapabilities{
				DurableRouting: true, DistributedState: true, NativeAccess: true,
			},
		},
		{
			name: "management API without authority",
			configure: func(cfg *config.RouterConfig) {
				cfg.ManagementAPI.Enabled = true
			},
			wantErr: "management_api.enabled requires",
		},
		{
			name: "runtime store without authority",
			configure: func(cfg *config.RouterConfig) {
				cfg.AccessRuntimeStore = runtimeStore()
			},
			wantErr: "stores.runtime.redis requires",
		},
		{
			name: "access without runtime store",
			configure: func(cfg *config.RouterConfig) {
				cfg.AccessStore = managementStore()
				cfg.Access.Enabled = true
			},
			wantErr: "access.enabled requires",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			if test.configure != nil {
				test.configure(&cfg)
			}
			got, err := Derive(&cfg)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("Derive() error = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Derive() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("Derive() = %+v, want %+v", got, test.want)
			}
		})
	}
}

func TestDeriveRejectsNilConfig(t *testing.T) {
	if _, err := Derive(nil); err == nil {
		t.Fatal("Derive(nil) unexpectedly succeeded")
	}
}
