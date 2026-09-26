//go:build riscv64

package memory

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var errValkeyGlideUnavailable = fmt.Errorf("valkey-glide native client is unavailable on linux/riscv64")

// ValkeyStore is a compile-time stub on linux/riscv64.
type ValkeyStore struct {
	enabled bool
}

// ValkeyStoreOptions contains configuration for creating a ValkeyStore.
type ValkeyStoreOptions struct {
	Config          config.MemoryConfig
	ValkeyConfig    *config.MemoryValkeyConfig
	Enabled         bool
	EmbeddingConfig *EmbeddingConfig
}

// NewValkeyStore returns a disabled stub, or an error when Valkey is requested.
func NewValkeyStore(options ValkeyStoreOptions) (*ValkeyStore, error) {
	if !options.Enabled {
		return &ValkeyStore{enabled: false}, nil
	}
	return nil, errValkeyGlideUnavailable
}

func (v *ValkeyStore) Store(context.Context, *Memory) error { return errValkeyGlideUnavailable }

func (v *ValkeyStore) Retrieve(context.Context, RetrieveOptions) ([]*RetrieveResult, error) {
	return nil, errValkeyGlideUnavailable
}

func (v *ValkeyStore) Get(context.Context, string) (*Memory, error) {
	return nil, errValkeyGlideUnavailable
}

func (v *ValkeyStore) Update(context.Context, string, *Memory) error {
	return errValkeyGlideUnavailable
}

func (v *ValkeyStore) List(context.Context, ListOptions) (*ListResult, error) {
	return nil, errValkeyGlideUnavailable
}

func (v *ValkeyStore) Forget(context.Context, string) error { return errValkeyGlideUnavailable }

func (v *ValkeyStore) forgetIfCurrent(context.Context, memoryVersion) (bool, error) {
	return false, errValkeyGlideUnavailable
}

func (v *ValkeyStore) ForgetByScope(context.Context, MemoryScope) error {
	return errValkeyGlideUnavailable
}

func (v *ValkeyStore) IsEnabled() bool { return v.enabled }

func (v *ValkeyStore) CheckConnection(context.Context) error {
	if !v.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (v *ValkeyStore) Close() error { return nil }

var _ Store = (*ValkeyStore)(nil)
