//go:build riscv64

package vectorstore

import (
	"context"
	"fmt"
)

var errValkeyGlideUnavailable = fmt.Errorf("valkey-glide native client is unavailable on linux/riscv64")

// ValkeyBackend is a compile-time stub on linux/riscv64.
type ValkeyBackend struct {
	collectionPrefix string
	indexM           int
	indexEf          int
	metricType       string
}

// NewValkeyBackend reports that valkey-glide is unavailable on linux/riscv64.
func NewValkeyBackend(ValkeyBackendConfig) (*ValkeyBackend, error) {
	return nil, errValkeyGlideUnavailable
}

func (*ValkeyBackend) CreateCollection(context.Context, string, int) error {
	return errValkeyGlideUnavailable
}

func (*ValkeyBackend) DeleteCollection(context.Context, string) error {
	return errValkeyGlideUnavailable
}

func (*ValkeyBackend) CollectionExists(context.Context, string) (bool, error) {
	return false, errValkeyGlideUnavailable
}

func (*ValkeyBackend) InsertChunks(context.Context, string, []EmbeddedChunk) error {
	return errValkeyGlideUnavailable
}

func (*ValkeyBackend) DeleteByFileID(context.Context, string, string) error {
	return errValkeyGlideUnavailable
}

func (*ValkeyBackend) Search(context.Context, string, []float32, int, float32, map[string]interface{}) ([]SearchResult, error) {
	return nil, errValkeyGlideUnavailable
}

func (*ValkeyBackend) Close() error { return nil }

var _ VectorStoreBackend = (*ValkeyBackend)(nil)
