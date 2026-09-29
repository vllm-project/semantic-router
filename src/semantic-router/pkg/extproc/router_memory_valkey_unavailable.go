//go:build riscv64

package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func createValkeyMemoryStore(*config.RouterConfig, ...*embedding.Set) (memory.Store, error) {
	return nil, fmt.Errorf("valkey-glide native client is unavailable on linux/riscv64")
}
