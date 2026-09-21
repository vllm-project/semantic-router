//go:build riscv64

package cache

import (
	"context"
	"fmt"
	"strings"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

var errValkeyGlideUnavailable = fmt.Errorf("valkey-glide native client is unavailable on linux/riscv64")

// ValkeyCache is a compile-time stub on linux/riscv64. valkey-glide links
// libglide_ffi, which has no riscv64 artifact in this cut.
type ValkeyCache struct {
	embeddingProvider embedding.Provider
	config            *routerconfig.ValkeyConfig
	embeddingModel    string
	enabled           bool
	hitCount          int64
	missCount         int64
	lastCleanupTime   *time.Time
	searchFn          func(context.Context, []string) (any, error)
}

// ValkeyCacheOptions contains configuration parameters for Valkey cache initialization.
type ValkeyCacheOptions struct {
	EmbeddingProvider   embedding.Provider
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	Config              *routerconfig.ValkeyConfig
	EmbeddingModel      string
}

// NewValkeyCache returns a disabled stub, or an error when Valkey is requested.
func NewValkeyCache(options ValkeyCacheOptions) (*ValkeyCache, error) {
	if !options.Enabled {
		return &ValkeyCache{enabled: false}, nil
	}
	if options.Config == nil {
		return nil, fmt.Errorf("valkey config is required")
	}
	options.Config.Index.VectorField.MetricType = strings.ToUpper(options.Config.Index.VectorField.MetricType)
	return nil, errValkeyGlideUnavailable
}

func (c *ValkeyCache) getEmbedding(ctx context.Context, text string) ([]float32, error) {
	return computeCacheEmbedding(ctx, c.embeddingProvider, text)
}

func (c *ValkeyCache) embeddingDimension() int {
	if c == nil || c.config == nil {
		return semanticCacheEmbeddingDimension(0, "")
	}
	return semanticCacheEmbeddingDimension(c.config.Index.VectorField.Dimension, c.embeddingModel)
}

func (c *ValkeyCache) IsEnabled() bool { return c.enabled }

func (c *ValkeyCache) CheckConnection(context.Context) error {
	if !c.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (c *ValkeyCache) AddPendingRequest(string, string, string, []byte, int) error {
	if !c.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (c *ValkeyCache) UpdateWithResponse(string, []byte, int) error {
	if !c.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (c *ValkeyCache) AddEntry(context.Context, string, string, string, []byte, []byte, int) error {
	if !c.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (c *ValkeyCache) FindSimilar(string, string) ([]byte, bool, error) {
	return nil, false, nil
}

func (c *ValkeyCache) FindSimilarWithThreshold(string, string, float32) ([]byte, bool, error) {
	return nil, false, nil
}

func (c *ValkeyCache) LookupSimilarWithThreshold(context.Context, string, string, float32) (LookupResult, error) {
	return LookupResult{}, nil
}

func (c *ValkeyCache) FindExact(context.Context, string, string) (LookupResult, error) {
	return LookupResult{}, nil
}

func (c *ValkeyCache) AddExact(context.Context, string, string, []byte, int) error {
	if !c.enabled {
		return nil
	}
	return errValkeyGlideUnavailable
}

func (c *ValkeyCache) Close() error { return nil }

func (c *ValkeyCache) GetStats() CacheStats {
	hits := c.hitCount
	misses := c.missCount
	total := hits + misses
	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}
	stats := CacheStats{
		TotalEntries: 0,
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
	if c.lastCleanupTime != nil {
		stats.LastCleanupTime = c.lastCleanupTime
	}
	return stats
}
