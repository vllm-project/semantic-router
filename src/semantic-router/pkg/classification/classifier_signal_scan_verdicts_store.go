package classification

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const sharedVerdictConcurrency = 16

// sharedVerdictStore owns one connection pool, outage state and bounded renewal
// queue per classifier generation. Recipe views share these resources only;
// their keys and in-process caches remain isolated.
type sharedVerdictStore struct {
	client         *redis.Client
	secret         []byte
	timeout        time.Duration
	retryAfter     atomic.Int64
	errorWarning   atomic.Int64
	invalidWarning atomic.Int64
	slots          chan struct{}
	ctx            context.Context
	cancel         context.CancelFunc
	worker         sync.WaitGroup
	closeOnce      sync.Once
	closeErr       error
	refreshMu      sync.Mutex
	refreshes      map[string]*sharedVerdictRefresh
	wake           chan struct{}
}

type sharedWindowVerdicts struct {
	*sharedVerdictStore
	prefix string
}

func (c *Classifier) needsSharedVerdicts() bool {
	if !c.Config.GuardVerdicts.Shared() {
		return false
	}
	for _, asked := range c.preparedSystemOneRules() {
		if guardScanned(asked) && asked.rule.EffectiveStage() == config.SystemOneStageRequest {
			return true
		}
	}
	return false
}

func (c *Classifier) initializeSharedVerdicts() error {
	if c.sharedVerdicts != nil || !c.needsSharedVerdicts() {
		return nil
	}
	store, err := newSharedWindowVerdicts(c.Config.GuardVerdicts, string(c.Config.RoutingScope))
	if err != nil {
		return err
	}
	c.sharedVerdicts, c.ownsSharedVerdicts = store, true
	return nil
}

func newSharedWindowVerdicts(cfg *config.GuardVerdictStoreConfig, recipe string) (*sharedWindowVerdicts, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if !cfg.Shared() {
		return nil, fmt.Errorf("shared guard verdicts require backend: redis")
	}
	secret, err := hex.DecodeString(os.Getenv(cfg.HMACKeyEnv))
	if err != nil || len(secret) != sha256.Size {
		return nil, fmt.Errorf("guard verdict hmac_key_env must name a secret containing 64 hex digits")
	}
	password := os.Getenv(cfg.Redis.PasswordEnv)
	if password == "" {
		return nil, fmt.Errorf("guard verdict redis.password_env must name a nonempty secret")
	}
	passwordKey, _ := hex.DecodeString(password)
	if hmac.Equal(passwordKey, secret) || hmac.Equal([]byte(password), secret) {
		return nil, fmt.Errorf("guard verdict HMAC key must not be the Redis password")
	}
	timeout := time.Duration(cfg.EffectiveTimeoutMs()) * time.Millisecond
	options := &redis.Options{
		Addr: cfg.Redis.Address, Username: cfg.Redis.Username, Password: password, DB: cfg.Redis.Database,
		DialTimeout: timeout, ReadTimeout: timeout, WriteTimeout: timeout, PoolTimeout: timeout,
		ContextTimeoutEnabled: true, MaxRetries: -1, DialerRetries: 1,
		PoolSize: sharedVerdictConcurrency, MaxActiveConns: sharedVerdictConcurrency,
	}
	if cfg.Redis.TLS {
		options.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}
	ctx, cancel := context.WithCancel(context.Background())
	store := &sharedVerdictStore{
		client: redis.NewClient(options), secret: secret, timeout: timeout,
		slots: make(chan struct{}, sharedVerdictConcurrency), ctx: ctx, cancel: cancel,
		refreshes: make(map[string]*sharedVerdictRefresh), wake: make(chan struct{}, 1),
	}
	store.worker.Add(1)
	go store.refreshLoop()
	return store.forRecipe(recipe), nil
}

func (s *sharedVerdictStore) forRecipe(recipe string) *sharedWindowVerdicts {
	scope := sha256.Sum256([]byte(recipe))
	return &sharedWindowVerdicts{sharedVerdictStore: s, prefix: fmt.Sprintf("vsr:guard-verdicts:v1:%x:", scope)}
}

func (s *sharedVerdictStore) Close() error {
	s.closeOnce.Do(func() {
		s.cancel()
		s.worker.Wait()
		s.closeErr = s.client.Close()
		s.refreshMu.Lock()
		clear(s.refreshes)
		s.refreshMu.Unlock()
	})
	return s.closeErr
}

func (s *sharedWindowVerdicts) available(ctx context.Context, op string) bool {
	if s == nil {
		return false
	}
	if ctx.Err() != nil || s.ctx.Err() != nil {
		metrics.RecordGuardVerdictStoreOp(op, "canceled")
		return false
	}
	if time.Now().UnixNano() < s.retryAfter.Load() {
		metrics.RecordGuardVerdictStoreOp(op, "cooldown")
		return false
	}
	return true
}

// A burst of bundle lookups waits within its lookup deadline, not in Redis's
// pool. Capacity exhaustion is a miss, never evidence that Redis is unhealthy.
func (s *sharedWindowVerdicts) acquire(ctx context.Context, op string) bool {
	select {
	case s.slots <- struct{}{}:
		return true
	case <-ctx.Done():
		metrics.RecordGuardVerdictStoreOp(op, "busy")
		return false
	}
}

func (s *sharedWindowVerdicts) failed(ctx context.Context, op string, err error) {
	if err == nil {
		return
	}
	metrics.RecordGuardVerdictStoreOp(op, "error")
	if ctx.Err() == nil && s.ctx.Err() == nil {
		s.retryAfter.Store(time.Now().Add(time.Second).UnixNano())
		s.warn(&s.errorWarning, "unavailable", op)
	}
}

func (s *sharedWindowVerdicts) warn(next *atomic.Int64, reason, op string) {
	now, previous := time.Now(), next.Load()
	if now.UnixNano() < previous || !next.CompareAndSwap(previous, now.Add(time.Minute).UnixNano()) {
		return
	}
	// No Redis errors, endpoints, keys or payloads: those can contain secrets
	// or attacker-controlled strings. Counters retain every failure.
	logging.ComponentWarnEvent("guard_verdicts", "shared_store_"+reason, map[string]interface{}{
		"operation": op, "fallback": "model_scan",
	})
}
