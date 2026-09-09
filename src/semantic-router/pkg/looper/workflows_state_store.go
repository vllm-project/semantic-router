package looper

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultWorkflowStateRedisAddress = "localhost:6379"
	defaultWorkflowStateKeyPrefix    = "vllm-sr:flow:state:"
	defaultWorkflowStateFileDir      = "vllm-sr-flow-state"

	// maxStatePayloadBytes caps the serialised size of a single workflow
	// tool state entry. This prevents a runaway workflow with huge tool
	// trajectories from blowing up memory or Redis bandwidth.
	maxStatePayloadBytes = 512 * 1024 // 512 KiB

	// maxMemoryStateEntries caps how many in-flight states the memory
	// backend will hold. After cleanup, if the map is still at capacity
	// we reject the Put to prevent unbounded growth.
	maxMemoryStateEntries = 10_000

	// maxAggregateStateBytes caps the total serialized bytes stored across all
	// in-flight memory/file states to prevent OOMs even if entries are small.
	maxAggregateStateBytes = 100 * 1024 * 1024 // 100 MiB

	// workflowStateSweeperInterval is how often the background goroutine
	// proactively purges expired entries. Keeps memory/file state bounded
	// even when no new requests arrive.
	workflowStateSweeperInterval = 60 * time.Second
)

// workflowStateClaimLease is how long a resume may hold exclusive access
// before another request can recover the same durable state.
var (
	workflowStateClaimLeaseMu sync.RWMutex
	workflowStateClaimLease   = 30 * time.Second
)

func currentWorkflowStateClaimLease() time.Duration {
	workflowStateClaimLeaseMu.RLock()
	defer workflowStateClaimLeaseMu.RUnlock()
	return workflowStateClaimLease
}

type workflowStateClaim struct {
	Recipe config.RecipeName
	ID     string
	Token  string
	State  *workflowPendingToolState
}

type workflowToolStateStore interface {
	Put(ctx context.Context, state *workflowPendingToolState) (string, error)
	Claim(ctx context.Context, recipe config.RecipeName, id string) (*workflowStateClaim, bool, error)
	Commit(ctx context.Context, recipe config.RecipeName, id, token string) error
	Release(ctx context.Context, recipe config.RecipeName, id, token string) error
	Clear(ctx context.Context) error
	Close() error
}

type workflowStepResultState struct {
	Step             workflowPlanStep                   `json:"step"`
	Responses        []*ModelResponse                   `json:"responses,omitempty"`
	Failed           []FusionFailedModel                `json:"failed,omitempty"`
	ToolTrajectories map[string][]workflowAgentToolTurn `json:"tool_trajectories,omitempty"`
}

func (r workflowStepResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(workflowStepResultState{
		Step:             r.step,
		Responses:        r.responses,
		Failed:           r.failed,
		ToolTrajectories: cloneWorkflowToolTrajectories(r.toolTrajectories),
	})
}

func (r *workflowStepResult) UnmarshalJSON(data []byte) error {
	var state workflowStepResultState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	r.step = state.Step
	r.responses = state.Responses
	r.failed = state.Failed
	r.toolTrajectories = cloneWorkflowToolTrajectories(state.ToolTrajectories)
	return nil
}

func newWorkflowToolStateStoreFromConfig(flow config.FlowRuntimeConfig) workflowToolStateStore {
	stateCfg := flow.State.WithDefaults()
	switch stateCfg.StoreBackend {
	case config.WorkflowStateBackendMemory:
		return newWorkflowMemoryToolStateStore(stateCfg.TTL())
	case config.WorkflowStateBackendRedis:
		return newWorkflowRedisToolStateStore(stateCfg.Redis, stateCfg.TTL())
	case config.WorkflowStateBackendFile:
		return newWorkflowFileToolStateStore(stateCfg.File.Directory, stateCfg.TTL())
	default:
		logging.ComponentWarnEvent("looper", "workflow_state_backend_unknown", map[string]interface{}{
			"backend":  stateCfg.StoreBackend,
			"fallback": config.WorkflowStateBackendFile,
		})
		return newWorkflowFileToolStateStore(stateCfg.File.Directory, stateCfg.TTL())
	}
}

func normalizeWorkflowToolStateForStore(state *workflowPendingToolState) {
	if state.ID == "" {
		state.ID = newWorkflowToolStateID()
	}
	if state.CreatedAt.IsZero() {
		state.CreatedAt = time.Now().UTC()
	}
	state.RecipeName = string(normalizeWorkflowRecipeName(config.RecipeName(state.RecipeName)))
	state.ClaimToken = ""
	state.ClaimedAt = time.Time{}
}

func workflowToolStateExpired(state *workflowPendingToolState, ttl time.Duration, now time.Time) bool {
	if state == nil || ttl <= 0 {
		return false
	}
	return now.Sub(state.CreatedAt) > ttl
}

// checkPayloadSize rejects payloads that exceed the hard cap. Applied in
// every backend's Put path after json.Marshal so the limit is on wire bytes.
func checkPayloadSize(data []byte) error {
	if len(data) > maxStatePayloadBytes {
		return fmt.Errorf("workflow state payload %d bytes exceeds limit %d", len(data), maxStatePayloadBytes)
	}
	return nil
}

type memoryStateEntry struct {
	state        *workflowPendingToolState
	size         int64
	claimToken   string
	claimedUntil time.Time
}

type workflowMemoryToolStateStore struct {
	mu           sync.Mutex
	ttl          time.Duration
	states       map[string]memoryStateEntry
	currentBytes int64
	done         chan struct{}
	closeOnce    sync.Once
	wg           sync.WaitGroup
}

func newWorkflowMemoryToolStateStore(ttl time.Duration) *workflowMemoryToolStateStore {
	s := &workflowMemoryToolStateStore{
		ttl:    ttl,
		states: map[string]memoryStateEntry{},
		done:   make(chan struct{}),
	}
	s.wg.Add(1)
	go s.sweepLoop()
	return s
}

func (s *workflowMemoryToolStateStore) sweepLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(workflowStateSweeperInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case now := <-ticker.C:
			s.mu.Lock()
			s.cleanupLocked(now.UTC())
			s.mu.Unlock()
		}
	}
}

func (s *workflowMemoryToolStateStore) Put(_ context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	// Marshal early to enforce payload cap before touching the map.
	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked(time.Now().UTC())
	key, err := workflowNamespacedStateID(config.RecipeName(state.RecipeName), state.ID)
	if err != nil {
		return "", err
	}
	_, replacing := s.states[key]
	if !replacing && len(s.states) >= maxMemoryStateEntries {
		return "", fmt.Errorf("workflow memory state store at capacity (%d entries)", maxMemoryStateEntries)
	}
	var oldSize int64
	if old, exists := s.states[key]; exists {
		oldSize = old.size
	}
	if s.currentBytes+int64(len(data))-oldSize > maxAggregateStateBytes {
		return "", fmt.Errorf("workflow memory state store at capacity (%d bytes, max %d)", s.currentBytes+int64(len(data))-oldSize, maxAggregateStateBytes)
	}
	s.states[key] = memoryStateEntry{
		state: state,
		size:  int64(len(data)),
	}
	s.currentBytes += int64(len(data)) - oldSize
	return state.ID, nil
}

func (s *workflowMemoryToolStateStore) Claim(_ context.Context, recipe config.RecipeName, id string) (*workflowStateClaim, bool, error) {
	key, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		return nil, false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	s.cleanupLocked(now)
	if _, ok := s.states[id]; ok {
		return nil, false, errWorkflowStateUnscoped
	}
	entry, ok := s.states[key]
	if !ok {
		return nil, false, nil
	}
	if claimErr := workflowStateClaimable(entry.state, recipe); claimErr != nil {
		return nil, false, claimErr
	}
	if entry.claimToken != "" && now.Before(entry.claimedUntil) {
		return nil, false, nil
	}
	token := newWorkflowToolStateID()
	entry.claimToken = token
	entry.claimedUntil = now.Add(currentWorkflowStateClaimLease())
	s.states[key] = entry
	return &workflowStateClaim{
		Recipe: normalizeWorkflowRecipeName(recipe),
		ID:     id,
		Token:  token,
		State:  entry.state,
	}, true, nil
}

func (s *workflowMemoryToolStateStore) Commit(_ context.Context, recipe config.RecipeName, id, token string) error {
	key, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.states[key]
	if !ok {
		return nil
	}
	if entry.claimToken == token {
		s.currentBytes -= entry.size
		delete(s.states, key)
		return nil
	}
	if entry.claimToken == "" {
		return nil
	}
	return fmt.Errorf("workflow state %q claim is not held", id)
}

func (s *workflowMemoryToolStateStore) Release(_ context.Context, recipe config.RecipeName, id, token string) error {
	key, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.states[key]
	if !ok || entry.claimToken != token {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	entry.claimToken = ""
	entry.claimedUntil = time.Time{}
	s.states[key] = entry
	return nil
}

func (s *workflowMemoryToolStateStore) Clear(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.states = map[string]memoryStateEntry{}
	s.currentBytes = 0
	return nil
}

func (s *workflowMemoryToolStateStore) Close() error {
	s.closeOnce.Do(func() {
		close(s.done)
		s.wg.Wait()
	})
	return nil
}

func (s *workflowMemoryToolStateStore) cleanupLocked(now time.Time) {
	for id, entry := range s.states {
		if workflowToolStateExpired(entry.state, s.ttl, now) {
			s.currentBytes -= entry.size
			delete(s.states, id)
		}
	}
}

type workflowFileToolStateStore struct {
	mu           sync.Mutex
	dir          string
	ttl          time.Duration
	done         chan struct{}
	closeOnce    sync.Once
	wg           sync.WaitGroup
	currentBytes int64
}

func cleanupStateStoreDirAndGetInitialBytes(storeDir string) int64 {
	var initialBytes int64
	entries, err := os.ReadDir(storeDir)
	if err != nil {
		return 0
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			continue
		}
		path := filepath.Join(storeDir, name)
		if strings.Contains(name, ".tmp-") {
			_ = os.Remove(path)
			continue
		}
		if strings.Contains(name, ".take-") {
			recoverInterruptedFileTake(storeDir, name)
			continue
		}
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		initialBytes += info.Size()
	}
	return initialBytes
}

func recoverInterruptedFileTake(storeDir, name string) {
	takePath := filepath.Join(storeDir, name)
	jsonName, _, found := strings.Cut(name, ".take-")
	if !found || !strings.HasSuffix(jsonName, ".json") {
		_ = os.Remove(takePath)
		return
	}
	jsonPath := filepath.Join(storeDir, jsonName)
	if _, err := os.Stat(jsonPath); err == nil {
		_ = os.Remove(takePath)
		return
	}
	_ = os.Rename(takePath, jsonPath)
}

func (s *workflowFileToolStateStore) sweepLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(workflowStateSweeperInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case now := <-ticker.C:
			s.cleanupExpired(now.UTC())
		}
	}
}

func workflowStateFileDir(dir string) string {
	if strings.TrimSpace(dir) != "" {
		return dir
	}
	base, err := os.UserCacheDir()
	if err != nil || strings.TrimSpace(base) == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, defaultWorkflowStateFileDir)
}

func (s *workflowFileToolStateStore) Put(_ context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return "", fmt.Errorf("create workflow state directory: %w", err)
	}

	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
	}

	path, err := s.pathForID(config.RecipeName(state.RecipeName), state.ID)
	if err != nil {
		return "", err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.cleanupExpiredLocked(time.Now().UTC())

	var oldSize int64
	if info, statErr := os.Stat(path); statErr == nil {
		oldSize = info.Size()
	}

	if s.currentBytes+int64(len(data))-oldSize > maxAggregateStateBytes {
		return "", fmt.Errorf("workflow file state store at capacity (%d bytes, max %d)", s.currentBytes+int64(len(data))-oldSize, maxAggregateStateBytes)
	}

	tmp := path + ".tmp-" + newWorkflowToolStateID()
	if writeErr := os.WriteFile(tmp, data, 0o600); writeErr != nil {
		return "", fmt.Errorf("write workflow state: %w", writeErr)
	}
	if renameErr := os.Rename(tmp, path); renameErr != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("commit workflow state: %w", renameErr)
	}

	committedInfo, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("stat workflow state: %w", err)
	}
	s.currentBytes += committedInfo.Size() - oldSize
	return state.ID, nil
}

func (s *workflowFileToolStateStore) Claim(_ context.Context, recipe config.RecipeName, id string) (*workflowStateClaim, bool, error) {
	path, err := s.pathForID(recipe, id)
	if err != nil {
		return nil, false, err
	}
	legacyPath, err := s.legacyPathForID(id)
	if err != nil {
		return nil, false, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	s.cleanupExpiredLocked(now)

	if _, statErr := os.Stat(legacyPath); statErr == nil {
		if _, namespacedErr := os.Stat(path); errors.Is(namespacedErr, os.ErrNotExist) {
			return nil, false, errWorkflowStateUnscoped
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("read workflow state: %w", err)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, false, fmt.Errorf("parse workflow state: %w", err)
	}
	if workflowToolStateExpired(&state, s.ttl, now) {
		return nil, false, nil
	}
	if claimErr := workflowStateClaimable(&state, recipe); claimErr != nil {
		return nil, false, claimErr
	}
	if state.ClaimToken != "" && !state.ClaimedAt.IsZero() && now.Sub(state.ClaimedAt) < currentWorkflowStateClaimLease() {
		return nil, false, nil
	}
	token := newWorkflowToolStateID()
	state.ClaimToken = token
	state.ClaimedAt = now
	if err := s.writeStateLocked(path, &state); err != nil {
		return nil, false, err
	}
	state.ClaimToken = ""
	state.ClaimedAt = time.Time{}
	return &workflowStateClaim{
		Recipe: normalizeWorkflowRecipeName(recipe),
		ID:     id,
		Token:  token,
		State:  &state,
	}, true, nil
}

func (s *workflowFileToolStateStore) Commit(_ context.Context, recipe config.RecipeName, id, token string) error {
	path, err := s.pathForID(recipe, id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, info, err := s.readStateLocked(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if state.ClaimToken == token {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("commit workflow state: %w", err)
		}
		s.currentBytes -= info.Size()
		if s.currentBytes < 0 {
			s.currentBytes = 0
		}
		return nil
	}
	if state.ClaimToken == "" {
		return nil
	}
	return fmt.Errorf("workflow state %q claim is not held", id)
}

func (s *workflowFileToolStateStore) Release(_ context.Context, recipe config.RecipeName, id, token string) error {
	path, err := s.pathForID(recipe, id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, _, err := s.readStateLocked(path)
	if err != nil {
		return err
	}
	if state.ClaimToken != token {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	state.ClaimToken = ""
	state.ClaimedAt = time.Time{}
	return s.writeStateLocked(path, state)
}

func (s *workflowFileToolStateStore) readStateLocked(path string) (*workflowPendingToolState, os.FileInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, nil, fmt.Errorf("stat workflow state: %w", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read workflow state: %w", err)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, nil, fmt.Errorf("parse workflow state: %w", err)
	}
	return &state, info, nil
}

func (s *workflowFileToolStateStore) writeStateLocked(path string, state *workflowPendingToolState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return sizeErr
	}
	var oldSize int64
	if info, statErr := os.Stat(path); statErr == nil {
		oldSize = info.Size()
	}
	tmp := path + ".tmp-" + newWorkflowToolStateID()
	if writeErr := os.WriteFile(tmp, data, 0o600); writeErr != nil {
		return fmt.Errorf("write workflow state: %w", writeErr)
	}
	if renameErr := os.Rename(tmp, path); renameErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("commit workflow state: %w", renameErr)
	}
	committedInfo, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat workflow state: %w", err)
	}
	s.currentBytes += committedInfo.Size() - oldSize
	return nil
}

func (s *workflowFileToolStateStore) Clear(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	entries, err := os.ReadDir(s.dir)
	if errors.Is(err, os.ErrNotExist) {
		s.currentBytes = 0
		return nil
	}
	if err != nil {
		return fmt.Errorf("list workflow states: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") && !strings.Contains(name, ".take-") && !strings.Contains(name, ".tmp-") {
			continue
		}
		path := filepath.Join(s.dir, name)
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove workflow state %s: %w", name, err)
		}
	}
	s.currentBytes = 0
	return nil
}

func (s *workflowFileToolStateStore) stopSweeper() error {
	s.closeOnce.Do(func() {
		close(s.done)
		s.wg.Wait()
	})
	return nil
}

func (s *workflowFileToolStateStore) cleanupExpired(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
}

func (s *workflowFileToolStateStore) cleanupExpiredLocked(now time.Time) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(s.dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var state workflowPendingToolState
		if err := json.Unmarshal(data, &state); err != nil || workflowToolStateExpired(&state, s.ttl, now) {
			if err := os.Remove(path); err == nil {
				s.currentBytes -= info.Size()
				if s.currentBytes < 0 {
					s.currentBytes = 0
				}
			}
		}
	}
}

func (s *workflowFileToolStateStore) pathForID(recipe config.RecipeName, id string) (string, error) {
	namespaced, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		return "", err
	}
	return filepath.Join(s.dir, namespaced+".json"), nil
}

func (s *workflowFileToolStateStore) legacyPathForID(id string) (string, error) {
	if !validWorkflowStateID(id) {
		return "", fmt.Errorf("invalid workflow state id %q", id)
	}
	return filepath.Join(s.dir, id+".json"), nil
}

func (s *workflowFileToolStateStore) replaceTTL(ttl time.Duration) {
	s.mu.Lock()
	s.ttl = ttl
	s.mu.Unlock()
}

func validWorkflowStateID(id string) bool {
	if id == "" {
		return false
	}
	for _, ch := range id {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			continue
		}
		return false
	}
	return true
}

type workflowRedisToolStateStore struct {
	client    *redis.Client
	keyPrefix string
	ttl       time.Duration
	closeOnce sync.Once
}

func newWorkflowRedisToolStateStore(cfg config.WorkflowStateRedisConfig, ttl time.Duration) *workflowRedisToolStateStore {
	address := strings.TrimSpace(cfg.Address)
	if address == "" {
		address = defaultWorkflowStateRedisAddress
	}
	keyPrefix := strings.TrimSpace(cfg.KeyPrefix)
	if keyPrefix == "" {
		keyPrefix = defaultWorkflowStateKeyPrefix
	}
	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = 10
	}
	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}
	opts := &redis.Options{
		Addr:       address,
		DB:         cfg.DB,
		Password:   cfg.Password,
		PoolSize:   poolSize,
		MaxRetries: maxRetries,
	}
	if cfg.UseTLS {
		opts.TLSConfig = &tls.Config{InsecureSkipVerify: cfg.TLSSkipVerify}
	}
	return &workflowRedisToolStateStore{
		client:    redis.NewClient(opts),
		keyPrefix: keyPrefix,
		ttl:       ttl,
	}
}

func (s *workflowRedisToolStateStore) Put(ctx context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
	}
	if err := s.client.Set(ctx, s.key(config.RecipeName(state.RecipeName), state.ID), data, s.ttl).Err(); err != nil {
		return "", fmt.Errorf("store workflow state in redis: %w", err)
	}
	_ = s.client.Del(ctx, s.claimKey(config.RecipeName(state.RecipeName), state.ID)).Err()
	return state.ID, nil
}

func (s *workflowRedisToolStateStore) Claim(ctx context.Context, recipe config.RecipeName, id string) (*workflowStateClaim, bool, error) {
	if !validWorkflowStateID(id) {
		return nil, false, fmt.Errorf("invalid workflow state id %q", id)
	}
	data, err := s.client.Get(ctx, s.key(recipe, id)).Bytes()
	if errors.Is(err, redis.Nil) {
		legacy, legacyErr := s.client.Exists(ctx, s.legacyKey(id)).Result()
		if legacyErr != nil {
			return nil, false, fmt.Errorf("inspect legacy workflow state in redis: %w", legacyErr)
		}
		if legacy > 0 {
			return nil, false, errWorkflowStateUnscoped
		}
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read workflow state from redis: %w", err)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, false, fmt.Errorf("parse workflow state: %w", err)
	}
	if claimErr := workflowStateClaimable(&state, recipe); claimErr != nil {
		return nil, false, claimErr
	}
	now := time.Now().UTC()
	if workflowToolStateExpired(&state, s.ttl, now) {
		_ = s.client.Del(ctx, s.key(recipe, id), s.claimKey(recipe, id)).Err()
		return nil, false, nil
	}
	token := newWorkflowToolStateID()
	claimedJSON, err := marshalWorkflowStateWithClaim(&state, token, now)
	if err != nil {
		return nil, false, err
	}
	result, err := workflowRedisClaimScript.Run(
		ctx,
		s.client,
		[]string{s.key(recipe, id), s.claimKey(recipe, id)},
		token,
		claimedJSON,
		strconv.FormatInt(currentWorkflowStateClaimLease().Milliseconds(), 10),
	).Result()
	if errors.Is(err, redis.Nil) || result == nil {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("claim workflow state from redis: %w", err)
	}
	snapshot, ok := workflowRedisScriptBytes(result)
	if !ok {
		return nil, false, fmt.Errorf("claim workflow state from redis returned %T", result)
	}
	if unmarshalErr := json.Unmarshal(snapshot, &state); unmarshalErr != nil {
		return nil, false, fmt.Errorf("parse workflow state: %w", unmarshalErr)
	}
	if claimErr := workflowStateClaimable(&state, recipe); claimErr != nil {
		_ = s.Release(ctx, recipe, id, token)
		return nil, false, claimErr
	}
	state.ClaimToken = ""
	state.ClaimedAt = time.Time{}
	return &workflowStateClaim{
		Recipe: normalizeWorkflowRecipeName(recipe),
		ID:     id,
		Token:  token,
		State:  &state,
	}, true, nil
}

func (s *workflowRedisToolStateStore) Commit(ctx context.Context, recipe config.RecipeName, id, token string) error {
	result, err := workflowRedisCommitScript.Run(
		ctx,
		s.client,
		[]string{s.key(recipe, id), s.claimKey(recipe, id)},
		token,
	).Result()
	if err != nil {
		return fmt.Errorf("commit workflow state in redis: %w", err)
	}
	accepted, _ := result.(int64)
	if accepted != 1 {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	return nil
}

func (s *workflowRedisToolStateStore) Release(ctx context.Context, recipe config.RecipeName, id, token string) error {
	data, err := s.client.Get(ctx, s.key(recipe, id)).Bytes()
	if errors.Is(err, redis.Nil) {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	if err != nil {
		return fmt.Errorf("read workflow state from redis: %w", err)
	}
	var state workflowPendingToolState
	if parseErr := json.Unmarshal(data, &state); parseErr != nil {
		return fmt.Errorf("parse workflow state: %w", parseErr)
	}
	if state.ClaimToken != token {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	released, err := marshalWorkflowStateWithClaim(&state, "", time.Time{})
	if err != nil {
		return err
	}
	result, err := workflowRedisReleaseScript.Run(
		ctx,
		s.client,
		[]string{s.key(recipe, id), s.claimKey(recipe, id)},
		token,
		released,
	).Result()
	if err != nil {
		return fmt.Errorf("release workflow state in redis: %w", err)
	}
	accepted, _ := result.(int64)
	if accepted != 1 {
		return fmt.Errorf("workflow state %q claim is not held", id)
	}
	return nil
}

func (s *workflowRedisToolStateStore) Clear(ctx context.Context) error {
	var cursor uint64
	for {
		keys, next, err := s.client.Scan(ctx, cursor, s.keyPrefix+"*", 100).Result()
		if err != nil {
			return fmt.Errorf("scan workflow states in redis: %w", err)
		}
		if len(keys) > 0 {
			if err := s.client.Del(ctx, keys...).Err(); err != nil {
				return fmt.Errorf("clear workflow states in redis: %w", err)
			}
		}
		if next == 0 {
			return nil
		}
		cursor = next
	}
}

func (s *workflowRedisToolStateStore) Close() error {
	var err error
	s.closeOnce.Do(func() {
		if s.client != nil {
			err = s.client.Close()
		}
	})
	return err
}

func (s *workflowRedisToolStateStore) key(recipe config.RecipeName, id string) string {
	namespaced, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		return s.keyPrefix + id
	}
	return s.keyPrefix + namespaced
}

func (s *workflowRedisToolStateStore) claimKey(recipe config.RecipeName, id string) string {
	return s.key(recipe, id) + ":claim"
}

func (s *workflowRedisToolStateStore) legacyKey(id string) string {
	return s.keyPrefix + id
}

func workflowRedisScriptBytes(value interface{}) ([]byte, bool) {
	switch typed := value.(type) {
	case string:
		return []byte(typed), true
	case []byte:
		return typed, true
	default:
		return nil, false
	}
}

func marshalWorkflowStateWithClaim(state *workflowPendingToolState, token string, claimedAt time.Time) ([]byte, error) {
	if state == nil {
		return nil, fmt.Errorf("workflow tool state missing")
	}
	clone := *state
	clone.ClaimToken = token
	clone.ClaimedAt = claimedAt
	data, err := json.Marshal(&clone)
	if err != nil {
		return nil, fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return nil, sizeErr
	}
	return data, nil
}

var workflowRedisClaimScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
if not value then
  return false
end
local live = redis.call("GET", KEYS[2])
if live then
  local needle = '"claim_token":"' .. live .. '"'
  if string.find(value, needle, 1, true) then
    return false
  end
end
local ttl = redis.call("PTTL", KEYS[1])
redis.call("SET", KEYS[1], ARGV[2])
if ttl and ttl > 0 then
  redis.call("PEXPIRE", KEYS[1], ttl)
end
redis.call("SET", KEYS[2], ARGV[1], "PX", tonumber(ARGV[3]))
return value
`)

var workflowRedisCommitScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
local claim = redis.call("GET", KEYS[2])
local needle = '"claim_token":"' .. ARGV[1] .. '"'
if not value then
  redis.call("DEL", KEYS[2])
  return 1
end
if string.find(value, needle, 1, true) then
  redis.call("DEL", KEYS[1], KEYS[2])
  return 1
end
if claim == ARGV[1] or not claim then
  redis.call("DEL", KEYS[2])
  return 1
end
return 0
`)

var workflowRedisReleaseScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
if not value then
  return 0
end
local needle = '"claim_token":"' .. ARGV[1] .. '"'
if not string.find(value, needle, 1, true) then
  return 0
end
local ttl = redis.call("PTTL", KEYS[1])
redis.call("SET", KEYS[1], ARGV[2])
if ttl and ttl > 0 then
  redis.call("PEXPIRE", KEYS[1], ttl)
end
redis.call("DEL", KEYS[2])
return 1
`)
