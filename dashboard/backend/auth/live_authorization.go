package auth

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"
)

const defaultLiveAuthorizationRecheckInterval = 15 * time.Second

const (
	maxLiveAuthorizationWatchers           = 2048
	maxLiveAuthorizationWatchersPerUser    = 64
	maxLiveAuthorizationWatchersPerSession = 16
)

var (
	ErrLiveAuthorizationInvalid  = errors.New("live authorization is no longer valid")
	ErrLivePermissionDenied      = errors.New("live authorization permission denied")
	ErrLiveAuthorizationCapacity = errors.New("live authorization capacity reached")
)

type liveAuthorizationContextKey struct{}

type liveAuthorizationState struct {
	service          *Service
	claims           TokenClaims
	credentialSource CredentialSource
}

type authorizationWatcher struct {
	id        uint64
	userID    string
	sessionID string
	cancel    func(error)
}

type liveAuthorizationRegistry struct {
	mu              sync.Mutex
	nextID          uint64
	watchers        map[uint64]*authorizationWatcher
	recheckInterval time.Duration
}

func newLiveAuthorizationRegistry() *liveAuthorizationRegistry {
	return &liveAuthorizationRegistry{
		watchers:        make(map[uint64]*authorizationWatcher),
		recheckInterval: defaultLiveAuthorizationRecheckInterval,
	}
}

func withLiveAuthorization(
	ctx context.Context,
	service *Service,
	claims *TokenClaims,
	credentialSource CredentialSource,
) context.Context {
	if service == nil || claims == nil {
		return ctx
	}
	claimsCopy := *claims
	return context.WithValue(ctx, liveAuthorizationContextKey{}, liveAuthorizationState{
		service:          service,
		claims:           claimsCopy,
		credentialSource: credentialSource,
	})
}

// HasLiveAuthorization reports whether a request passed through the real auth
// middleware. Internal handler composition may intentionally omit this state;
// browser-originated OpenClaw mutations use it to distinguish trusted internal
// actor messages from user-controlled payloads.
func HasLiveAuthorization(ctx context.Context) bool {
	state, ok := ctx.Value(liveAuthorizationContextKey{}).(liveAuthorizationState)
	return ok && state.service != nil
}

// RevalidateAuthorization synchronously reads current session, account, role,
// and permission state. Long-lived transports call it before every mutation so
// a stale handshake snapshot can never authorize a write.
func RevalidateAuthorization(ctx context.Context, permission string) (AuthContext, error) {
	state, ok := ctx.Value(liveAuthorizationContextKey{}).(liveAuthorizationState)
	if !ok || state.service == nil {
		ac, exists := authContextFromContext(ctx)
		if !exists {
			return AuthContext{}, ErrLiveAuthorizationInvalid
		}
		if permission != "" && !ac.Perms[permission] {
			return AuthContext{}, ErrLivePermissionDenied
		}
		return ac, nil
	}

	user, perms, err := state.service.ResolveSessionUser(ctx, &state.claims)
	if err != nil {
		return AuthContext{}, errors.Join(ErrLiveAuthorizationInvalid, err)
	}
	if permission != "" && !perms[permission] {
		return AuthContext{}, ErrLivePermissionDenied
	}
	return AuthContext{
		UserID:           user.ID,
		SessionID:        state.claims.ID,
		Email:            user.Email,
		Name:             user.Name,
		Role:             user.Role,
		Perms:            perms,
		CredentialSource: state.credentialSource,
	}, nil
}

func authContextFromContext(ctx context.Context) (AuthContext, bool) {
	value := ctx.Value(authContextKey)
	ac, ok := value.(AuthContext)
	return ac, ok
}

func shouldMonitorLiveAuthorization(r *http.Request) bool {
	if r == nil {
		return false
	}
	if IsWebSocketUpgradeRequest(r) {
		return true
	}
	if strings.Contains(strings.ToLower(r.Header.Get("Accept")), "text/event-stream") {
		return true
	}
	path := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(r.URL.Path)), "/")
	return strings.HasSuffix(path, "/stream")
}

func (s *Service) monitorAuthorization(
	parent context.Context,
	claims *TokenClaims,
	permission string,
) (context.Context, func(), error) {
	if s == nil || s.liveAuthorization == nil || claims == nil {
		return parent, func() {}, ErrLiveAuthorizationInvalid
	}

	ctx, cancelCause := context.WithCancelCause(parent)
	var once sync.Once
	watcher := &authorizationWatcher{
		userID:    claims.UserID,
		sessionID: claims.ID,
	}
	cleanup := func(cause error) {
		once.Do(func() {
			s.liveAuthorization.unregister(watcher.id)
			cancelCause(cause)
		})
	}
	watcher.cancel = cleanup
	if err := s.liveAuthorization.register(watcher); err != nil {
		cancelCause(err)
		return ctx, func() {}, err
	}

	// Close the check/register race: a revocation committed between the
	// middleware's first lookup and registry insertion is caught before the
	// downstream long-lived handler starts.
	if err := s.revalidateLiveClaims(parent, claims, permission); err != nil {
		cleanup(err)
		return ctx, func() {}, err
	}

	interval := s.liveAuthorization.interval()
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		var expiry <-chan time.Time
		var expiryTimer *time.Timer
		if claims.ExpiresAt != nil {
			delay := time.Until(claims.ExpiresAt.Time)
			if delay <= 0 {
				cleanup(ErrLiveAuthorizationInvalid)
				return
			}
			expiryTimer = time.NewTimer(delay)
			expiry = expiryTimer.C
			defer expiryTimer.Stop()
		}

		for {
			select {
			case <-ctx.Done():
				return
			case <-expiry:
				cleanup(ErrLiveAuthorizationInvalid)
				return
			case <-ticker.C:
				if err := s.revalidateLiveClaims(ctx, claims, permission); err != nil {
					cleanup(err)
					return
				}
			}
		}
	}()

	return ctx, func() { cleanup(context.Canceled) }, nil
}

func (s *Service) revalidateLiveClaims(
	ctx context.Context,
	claims *TokenClaims,
	permission string,
) error {
	_, permissions, err := s.ResolveSessionUser(ctx, claims)
	if err != nil {
		return errors.Join(ErrLiveAuthorizationInvalid, err)
	}
	if permission != "" && !permissions[permission] {
		return ErrLivePermissionDenied
	}
	return nil
}

func (r *liveAuthorizationRegistry) register(watcher *authorizationWatcher) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.watchers) >= maxLiveAuthorizationWatchers {
		return ErrLiveAuthorizationCapacity
	}
	userWatchers := 0
	sessionWatchers := 0
	for _, existing := range r.watchers {
		if existing.userID == watcher.userID {
			userWatchers++
		}
		if existing.sessionID == watcher.sessionID {
			sessionWatchers++
		}
	}
	if userWatchers >= maxLiveAuthorizationWatchersPerUser ||
		sessionWatchers >= maxLiveAuthorizationWatchersPerSession {
		return ErrLiveAuthorizationCapacity
	}
	r.nextID++
	watcher.id = r.nextID
	r.watchers[watcher.id] = watcher
	return nil
}

func (r *liveAuthorizationRegistry) unregister(id uint64) {
	if id == 0 {
		return
	}
	r.mu.Lock()
	delete(r.watchers, id)
	r.mu.Unlock()
}

func (r *liveAuthorizationRegistry) interval() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.recheckInterval <= 0 {
		return defaultLiveAuthorizationRecheckInterval
	}
	return r.recheckInterval
}

func (r *liveAuthorizationRegistry) invalidateUser(userID string) {
	r.invalidate(func(watcher *authorizationWatcher) bool {
		return watcher.userID == userID
	})
}

func (r *liveAuthorizationRegistry) invalidateSession(sessionID string) {
	r.invalidate(func(watcher *authorizationWatcher) bool {
		return watcher.sessionID == sessionID
	})
}

func (r *liveAuthorizationRegistry) invalidateAll() {
	r.invalidate(func(*authorizationWatcher) bool { return true })
}

func (r *liveAuthorizationRegistry) invalidate(matches func(*authorizationWatcher) bool) {
	r.mu.Lock()
	cancelers := make([]func(error), 0)
	for _, watcher := range r.watchers {
		if matches(watcher) {
			cancelers = append(cancelers, watcher.cancel)
		}
	}
	r.mu.Unlock()
	for _, cancel := range cancelers {
		cancel(ErrLiveAuthorizationInvalid)
	}
}

func (s *Service) invalidateUserAuthorization(userID string) {
	if s != nil && s.liveAuthorization != nil {
		s.liveAuthorization.invalidateUser(strings.TrimSpace(userID))
	}
}

func (s *Service) invalidateSessionAuthorization(sessionID string) {
	if s != nil && s.liveAuthorization != nil {
		s.liveAuthorization.invalidateSession(strings.TrimSpace(sessionID))
	}
}
