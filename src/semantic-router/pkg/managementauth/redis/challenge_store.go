package redis

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	redisclient "github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	defaultChallengeTTL   = 2 * time.Minute
	defaultChallengeLimit = 10
)

var challengeIssuerPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

type ChallengeOptions struct {
	Client      *redisclient.Client
	KeyPrefix   string
	TTL         time.Duration
	RateLimit   int
	NewID       func() (string, error)
	RandomBytes func([]byte) (int, error)
}

type ChallengeStore struct {
	client      *redisclient.Client
	prefix      string
	ttl         time.Duration
	rateLimit   int
	newID       func() (string, error)
	randomBytes func([]byte) (int, error)
}

func NewChallengeStore(options ChallengeOptions) (*ChallengeStore, error) {
	if options.Client == nil || !keyPrefixPattern.MatchString(options.KeyPrefix) {
		return nil, errors.New("management exchange challenges require a Valkey client and canonical key prefix")
	}
	ttl := options.TTL
	if ttl == 0 {
		ttl = defaultChallengeTTL
	}
	limit := options.RateLimit
	if limit == 0 {
		limit = defaultChallengeLimit
	}
	if ttl < 30*time.Second || ttl > 10*time.Minute || limit < 1 || limit > 100 {
		return nil, errors.New("management exchange challenge TTL or rate limit is out of bounds")
	}
	newID := options.NewID
	if newID == nil {
		newID = func() (string, error) { return uuid.NewString(), nil }
	}
	randomBytes := options.RandomBytes
	if randomBytes == nil {
		randomBytes = rand.Read
	}
	return &ChallengeStore{client: options.Client, prefix: options.KeyPrefix, ttl: ttl, rateLimit: limit, newID: newID, randomBytes: randomBytes}, nil
}

func (store *ChallengeStore) Ready(ctx context.Context) error {
	if store == nil || store.client == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := store.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("check Management challenge Valkey: %w", err)
	}
	return nil
}

func (store *ChallengeStore) Create(ctx context.Context, issuerID, rateIdentity string, now time.Time) (managementauth.ExchangeChallenge, error) {
	if store == nil || !challengeIssuerPattern.MatchString(issuerID) || !canonicalRateIdentity(rateIdentity) || now.IsZero() {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationDenied
	}
	id, createErr := store.newID()
	if createErr != nil || !challengeIssuerPattern.MatchString(id) {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationUnavailable
	}
	nonceBytes := make([]byte, 32)
	if count, err := store.randomBytes(nonceBytes); err != nil || count != len(nonceBytes) {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationUnavailable
	}
	nonce := base64.RawURLEncoding.EncodeToString(nonceBytes)
	digest := challengeDigest(issuerID, nonce)
	value := issuerID + ":" + hex.EncodeToString(digest[:])
	rateDigest := sha256.Sum256([]byte(rateIdentity))
	accepted, createErr := createChallengeScript.Run(ctx, store.client,
		[]string{store.challengeKey(id), store.rateKey(hex.EncodeToString(rateDigest[:]))},
		value, int64(store.ttl/time.Millisecond), store.rateLimit,
	).Int64()
	if createErr != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("create Management exchange challenge: %w", createErr)
	}
	if accepted != 1 {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationDenied
	}
	return managementauth.ExchangeChallenge{ID: id, Nonce: nonce, ExpiresAt: now.UTC().Add(store.ttl)}, nil
}

func (store *ChallengeStore) Consume(ctx context.Context, id, issuerID, nonce string, now time.Time) error {
	if store == nil || !challengeIssuerPattern.MatchString(id) || !challengeIssuerPattern.MatchString(issuerID) ||
		now.IsZero() || len(nonce) != 43 || strings.TrimSpace(nonce) != nonce {
		return managementauth.ErrAuthenticationDenied
	}
	digest := challengeDigest(issuerID, nonce)
	expected := issuerID + ":" + hex.EncodeToString(digest[:])
	consumed, err := consumeChallengeScript.Run(ctx, store.client, []string{store.challengeKey(id)}, expected).Int64()
	if err != nil {
		return fmt.Errorf("consume Management exchange challenge: %w", err)
	}
	if consumed != 1 {
		return managementauth.ErrAuthenticationDenied
	}
	return nil
}

func (store *ChallengeStore) challengeKey(id string) string {
	return store.prefix + ":management:challenge:" + id
}

func (store *ChallengeStore) rateKey(id string) string {
	return store.prefix + ":management:challenge-rate:" + id
}

func challengeDigest(issuerID, nonce string) [sha256.Size]byte {
	hash := sha256.New()
	_, _ = hash.Write([]byte(issuerID))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(nonce))
	var result [sha256.Size]byte
	copy(result[:], hash.Sum(nil))
	return result
}

func canonicalRateIdentity(value string) bool {
	return len(value) >= 1 && len(value) <= 256 && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n\t")
}

var createChallengeScript = redisclient.NewScript(`
local count = redis.call('INCR', KEYS[2])
if count == 1 then redis.call('PEXPIRE', KEYS[2], ARGV[2]) end
if count > tonumber(ARGV[3]) then return 0 end
local inserted = redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2], 'NX')
if not inserted then return 0 end
return 1
`)

var consumeChallengeScript = redisclient.NewScript(`
local value = redis.call('GET', KEYS[1])
if not value or value ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
return 1
`)

var _ managementauth.ExchangeChallengeStore = (*ChallengeStore)(nil)
