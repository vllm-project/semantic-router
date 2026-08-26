package postgres

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	defaultChallengeTTL   = 2 * time.Minute
	defaultChallengeLimit = 10
)

type ChallengeOptions struct {
	Database    *sql.DB
	TTL         time.Duration
	RateLimit   int
	NewID       func() (string, error)
	RandomBytes func([]byte) (int, error)
}

type ChallengeStore struct {
	db          *sql.DB
	ttl         time.Duration
	rateLimit   int
	newID       func() (string, error)
	randomBytes func([]byte) (int, error)
}

func NewChallengeStore(options ChallengeOptions) (*ChallengeStore, error) {
	if options.Database == nil {
		return nil, errors.New("management exchange challenges require a PostgreSQL database")
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
	return &ChallengeStore{db: options.Database, ttl: ttl, rateLimit: limit, newID: newID, randomBytes: randomBytes}, nil
}

func (store *ChallengeStore) Ready(ctx context.Context) error {
	if store == nil || store.db == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := store.db.PingContext(ctx); err != nil {
		return fmt.Errorf("check Management challenge PostgreSQL: %w", err)
	}
	return nil
}

func (store *ChallengeStore) Create(
	ctx context.Context,
	issuerID, rateIdentity string,
	now time.Time,
) (managementauth.ExchangeChallenge, error) {
	if store == nil || store.db == nil || !canonicalUUID(issuerID) || !canonicalRateIdentity(rateIdentity) || now.IsZero() {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationDenied
	}
	id, err := store.newID()
	if err != nil || !canonicalUUID(id) {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationUnavailable
	}
	nonceBytes := make([]byte, 32)
	if count, randomErr := store.randomBytes(nonceBytes); randomErr != nil || count != len(nonceBytes) {
		return managementauth.ExchangeChallenge{}, managementauth.ErrAuthenticationUnavailable
	}
	nonce := base64.RawURLEncoding.EncodeToString(nonceBytes)
	nonceDigest := challengeDigest(issuerID, nonce)
	rateDigest := sha256.Sum256([]byte(rateIdentity))
	expiresAt := now.UTC().Add(store.ttl)

	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("begin Management challenge creation: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	unsignedLockID := binary.BigEndian.Uint64(rateDigest[:8]) & uint64(math.MaxInt64)
	// #nosec G115 -- masking the sign bit bounds the advisory-lock key to MaxInt64.
	lockID := int64(unsignedLockID)
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock($1)`, lockID); err != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("lock Management challenge rate identity: %w", err)
	}
	var count int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM management_exchange_challenges
WHERE rate_identity_digest=$1 AND consumed_at IS NULL AND expires_at>$2`, rateDigest[:], now.UTC()).Scan(&count); err != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("count Management exchange challenges: %w", err)
	}
	if count >= store.rateLimit {
		return managementauth.ExchangeChallenge{}, &managementauth.ChallengeCapacityError{RetryAfter: store.ttl}
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_exchange_challenges
  (id,issuer_id,nonce_digest,rate_identity_digest,expires_at)
VALUES ($1,$2,$3,$4,$5)`, id, issuerID, nonceDigest[:], rateDigest[:], expiresAt); err != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("insert Management exchange challenge: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return managementauth.ExchangeChallenge{}, fmt.Errorf("commit Management exchange challenge: %w", err)
	}
	return managementauth.ExchangeChallenge{ID: id, Nonce: nonce, ExpiresAt: expiresAt}, nil
}

func (store *ChallengeStore) Consume(
	ctx context.Context,
	id, issuerID, nonce, rateIdentity string,
	now time.Time,
) error {
	if store == nil || store.db == nil || !canonicalUUID(id) || !canonicalUUID(issuerID) || now.IsZero() ||
		len(nonce) != 43 || strings.TrimSpace(nonce) != nonce || !canonicalRateIdentity(rateIdentity) {
		return managementauth.ErrAuthenticationDenied
	}
	digest := challengeDigest(issuerID, nonce)
	rateDigest := sha256.Sum256([]byte(rateIdentity))
	result, err := store.db.ExecContext(ctx, `UPDATE management_exchange_challenges
SET consumed_at=$5
WHERE id=$1 AND issuer_id=$2 AND nonce_digest=$3 AND rate_identity_digest=$4
  AND consumed_at IS NULL AND expires_at>$5`,
		id, issuerID, digest[:], rateDigest[:], now.UTC())
	if err != nil {
		return fmt.Errorf("consume Management exchange challenge: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil || rows != 1 {
		return managementauth.ErrAuthenticationDenied
	}
	return nil
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
	return len(value) >= 1 && len(value) <= 256 && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n\t")
}

var _ managementauth.ExchangeChallengeStore = (*ChallengeStore)(nil)
