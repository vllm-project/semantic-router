// Package maskingregistry keeps masking placeholders for good: one index per
// value, type and scope, across requests, sessions and restarts. It stores
// keyed digests of values, never the values.
package maskingregistry

import (
	"crypto/hkdf"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

// Store is a namespace's allocation history. Prepare reports which of indices
// are issued and reserves the unknown ones for good, so they are never
// allocated. Allocate returns the index of digest, allocating the lowest free
// one the first time. Claim records that template renders typ, so two types
// never share a placeholder spelling. Each call commits before it returns.
type Store interface {
	Prepare(scope, typ string, indices []int64) (map[int64]bool, error)
	Allocate(scope, typ, digest string) (int64, error)
	Claim(typ, template string) error
}

// Identity is an opened namespace: its store and the keys derived from its
// secret.
type Identity struct {
	Store              Store
	name               string
	scopeKey, valueKey []byte
	SessionScope       bool

	mu      sync.Mutex
	claimed map[string]string
}

// encode length-prefixes each part, so no two part lists encode alike.
func encode(parts ...string) []byte {
	var out []byte
	for _, p := range parts {
		out = binary.BigEndian.AppendUint64(out, uint64(len(p)))
		out = append(out, p...)
	}
	return out
}

func mac(key []byte, parts ...string) string {
	h := hmac.New(sha256.New, key)
	h.Write(encode(parts...))
	return hex.EncodeToString(h.Sum(nil))
}

// Scope is the keyed digest of a request's scope: its verified principal, or
// the installation when principal is empty, and its root session when the
// namespace separates sessions.
func (i *Identity) Scope(principal, session string) (string, error) {
	kind := "principal"
	if principal == "" {
		kind = "installation"
	}
	if !i.SessionScope {
		session = ""
	} else if session == "" {
		return "", errors.New("masking registry: session_scope needs a request with a session id")
	}
	return mac(i.scopeKey, "vsr-masking-v1", i.name, kind, principal, session), nil
}

// Digest is the keyed digest a value is stored under.
func (i *Identity) Digest(scope, typ, value string) string {
	return mac(i.valueKey, "vsr-masking-v1", i.name, scope, typ, value)
}

// Claim records template for typ, once per process.
func (i *Identity) Claim(typ, template string) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if owner, ok := i.claimed[template]; ok && owner == typ {
		return nil
	}
	if err := i.Store.Claim(typ, template); err != nil {
		return err
	}
	i.claimed[template] = typ
	return nil
}

// opened shares one handle per namespace across configuration generations: a
// file store keeps its process lock for the life of the process.
var opened = struct {
	sync.Mutex
	identities map[string]*Identity
}{identities: map[string]*Identity{}}

func readKey(ns config.MaskingIdentityNamespace) ([]byte, error) {
	if ns.KeyFile != "" {
		key, err := os.ReadFile(ns.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("masking registry: read key_file: %w", err)
		}
		if len(key) != 32 {
			return nil, fmt.Errorf("masking registry: key_file %s must hold 32 bytes", ns.KeyFile)
		}
		return key, nil
	}
	key, err := hex.DecodeString(strings.TrimSpace(os.Getenv(ns.KeyEnv)))
	if err != nil || len(key) != 32 {
		return nil, fmt.Errorf("masking registry: %s must hold 64 hex digits", ns.KeyEnv)
	}
	return key, nil
}

// Open returns namespace name, opening its key and store the first time. It
// refuses a store made under another key or namespace.
func Open(name string, ns config.MaskingIdentityNamespace) (*Identity, error) {
	key, err := readKey(ns)
	if err != nil {
		return nil, err
	}
	header := mac(key, "vsr-masking-header-v1", name, fmt.Sprint(ns.SessionScope))
	cacheKey := strings.Join([]string{name, header, ns.Registry.Backend, ns.Registry.Path, postgresTarget(ns.Registry.Postgres)}, "\x00")
	opened.Lock()
	defer opened.Unlock()
	if existing := opened.identities[cacheKey]; existing != nil {
		return existing, nil
	}
	var store Store
	switch ns.Registry.Backend {
	case "file":
		store, err = OpenFile(ns.Registry.Path, name, header)
	case "postgres":
		store, err = openPostgres(postgresConfig(ns.Registry.Postgres), name, header)
	default:
		err = errors.New("masking registry: backend must be file or postgres")
	}
	if err != nil {
		return nil, err
	}
	derive := func(label string) []byte {
		derived, _ := hkdf.Key(sha256.New, key, nil, label, 32)
		return derived
	}
	identity := &Identity{
		Store: store, name: name, SessionScope: ns.SessionScope,
		scopeKey: derive("vsr-masking-v1/scope"), valueKey: derive("vsr-masking-v1/value"),
		claimed: map[string]string{},
	}
	opened.identities[cacheKey] = identity
	return identity, nil
}

func postgresTarget(c *config.RouterReplayPostgresConfig) string {
	if c == nil {
		return ""
	}
	return fmt.Sprintf("%s:%d/%s?user=%s", c.Host, c.Port, c.Database, c.User)
}

func postgresConfig(c *config.RouterReplayPostgresConfig) *postgres.Config {
	if c == nil {
		return nil
	}
	return &postgres.Config{
		Host: c.Host, Port: c.Port, Database: c.Database, User: c.User, Password: c.Password,
		SSLMode: c.SSLMode, MaxOpenConns: c.MaxOpenConns, MaxIdleConns: c.MaxIdleConns,
		ConnMaxLifetime: c.ConnMaxLifetime,
	}
}
