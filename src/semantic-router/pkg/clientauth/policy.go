// Package clientauth decides which client keys an entrypoint accepts. It
// compiles the entrypoints' client_auth policies against the serving process's
// environment into an immutable Policy, which keeps digests of the keys and
// never the keys themselves.
package clientauth

import (
	"crypto/sha256"
	"crypto/subtle"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Policy holds the keys each protected entrypoint name accepts, per API
// namespace. A nil Policy protects nothing.
type Policy struct {
	protected map[name][]digest
}

type name struct {
	api   config.InferenceAPI
	model string
}

type digest [sha256.Size]byte

// Problem is one key reference that does not resolve to a usable key.
type Problem struct {
	// Path locates the reference in the canonical document.
	Path string
	// Message says what is wrong with the reference, never what its variable
	// holds.
	Message string
}

// Compile resolves the keys of every protected entrypoint with lookup, which
// reads the serving process's environment. Each referenced variable must hold
// one Bearer token. Compile returns every Problem it finds, and a Policy only
// when there is none.
func Compile(entrypoints []config.EntrypointMapping, lookup func(string) (string, bool)) (*Policy, []Problem) {
	policy := &Policy{protected: make(map[name][]digest)}
	var problems []Problem
	for index, entrypoint := range entrypoints {
		auth := entrypoint.ClientAuth
		if auth == nil {
			continue
		}
		path := fmt.Sprintf("entrypoints[%d].client_auth", index)
		if auth.Type != config.ClientAuthBearerAPIKey {
			problems = append(problems, Problem{Path: path + ".type", Message: fmt.Sprintf("unsupported type %q", auth.Type)})
			continue
		}
		digests := make([]digest, 0, len(auth.APIKeyEnvs))
		resolved := true
		for position, env := range auth.APIKeyEnvs {
			key, problem := resolve(env, lookup)
			if problem != "" {
				problems = append(problems, Problem{Path: fmt.Sprintf("%s.api_key_envs[%d]", path, position), Message: problem})
				resolved = false
				continue
			}
			digests = appendDigest(digests, sha256.Sum256([]byte(key)))
		}
		// An entrypoint whose keys do not all resolve accepts none of them.
		if !resolved {
			digests = nil
		}
		api := namespace(entrypoint.API)
		for _, model := range entrypoint.ModelNames {
			policy.protected[name{api: api, model: model}] = digests
		}
	}
	if len(problems) > 0 {
		return nil, problems
	}
	return policy, nil
}

// resolve returns the key env holds, or why it holds none.
func resolve(env string, lookup func(string) (string, bool)) (string, string) {
	key, set := lookup(env)
	switch {
	case !set:
		return "", fmt.Sprintf("environment variable %s is not set", env)
	case key == "":
		return "", fmt.Sprintf("environment variable %s is empty", env)
	case !isToken(key):
		return "", fmt.Sprintf("environment variable %s must hold one Bearer token: visible ASCII characters, no spaces", env)
	}
	return key, ""
}

// isToken reports whether key can arrive whole as a Bearer credential.
func isToken(key string) bool {
	for i := 0; i < len(key); i++ {
		if key[i] < '!' || key[i] > '~' {
			return false
		}
	}
	return true
}

func appendDigest(digests []digest, d digest) []digest {
	for _, existing := range digests {
		if existing == d {
			return digests
		}
	}
	return append(digests, d)
}

// namespace is the API namespace an entrypoint without one belongs to.
func namespace(api config.InferenceAPI) config.InferenceAPI {
	if api == "" {
		return config.ChatAPI
	}
	return api
}

// Requires reports whether model, in the api namespace, needs a client key.
// Names the policy does not protect, including names that are no entrypoint,
// need none.
func (p *Policy) Requires(api config.InferenceAPI, model string) bool {
	_, protected := p.keys(api, model)
	return protected
}

// Authorize reports whether key admits a request for model in the api
// namespace: always for a name Requires reports false for, otherwise only for
// one of the keys the entrypoint accepts. It compares in constant time.
func (p *Policy) Authorize(api config.InferenceAPI, model, key string) bool {
	digests, protected := p.keys(api, model)
	if !protected {
		return true
	}
	if key == "" {
		return false
	}
	presented := sha256.Sum256([]byte(key))
	match := 0
	for _, accepted := range digests {
		match |= subtle.ConstantTimeCompare(presented[:], accepted[:])
	}
	return match == 1
}

// AnyRequired reports whether any entrypoint needs a client key. Stored
// objects without an origin are accessible only while it is false.
func (p *Policy) AnyRequired() bool {
	return p != nil && len(p.protected) > 0
}

func (p *Policy) keys(api config.InferenceAPI, model string) ([]digest, bool) {
	if p == nil {
		return nil, false
	}
	digests, protected := p.protected[name{api: namespace(api), model: model}]
	return digests, protected
}
