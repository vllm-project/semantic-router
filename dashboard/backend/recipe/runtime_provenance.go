package recipe

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const (
	maxConfigHashResponseBytes                                = 64 << 10
	configHashObservationTimeout                              = 2 * time.Second
	ProvenanceVerified           ProvenanceVerificationStatus = "verified"
	ProvenanceUnverified         ProvenanceVerificationStatus = "unverified"
)

// ProvenanceVerificationStatus is independent of probe assertion success. A
// probe can pass its routing assertions while its runtime receipt remains
// unverified because the Router is old, unavailable, changing, or pending.
type ProvenanceVerificationStatus string

// RouterConfigHashes is one normalized observation from Router /config/hash.
// Every non-empty digest uses the same sha256:<lowercase-hex> representation as
// managed Recipe package digests.
type RouterConfigHashes struct {
	SourceConfigHash     string `json:"source_config_hash,omitempty"`
	GeneratedRuntimeHash string `json:"generated_runtime_hash,omitempty"`
	ActiveRuntimeHash    string `json:"active_runtime_hash,omitempty"`
	ActivationStatus     string `json:"activation_status,omitempty"`
}

// ValidationProvenance binds a probe result to its immutable five-file package
// and records Router identity on both sides of the Eval call. Before and After
// are retained even when verification fails so a caller can distinguish a
// missing contract from a mid-validation activation.
type ValidationProvenance struct {
	Status            ProvenanceVerificationStatus `json:"status"`
	Reason            string                       `json:"reason,omitempty"`
	PackageHash       string                       `json:"package_hash"`
	PackageConfigHash string                       `json:"package_config_hash"`
	Before            RouterConfigHashes           `json:"before"`
	After             RouterConfigHashes           `json:"after"`
}

// ConfigHashObserver is implemented by Router clients that support the
// versioned /config/hash identity contract.
type ConfigHashObserver interface {
	ObserveConfigHash(context.Context) (RouterConfigHashes, error)
}

type routerConfigHashResponse struct {
	Hash        string `json:"hash"`
	RuntimeHash string `json:"runtime_hash"`
	ActiveHash  string `json:"active_hash"`
	Status      string `json:"status"`
}

// ObserveConfigHash lets the production Eval client bracket an evaluation
// without introducing a second Router endpoint or transport configuration.
func (e *HTTPRouterEvaluator) ObserveConfigHash(ctx context.Context) (RouterConfigHashes, error) {
	endpoint, err := configHashEndpoint(e.BaseURL)
	if err != nil {
		return RouterConfigHashes{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return RouterConfigHashes{}, errors.New("router config hash URL is invalid")
	}
	if authErr := routerauth.RewriteAuthorization(request, e.CredentialProvider); authErr != nil {
		return RouterConfigHashes{}, errors.New("router management credential is unavailable")
	}
	client := noRedirectHTTPClient(e.Client)
	response, err := client.Do(request)
	if err != nil {
		return RouterConfigHashes{}, errors.New("router config hash is unavailable")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return RouterConfigHashes{}, fmt.Errorf("router config hash returned status %d", response.StatusCode)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxConfigHashResponseBytes+1))
	if err != nil {
		return RouterConfigHashes{}, errors.New("read router config hash response")
	}
	if len(payload) > maxConfigHashResponseBytes {
		return RouterConfigHashes{}, errors.New("router config hash response exceeds size limit")
	}
	var raw routerConfigHashResponse
	decoder := json.NewDecoder(bytes.NewReader(payload))
	if err := decoder.Decode(&raw); err != nil {
		return RouterConfigHashes{}, errors.New("router config hash response is invalid")
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return RouterConfigHashes{}, errors.New("router config hash response contains trailing data")
	}
	return normalizeConfigHashObservation(RouterConfigHashes{
		SourceConfigHash:     raw.Hash,
		GeneratedRuntimeHash: raw.RuntimeHash,
		ActiveRuntimeHash:    raw.ActiveHash,
		ActivationStatus:     raw.Status,
	})
}

func configHashEndpoint(base string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(base))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("router API URL is not configured")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/config/hash"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	parsed.User = nil
	return parsed.String(), nil
}

func noRedirectHTTPClient(configured *http.Client) *http.Client {
	if configured == nil {
		configured = &http.Client{}
	}
	clone := *configured
	clone.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &clone
}

type validationProvenanceSession struct {
	result      ValidationProvenance
	observer    ConfigHashObserver
	localBefore string
	localErr    error
	beforeErr   error
}

func (s *Service) beginValidationProvenance(ctx context.Context, snapshot *snapshot) validationProvenanceSession {
	session := validationProvenanceSession{
		result: ValidationProvenance{
			Status:            ProvenanceUnverified,
			PackageHash:       snapshot.recipeDigest,
			PackageConfigHash: digestBytes(snapshot.files["config"]),
		},
	}
	session.localBefore, session.localErr = s.boundRuntimeConfigDigest(snapshot.files["config"])
	observer, supported := s.evaluator.(ConfigHashObserver)
	if !supported {
		return session
	}
	session.observer = observer
	session.result.Before, session.beforeErr = observeConfigHashBounded(ctx, observer)
	return session
}

func (s *Service) finishValidationProvenance(
	ctx context.Context,
	snapshot *snapshot,
	session validationProvenanceSession,
) ValidationProvenance {
	var afterErr error
	if session.observer != nil {
		session.result.After, afterErr = observeConfigHashBounded(ctx, session.observer)
	}
	localAfter, localAfterErr := s.boundRuntimeConfigDigest(snapshot.files["config"])
	reason := provenanceVerificationReason(
		session,
		localAfter,
		localAfterErr,
		afterErr,
	)
	if reason == "" {
		session.result.Status = ProvenanceVerified
		session.result.Reason = ""
		return session.result
	}
	session.result.Status = ProvenanceUnverified
	session.result.Reason = reason
	return session.result
}

func observeConfigHashBounded(ctx context.Context, observer ConfigHashObserver) (RouterConfigHashes, error) {
	observationContext, cancel := context.WithTimeout(ctx, configHashObservationTimeout)
	defer cancel()
	observation, err := observer.ObserveConfigHash(observationContext)
	if err != nil {
		return RouterConfigHashes{}, err
	}
	return normalizeConfigHashObservation(observation)
}

func provenanceVerificationReason(
	session validationProvenanceSession,
	localAfter string,
	localAfterErr error,
	afterErr error,
) string {
	checks := []struct {
		failed bool
		reason string
	}{
		{session.observer == nil, "config_hash_unsupported"},
		{session.beforeErr != nil, "before_observation_unavailable"},
		{afterErr != nil, "after_observation_unavailable"},
		{session.localErr != nil, "package_runtime_binding_unavailable_before_validation"},
		{localAfterErr != nil, "package_runtime_binding_unavailable_after_validation"},
		{session.localBefore != localAfter, "dashboard_runtime_config_changed_during_validation"},
		{session.result.Before != session.result.After, "router_config_changed_during_validation"},
		{incompleteConfigHashes(session.result.Before), "router_config_hashes_incomplete"},
		{session.result.Before.SourceConfigHash != session.result.PackageConfigHash, "router_source_hash_does_not_match_package_config"},
		{session.result.Before.GeneratedRuntimeHash != session.localBefore, "router_runtime_hash_does_not_match_dashboard_runtime"},
		{!strings.EqualFold(session.result.Before.ActivationStatus, "active"), "router_runtime_not_active"},
		{session.result.Before.GeneratedRuntimeHash != session.result.Before.ActiveRuntimeHash, "router_runtime_hash_does_not_match_active_hash"},
	}
	for _, check := range checks {
		if check.failed {
			return check.reason
		}
	}
	return ""
}

func incompleteConfigHashes(hashes RouterConfigHashes) bool {
	return hashes.SourceConfigHash == "" || hashes.GeneratedRuntimeHash == "" || hashes.ActiveRuntimeHash == ""
}

func normalizeConfigHashObservation(observation RouterConfigHashes) (RouterConfigHashes, error) {
	var err error
	if observation.SourceConfigHash, err = normalizeSHA256Digest(observation.SourceConfigHash); err != nil {
		return RouterConfigHashes{}, fmt.Errorf("source config hash: %w", err)
	}
	if observation.GeneratedRuntimeHash, err = normalizeSHA256Digest(observation.GeneratedRuntimeHash); err != nil {
		return RouterConfigHashes{}, fmt.Errorf("generated runtime hash: %w", err)
	}
	if observation.ActiveRuntimeHash, err = normalizeSHA256Digest(observation.ActiveRuntimeHash); err != nil {
		return RouterConfigHashes{}, fmt.Errorf("active runtime hash: %w", err)
	}
	observation.ActivationStatus = strings.ToLower(strings.TrimSpace(observation.ActivationStatus))
	return observation, nil
}

func normalizeSHA256Digest(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", nil
	}
	value = strings.TrimPrefix(strings.ToLower(value), "sha256:")
	if len(value) != 64 {
		return "", errors.New("must be a SHA-256 digest")
	}
	if _, err := hex.DecodeString(value); err != nil {
		return "", errors.New("must be a SHA-256 digest")
	}
	return "sha256:" + value, nil
}
