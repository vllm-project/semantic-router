package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

// A saved config the Router hot-reloads goes through its config lifecycle.
// One the running containers can't take -- the Router answers
// restart_required, or a split stack's generated Envoy config changes --
// becomes a pending activation that `vllm-sr serve` applies.

const (
	restartRequiredCode = "restart_required"
	envoyRestartDetail  = "Envoy's generated configuration changed"
)

// restartNeededError reports a config the running containers can't take. It
// records nothing: the change that published the config records the pending
// activation once it stands, so a change that is rolled back leaves none.
type restartNeededError struct {
	detail string
}

func (e *restartNeededError) Error() string {
	return "restart required: " + e.detail
}

func asRestartNeeded(err error) (*restartNeededError, bool) {
	var restart *restartNeededError
	ok := errors.As(err, &restart)
	return restart, ok
}

// recordRestart records that the config saved in configPath waits for
// `vllm-sr serve`, and returns the message that says who applies it.
func recordRestart(configPath string, detail string) (string, error) {
	config, err := os.ReadFile(configPath)
	if err == nil {
		err = recordPendingActivation(configPath, config, activationReasonRestart, detail)
	}
	if err != nil {
		return "", err
	}
	return restartRequiredMessage(configPath), nil
}

// How long the Dashboard waits for the Router to judge a saved document.
const routerVerdictTimeout = 15 * time.Second

var routerVerdict struct {
	sync.RWMutex
	endpoint string
	client   *http.Client
	provider routerauth.CredentialProvider
}

// ConfigureRouterVerdict names the Router whose config lifecycle judges saved
// documents; without one, every saved config counts as hot-reloadable.
func ConfigureRouterVerdict(routerAPIURL string, provider routerauth.CredentialProvider) {
	endpoint, err := routerConfigHashURL(routerAPIURL)
	routerVerdict.Lock()
	defer routerVerdict.Unlock()
	routerVerdict.endpoint, routerVerdict.provider = "", provider
	if err == nil {
		routerVerdict.endpoint = endpoint
		routerVerdict.client = &http.Client{Transport: &http.Transport{Proxy: nil}, Timeout: 5 * time.Second}
	}
}

// waitForRouterVerdict waits for the Router to take or refuse the document with
// this SHA-256, written at writtenAt. A refusal is a restartNeededError when a
// restart is the only reason, and an error with the Router's reasons
// otherwise. Without a Router, or without its verdict in time, the document
// counts as applied.
func waitForRouterVerdict(ctx context.Context, documentSHA256 string, writtenAt time.Time) error {
	routerVerdict.RLock()
	endpoint, client, provider := routerVerdict.endpoint, routerVerdict.client, routerVerdict.provider
	routerVerdict.RUnlock()
	if endpoint == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, routerVerdictTimeout)
	defer cancel()
	for {
		if response, err := fetchRouterConfigHash(ctx, client, endpoint, provider); err == nil {
			// A retry of the same bytes can first see a rejection from the
			// previous write, before the Router starts processing this one.
			if attempt := response.Activation; attempt != nil && !attempt.StartedAt.IsZero() && attempt.StartedAt.Before(writtenAt) {
				response.Activation = nil
			}
			if done, verdict := judgeRouterAttempt(response, documentSHA256); done {
				return verdict
			}
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// judgeRouterAttempt reads the Router's verdict on one document: done once the
// Router activated, superseded or refused it, with the refusal as the error.
func judgeRouterAttempt(response routerConfigHash, documentSHA256 string) (done bool, err error) {
	digest := strings.TrimPrefix(documentSHA256, "sha256:")
	if response.GeneratedRuntimeHash != digest {
		return false, nil
	}
	if response.ActiveRuntimeHash == digest {
		// A restored document may already be active without a new attempt.
		return true, nil
	}
	attempt := response.Activation
	if attempt == nil || strings.TrimPrefix(attempt.DocumentHash, "sha256:") != digest {
		return false, nil
	}
	switch attempt.Status {
	case "superseded":
		// A later document replaced it; its own save reports on it.
		return true, nil
	case "failed", "rejected":
	default:
		return false, nil
	}
	if len(attempt.Reasons) == 0 {
		return true, fmt.Errorf("the Router did not apply it: its attempt %s", attempt.Status)
	}
	restart := true
	messages := make([]string, 0, len(attempt.Reasons))
	for _, reason := range attempt.Reasons {
		restart = restart && reason.Code == restartRequiredCode
		messages = append(messages, reason.Message)
	}
	if restart {
		return true, &restartNeededError{detail: strings.Join(messages, "; ")}
	}
	for i, reason := range attempt.Reasons {
		messages[i] = fmt.Sprintf("%s (%s)", reason.Message, reason.Code)
	}
	return true, fmt.Errorf("the Router did not apply it: %s", strings.Join(messages, "; "))
}
