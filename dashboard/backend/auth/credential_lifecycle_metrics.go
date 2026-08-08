package auth

import (
	"log"
	"strings"
	"sync"
	"sync/atomic"
)

const (
	credentialLifecycleFailureStore          = "store"
	credentialLifecycleFailureResponseEncode = "response_encode"
)

var credentialLifecycleTerminalFailures sync.Map

func recordCredentialLifecycleTerminalFailure(operation, reason string) {
	operation = normalizeCredentialLifecycleMetricLabel(operation, CredentialLifecycleAdminPasswordReset)
	reason = normalizeCredentialLifecycleMetricLabel(reason, "unknown")
	counterValue, _ := credentialLifecycleTerminalFailures.LoadOrStore(
		operation+"\x00"+reason,
		&atomic.Int64{},
	)
	counterValue.(*atomic.Int64).Add(1)
	log.Printf("auth credential lifecycle terminal failure: operation=%s reason=%s", operation, reason)
}

func credentialLifecycleTerminalFailureMetric(operation, reason string) int64 {
	operation = normalizeCredentialLifecycleMetricLabel(operation, CredentialLifecycleAdminPasswordReset)
	reason = normalizeCredentialLifecycleMetricLabel(reason, "unknown")
	counterValue, ok := credentialLifecycleTerminalFailures.Load(operation + "\x00" + reason)
	if !ok {
		return 0
	}
	return counterValue.(*atomic.Int64).Load()
}

func normalizeCredentialLifecycleMetricLabel(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
