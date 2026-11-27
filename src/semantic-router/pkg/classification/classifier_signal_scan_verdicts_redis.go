package classification

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const sharedVerdictMaxBytes = 256 << 10

type signedWindowVerdict struct {
	Payload json.RawMessage `json:"payload"`
	MAC     []byte          `json:"mac"`
}

type sharedWindowVerdict struct {
	Version   int                   `json:"version"`
	ExpiresAt int64                 `json:"expires_at"`
	Response  modelservice.Response `json:"response"`
}

func (s *sharedWindowVerdicts) key(key windowVerdictKey) string {
	return s.prefix + hex.EncodeToString(key[:])
}

func (s *sharedWindowVerdicts) signature(key string, payload []byte) []byte {
	mac := hmac.New(sha256.New, s.secret)
	writeVerdictField(mac, key)
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (s *sharedWindowVerdicts) get(ctx context.Context, key windowVerdictKey) (modelservice.Response, time.Time, bool) {
	if !s.available(ctx, "get") {
		return modelservice.Response{}, time.Time{}, false
	}
	lookup, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if !s.acquire(lookup, "get") {
		return modelservice.Response{}, time.Time{}, false
	}
	defer func() { <-s.slots }()
	name := s.key(key)
	// Bound untrusted data before decoding. Missing keys yield an empty string.
	data, err := s.client.GetRange(lookup, name, 0, sharedVerdictMaxBytes).Bytes()
	s.failed(ctx, "get", err)
	if err != nil {
		return modelservice.Response{}, time.Time{}, false
	}
	if len(data) == 0 {
		metrics.RecordGuardVerdictStoreOp("get", "miss")
		return modelservice.Response{}, time.Time{}, false
	}
	if len(data) > sharedVerdictMaxBytes {
		metrics.RecordGuardVerdictStoreOp("get", "oversized")
		return modelservice.Response{}, time.Time{}, false
	}
	var signed signedWindowVerdict
	if json.Unmarshal(data, &signed) != nil || !hmac.Equal(signed.MAC, s.signature(name, signed.Payload)) {
		metrics.RecordGuardVerdictStoreOp("get", "invalid")
		s.warn(&s.invalidWarning, "invalid_entry", "get")
		return modelservice.Response{}, time.Time{}, false
	}
	var verdict sharedWindowVerdict
	if json.Unmarshal(signed.Payload, &verdict) != nil || verdict.Version != 1 {
		metrics.RecordGuardVerdictStoreOp("get", "invalid")
		return modelservice.Response{}, time.Time{}, false
	}
	expires := time.UnixMilli(verdict.ExpiresAt)
	if !time.Now().Before(expires) {
		metrics.RecordGuardVerdictStoreOp("get", "expired")
		return modelservice.Response{}, time.Time{}, false
	}
	return verdict.Response, expires, true
}

// Publish a first model result before returning the request so the next
// replica can reuse it. Renewals use the background queue instead.
func (s *sharedWindowVerdicts) put(ctx context.Context, key windowVerdictKey, response modelservice.Response) time.Time {
	expires := time.Now().Add(windowVerdictTTL)
	if s.putUntil(ctx, key, response, expires) {
		return expires
	}
	return time.Time{}
}

func (s *sharedWindowVerdicts) putUntil(ctx context.Context, key windowVerdictKey, response modelservice.Response, expires time.Time) bool {
	if !s.available(ctx, "put") {
		return false
	}
	if !time.Now().Before(expires) {
		metrics.RecordGuardVerdictStoreOp("put", "expired")
		return false
	}
	response = cloneWindowResponse(response)
	for id, answer := range response.Answers {
		for i := range answer.Spans {
			answer.Spans[i].Text = ""
		}
		response.Answers[id] = answer
	}
	payload, err := json.Marshal(sharedWindowVerdict{Version: 1, ExpiresAt: expires.UnixMilli(), Response: response})
	if err != nil {
		metrics.RecordGuardVerdictStoreOp("put", "invalid")
		return false
	}
	name := s.key(key)
	data, err := json.Marshal(signedWindowVerdict{Payload: payload, MAC: s.signature(name, payload)})
	if err != nil || len(data) > sharedVerdictMaxBytes {
		metrics.RecordGuardVerdictStoreOp("put", "oversized")
		return false
	}
	write, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if !s.acquire(write, "put") {
		return false
	}
	defer func() { <-s.slots }()
	ttl := time.Until(expires)
	if ttl <= 0 {
		metrics.RecordGuardVerdictStoreOp("put", "expired")
		return false
	}
	err = s.client.Set(write, name, data, ttl).Err()
	s.failed(ctx, "put", err)
	if err != nil {
		return false
	}
	metrics.RecordGuardVerdictStoreOp("put", "ok")
	return true
}

func restoreWindowSpans(call *systemOneCall, response modelservice.Response) bool {
	var runes []rune
	for id, answer := range response.Answers {
		for i := range answer.Spans {
			if runes == nil {
				call.scan.mu.Lock()
				window := call.scan.windows[call.window]
				runes = []rune(call.scan.text[window.start:window.end])
				call.scan.mu.Unlock()
			}
			span := &answer.Spans[i]
			if span.Start < 0 || span.End > len(runes) || span.Start >= span.End {
				return false
			}
			span.Text = string(runes[span.Start:span.End])
		}
		response.Answers[id] = answer
	}
	return true
}
