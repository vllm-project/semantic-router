package auth

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

const maxAuthJSONBodyBytes int64 = 16 * 1024

var (
	errAuthUnsupportedMediaType = errors.New("content type must be application/json")
	errAuthRequestTooLarge      = errors.New("request body is too large")
	errAuthInvalidJSON          = errors.New("request body must contain one valid JSON object")
	errAuthInvalidUnicode       = errors.New("request body must contain valid Unicode")
)

func decodeAuthJSON(w http.ResponseWriter, r *http.Request, destination any) error {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return errAuthUnsupportedMediaType
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxAuthJSONBodyBytes)
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			return errAuthRequestTooLarge
		}
		return errAuthInvalidJSON
	}
	if !jsonunicode.Valid(rawBody) {
		return errAuthInvalidUnicode
	}

	decoder := json.NewDecoder(bytes.NewReader(rawBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			return errAuthRequestTooLarge
		}
		return errAuthInvalidJSON
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			return errAuthRequestTooLarge
		}
		return errAuthInvalidJSON
	}
	return nil
}

func writeAuthDecodeError(w http.ResponseWriter, err error) {
	status := http.StatusBadRequest
	switch {
	case errors.Is(err, errAuthUnsupportedMediaType):
		status = http.StatusUnsupportedMediaType
	case errors.Is(err, errAuthRequestTooLarge):
		status = http.StatusRequestEntityTooLarge
	}
	http.Error(w, err.Error(), status)
}

func writePasswordPolicyError(w http.ResponseWriter, err error) bool {
	policyErr, ok := asPasswordPolicyError(err)
	if !ok {
		return false
	}
	http.Error(w, policyErr.Message, http.StatusUnprocessableEntity)
	return true
}

func setAuthNoStoreHeaders(w http.ResponseWriter) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("X-Content-Type-Options", "nosniff")
}

func withAuthNoStore(handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
		handler(w, r)
	}
}

func loginRequestSource(r *http.Request) string {
	if r == nil {
		return ""
	}
	remote := strings.TrimSpace(r.RemoteAddr)
	if remote == "" {
		return ""
	}
	host := remote
	if parsedHost, _, err := net.SplitHostPort(remote); err == nil {
		host = parsedHost
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() || ip.IsLinkLocalUnicast() {
		// A loopback/private peer is commonly a shared ingress or sidecar. A
		// source bucket would then let a few failures lock out every user. Do
		// not trust forwarded headers without explicit trusted-proxy config;
		// deployments behind such a proxy must enforce a source limit there.
		return ""
	}
	return strings.ToLower(ip.String())
}

func writeLoginRateLimit(w http.ResponseWriter, rateErr *LoginRateLimitError) {
	retryAfter := time.Second
	if rateErr != nil && rateErr.RetryAfter > 0 {
		retryAfter = rateErr.RetryAfter
	}
	seconds := int64((retryAfter + time.Second - 1) / time.Second)
	w.Header().Set("Retry-After", strconv.FormatInt(seconds, 10))
	http.Error(w, "too many authentication attempts", http.StatusTooManyRequests)
}
