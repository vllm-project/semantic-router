package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

const fetchRawTimeout = 15 * time.Second

// FetchRawRequest is the request body for the fetch-raw endpoint.
type FetchRawRequest struct {
	URL string `json:"url"`
}

// FetchRawResponse is the response body for the fetch-raw endpoint.
type FetchRawResponse struct {
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

// FetchRawHandler returns an HTTP handler that proxies a GET request to the
// given URL and returns the raw text body without any HTML cleaning or
// truncation. Designed for fetching YAML/JSON config files from remote URLs,
// bypassing browser CORS restrictions.
func FetchRawHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "method not allowed, use POST"})
			return
		}

		var req FetchRawRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid request body"})
			return
		}

		targetURL := req.URL
		if targetURL == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "url is required"})
			return
		}

		policy := outboundPolicy(fetchRawTimeout)
		if _, err := policy.ValidateURL(targetURL); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid URL, must be http or https"})
			return
		}

		log.Printf("[FetchRaw] Fetching: %s", redactURLForLog(targetURL))

		// The destination is rechecked after DNS and on every redirect, so a
		// name that resolves to an internal address is refused at dial time.
		client := policy.NewClient()

		httpReq, err := http.NewRequest("GET", targetURL, nil)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: fmt.Sprintf("failed to create request: %v", err)})
			return
		}

		httpReq.Header.Set("User-Agent", getRandomUserAgent())
		httpReq.Header.Set("Accept", "text/plain, application/x-yaml, application/json, */*")

		resp, err := client.Do(httpReq)
		if err != nil {
			// A refused destination is the caller's error and must not echo the
			// resolved address or the reason back to them.
			if isForbiddenFetchTarget(err) {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "destination is not permitted"})
				return
			}
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: fmt.Sprintf("fetch failed: %v", err)})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: fmt.Sprintf("remote returned HTTP %d: %s", resp.StatusCode, resp.Status)})
			return
		}

		// Bound the decoded body and say so, rather than silently handing back a
		// truncated config the caller would parse as complete.
		body, err := safefetch.ReadBounded(resp.Body, fetchRawMaxResponseBytes)
		if errors.Is(err, safefetch.ErrResponseTooLarge) {
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "remote content exceeds the size limit"})
			return
		}
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: fmt.Sprintf("failed to read response: %v", err)})
			return
		}

		if len(body) == 0 {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "remote returned empty content"})
			return
		}

		log.Printf("[FetchRaw] Success, %d bytes from %s", len(body), redactURLForLog(targetURL))

		_ = json.NewEncoder(w).Encode(FetchRawResponse{Content: string(body)})
	}
}
