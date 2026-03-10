package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"time"
)

const (
	fetchRawMaxSize = 2 * 1024 * 1024 // 2 MB
	fetchRawTimeout = 15 * time.Second
)

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

		parsed, err := url.Parse(targetURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid URL, must be http or https"})
			return
		}

		log.Printf("[FetchRaw] Fetching: %s", targetURL)

		client := &http.Client{
			Timeout: fetchRawTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 10 {
					return fmt.Errorf("too many redirects")
				}
				return nil
			},
		}

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

		body, err := io.ReadAll(io.LimitReader(resp.Body, fetchRawMaxSize))
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

		log.Printf("[FetchRaw] Success, %d bytes from %s", len(body), targetURL)

		_ = json.NewEncoder(w).Encode(FetchRawResponse{Content: string(body)})
	}
}
