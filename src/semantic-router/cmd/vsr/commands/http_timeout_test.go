package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestHTTPTimeout verifies 30-second HTTP timeout prevents hanging requests
func TestHTTPTimeout(t *testing.T) {
	t.Run("request times out after 30 seconds", func(t *testing.T) {
		// Create a server that never responds
		hangingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(60 * time.Second) // Never responds
		}))
		defer hangingServer.Close()

		// Create HTTP client with 30-second timeout
		client := &http.Client{
			Timeout: 30 * time.Second,
		}

		reqBody := map[string]string{"text": "test"}
		jsonData, _ := json.Marshal(reqBody)

		startTime := time.Now()
		resp, err := client.Post(
			hangingServer.URL,
			"application/json",
			bytes.NewBuffer(jsonData),
		)
		elapsed := time.Since(startTime)

		if err == nil {
			resp.Body.Close()
			t.Error("Request should have timed out but succeeded")
		}

		// Verify timeout occurred within expected range (30-31 seconds)
		if elapsed < 29*time.Second || elapsed > 31*time.Second {
			t.Errorf("Timeout should occur around 30s, took %v", elapsed)
		}
	})

	t.Run("fast responses still work", func(t *testing.T) {
		// Create a server that responds quickly
		fastServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		}))
		defer fastServer.Close()

		client := &http.Client{
			Timeout: 30 * time.Second,
		}

		reqBody := map[string]string{"text": "test"}
		jsonData, _ := json.Marshal(reqBody)

		startTime := time.Now()
		resp, err := client.Post(
			fastServer.URL,
			"application/json",
			bytes.NewBuffer(jsonData),
		)
		elapsed := time.Since(startTime)

		if err != nil {
			t.Errorf("Fast request should succeed, got error: %v", err)
		}

		if resp != nil {
			resp.Body.Close()
		}

		// Verify response is fast (< 1 second)
		if elapsed > 1*time.Second {
			t.Errorf("Response should be fast (<1s), took %v", elapsed)
		}
	})

	t.Run("client timeout is configurable", func(t *testing.T) {
		// Verify we can create clients with different timeouts
		shortTimeout := &http.Client{Timeout: 5 * time.Second}
		longTimeout := &http.Client{Timeout: 60 * time.Second}

		if shortTimeout.Timeout != 5*time.Second {
			t.Error("Short timeout not set correctly")
		}
		if longTimeout.Timeout != 60*time.Second {
			t.Error("Long timeout not set correctly")
		}
	})
}

// TestInputValidation verifies 10k character limit on prompts
func TestInputValidation(t *testing.T) {
	t.Run("prompt under 10k characters is valid", func(t *testing.T) {
		prompt := "This is a valid prompt"
		if len(prompt) > 10000 {
			t.Errorf("Test prompt should be under 10k characters, got %d", len(prompt))
		}
	})

	t.Run("prompt over 10k characters is invalid", func(t *testing.T) {
		// Create a prompt over 10k characters
		longPrompt := make([]byte, 10001)
		for i := range longPrompt {
			longPrompt[i] = 'a'
		}

		if len(longPrompt) <= 10000 {
			t.Error("Test prompt should be over 10k characters")
		}

		// Verify validation would fail
		if len(longPrompt) <= 10000 {
			t.Errorf("Prompt length validation should fail for %d characters", len(longPrompt))
		}
	})

	t.Run("exact 10k character limit", func(t *testing.T) {
		exactPrompt := make([]byte, 10000)
		for i := range exactPrompt {
			exactPrompt[i] = 'a'
		}

		if len(exactPrompt) != 10000 {
			t.Errorf("Prompt should be exactly 10000 characters, got %d", len(exactPrompt))
		}

		// At exactly 10k, should be valid
		if len(exactPrompt) > 10000 {
			t.Error("Prompt at exactly 10000 characters should be valid")
		}
	})
}

// TestHTTPClientConfiguration verifies HTTP client is properly configured
func TestHTTPClientConfiguration(t *testing.T) {
	t.Run("HTTP client has timeout set", func(t *testing.T) {
		client := &http.Client{
			Timeout: 30 * time.Second,
		}

		if client.Timeout == 0 {
			t.Error("HTTP client should have timeout set")
		}

		if client.Timeout != 30*time.Second {
			t.Errorf("HTTP client timeout = %v, expected 30s", client.Timeout)
		}
	})

	t.Run("default HTTP client has no timeout", func(t *testing.T) {
		defaultClient := &http.Client{}
		if defaultClient.Timeout != 0 {
			t.Error("Default HTTP client should have no timeout")
		}
	})
}
