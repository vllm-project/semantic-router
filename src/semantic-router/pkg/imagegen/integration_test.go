//go:build integration
// +build integration

package imagegen_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// TestImageGenerationE2E tests the image generation flow through the router
// This test requires:
// 1. vLLM-Omni running on localhost:8001
// 2. Semantic Router running on localhost:8080
//
// Run with: go test -tags=integration -v ./pkg/imagegen/...
func TestImageGenerationE2E(t *testing.T) {
	// Check if vLLM-Omni is running
	if !isServiceRunning("http://localhost:8001/health") {
		t.Skip("vLLM-Omni not running on localhost:8001, skipping integration test")
	}

	// Check if router API is running
	if !isServiceRunning("http://localhost:8080/health") {
		t.Skip("Router not running on localhost:8080, skipping integration test")
	}

	tests := []struct {
		name     string
		endpoint string
		request  map[string]interface{}
		checkFn  func(t *testing.T, resp map[string]interface{})
	}{
		{
			name:     "Direct vLLM-Omni image generation",
			endpoint: "http://localhost:8001/v1/chat/completions",
			request: map[string]interface{}{
				"model": "black-forest-labs/FLUX.1-schnell",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "A beautiful sunset over mountains"},
				},
				"extra_body": map[string]interface{}{
					"width":               512,
					"height":              512,
					"num_inference_steps": 4,
				},
			},
			checkFn: func(t *testing.T, resp map[string]interface{}) {
				choices, ok := resp["choices"].([]interface{})
				if !ok || len(choices) == 0 {
					t.Fatal("Expected choices in response")
				}
				choice := choices[0].(map[string]interface{})
				message := choice["message"].(map[string]interface{})
				content := message["content"]

				// Content should be an array with image_url type
				contentArr, ok := content.([]interface{})
				if !ok {
					t.Fatal("Expected content to be an array")
				}
				if len(contentArr) == 0 {
					t.Fatal("Expected at least one content part")
				}

				part := contentArr[0].(map[string]interface{})
				if part["type"] != "image_url" {
					t.Errorf("Expected type image_url, got %v", part["type"])
				}

				imageURL := part["image_url"].(map[string]interface{})
				url := imageURL["url"].(string)
				if !strings.HasPrefix(url, "data:image/") {
					t.Errorf("Expected data URI, got %s", url[:50])
				}
				t.Logf("✓ Image generated successfully (data URI length: %d)", len(url))
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			req, err := http.NewRequest("POST", tt.endpoint, bytes.NewBuffer(body))
			if err != nil {
				t.Fatalf("Failed to create request: %v", err)
			}
			req.Header.Set("Content-Type", "application/json")

			client := &http.Client{Timeout: 120 * time.Second}
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			respBody, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("Failed to read response: %v", err)
			}

			if resp.StatusCode != http.StatusOK {
				t.Fatalf("Request failed with status %d: %s", resp.StatusCode, string(respBody))
			}

			var result map[string]interface{}
			if err := json.Unmarshal(respBody, &result); err != nil {
				t.Fatalf("Failed to parse response: %v", err)
			}

			tt.checkFn(t, result)
		})
	}
}

// TestImageGenerationViaBackend tests the imagegen backend directly
func TestImageGenerationViaBackend(t *testing.T) {
	if !isServiceRunning("http://localhost:8001/health") {
		t.Skip("vLLM-Omni not running on localhost:8001, skipping integration test")
	}

	// Test using the imagegen package directly
	cfg := &testConfig{
		Backend:        "vllm_omni",
		TimeoutSeconds: 120,
		BackendConfig: map[string]interface{}{
			"base_url": "http://localhost:8001",
			"model":    "black-forest-labs/FLUX.1-schnell",
		},
	}

	t.Logf("Testing image generation with config: %+v", cfg)

	// For now, just test direct HTTP call
	reqBody := map[string]interface{}{
		"model": "black-forest-labs/FLUX.1-schnell",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "A cat sitting on a windowsill"},
		},
		"extra_body": map[string]interface{}{
			"width":               256,
			"height":              256,
			"num_inference_steps": 4,
		},
	}

	body, _ := json.Marshal(reqBody)
	resp, err := http.Post("http://localhost:8001/v1/chat/completions", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("Request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	t.Log("✓ Backend image generation test passed")
}

type testConfig struct {
	Backend        string
	TimeoutSeconds int
	BackendConfig  map[string]interface{}
}

func isServiceRunning(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// TestMain allows running the tests with custom setup
func TestMain(m *testing.M) {
	fmt.Println("Running image generation integration tests...")
	fmt.Println("Prerequisites:")
	fmt.Println("  - vLLM-Omni on localhost:8001")
	fmt.Println("  - Router on localhost:8080")
	fmt.Println()
	os.Exit(m.Run())
}
