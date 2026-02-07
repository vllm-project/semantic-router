//go:build openai_validation

// Integration tests that validate pkg/openai client behavior against the real OpenAI API.
// Run with: go test -tags=openai_validation ./src/semantic-router/pkg/openai -v
// Requires OPENAI_API_KEY. Optionally set OPENAI_BASE_URL (default https://api.openai.com).

package openai

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

// placeholderKeys are example/placeholder values; validation tests skip when these are set.
var placeholderKeys = []string{"sk-your-key", "sk-your-real-key", "sk-...", "your-api-key"}

func getValidationConfig(t *testing.T) (baseURL, apiKey string, skip bool) {
	apiKey = os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping OpenAI API validation tests")
		return "", "", true
	}
	apiKeyLower := strings.ToLower(strings.TrimSpace(apiKey))
	for _, p := range placeholderKeys {
		if apiKeyLower == p {
			t.Skip("OPENAI_API_KEY looks like a placeholder; skipping OpenAI API validation tests (set a real key to run)")
			return "", "", true
		}
	}
	if len(apiKey) < 20 {
		t.Skip("OPENAI_API_KEY too short (likely placeholder); skipping OpenAI API validation tests")
		return "", "", true
	}
	baseURL = os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	baseURL = strings.TrimSuffix(baseURL, "/v1")
	return baseURL, apiKey, false
}

func TestValidation_Files_List(t *testing.T) {
	baseURL, apiKey, skip := getValidationConfig(t)
	if skip {
		return
	}
	ctx := context.Background()
	client := NewFileStoreClient(baseURL, apiKey)
	list, err := client.ListFiles(ctx, "assistants")
	if err != nil {
		t.Fatalf("ListFiles: %v", err)
	}
	if list.Object == "" {
		t.Error("ListFiles: response missing object")
	}
	if list.Data == nil {
		t.Error("ListFiles: response data is nil")
	}
}

func TestValidation_Files_UploadGetDelete(t *testing.T) {
	baseURL, apiKey, skip := getValidationConfig(t)
	if skip {
		return
	}
	ctx := context.Background()
	client := NewFileStoreClient(baseURL, apiKey)

	content := []byte("OpenAI API validation test content.\n")
	file, err := client.UploadFile(ctx, bytes.NewReader(content), "validation_test.txt", "assistants")
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if file.ID == "" {
		t.Error("UploadFile: response missing id")
	}

	got, err := client.GetFile(ctx, file.ID)
	if err != nil {
		t.Fatalf("GetFile: %v", err)
	}
	if got.ID != file.ID {
		t.Errorf("GetFile: id mismatch %q vs %q", got.ID, file.ID)
	}

	err = client.DeleteFile(ctx, file.ID)
	if err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}
}

func TestValidation_VectorStores_List(t *testing.T) {
	baseURL, apiKey, skip := getValidationConfig(t)
	if skip {
		return
	}
	ctx := context.Background()
	client := NewVectorStoreClient(baseURL, apiKey)
	list, err := client.ListVectorStores(ctx, 5, "", "", "")
	if err != nil {
		t.Fatalf("ListVectorStores: %v", err)
	}
	if list.Object == "" {
		t.Error("ListVectorStores: response missing object")
	}
	if list.Data == nil {
		t.Error("ListVectorStores: response data is nil")
	}
}

func TestValidation_VectorStores_CreateGetUpdateDelete(t *testing.T) {
	baseURL, apiKey, skip := getValidationConfig(t)
	if skip {
		return
	}
	ctx := context.Background()
	client := NewVectorStoreClient(baseURL, apiKey)

	vs, err := client.CreateVectorStore(ctx, &CreateVectorStoreRequest{
		Name: "e2e-validation-test-vs",
	})
	if err != nil {
		t.Fatalf("CreateVectorStore: %v", err)
	}
	if vs.ID == "" {
		t.Error("CreateVectorStore: response missing id")
	}
	defer func() {
		_ = client.DeleteVectorStore(ctx, vs.ID)
	}()

	got, err := client.GetVectorStore(ctx, vs.ID)
	if err != nil {
		t.Fatalf("GetVectorStore: %v", err)
	}
	if got.ID != vs.ID {
		t.Errorf("GetVectorStore: id mismatch %q vs %q", got.ID, vs.ID)
	}

	_, err = client.UpdateVectorStore(ctx, vs.ID, &UpdateVectorStoreRequest{
		Name: "e2e-validation-test-vs-updated",
	})
	if err != nil {
		t.Fatalf("UpdateVectorStore: %v", err)
	}

	err = client.DeleteVectorStore(ctx, vs.ID)
	if err != nil {
		t.Fatalf("DeleteVectorStore: %v", err)
	}
}

func TestValidation_VectorStore_Search(t *testing.T) {
	baseURL, apiKey, skip := getValidationConfig(t)
	if skip {
		return
	}
	ctx := context.Background()
	client := NewVectorStoreClient(baseURL, apiKey)

	vs, err := client.CreateVectorStore(ctx, &CreateVectorStoreRequest{
		Name: "e2e-validation-search",
	})
	if err != nil {
		t.Fatalf("CreateVectorStore: %v", err)
	}
	defer func() {
		_ = client.DeleteVectorStore(ctx, vs.ID)
	}()

	resp, err := client.SearchVectorStore(ctx, vs.ID, "test query", 5, nil)
	if err != nil {
		t.Fatalf("SearchVectorStore: %v", err)
	}
	if resp.Object == "" {
		t.Error("SearchVectorStore: response missing object")
	}
	if resp.Data == nil {
		t.Error("SearchVectorStore: response data is nil")
	}
}
