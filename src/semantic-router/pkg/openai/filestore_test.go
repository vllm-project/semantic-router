package openai

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFileStoreClient_UploadFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/files" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Missing or invalid Authorization header")
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"file-123","object":"file","bytes":1024,"created_at":1234567890,"filename":"test.pdf","purpose":"assistants","status":"uploaded"}`))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	fileReader := bytes.NewReader([]byte("test file content"))

	file, err := client.UploadFile(context.Background(), fileReader, "test.pdf", "assistants")
	if err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}

	if file.ID != "file-123" {
		t.Errorf("Expected file ID 'file-123', got '%s'", file.ID)
	}
	if file.Filename != "test.pdf" {
		t.Errorf("Expected filename 'test.pdf', got '%s'", file.Filename)
	}
	if file.Purpose != "assistants" {
		t.Errorf("Expected purpose 'assistants', got '%s'", file.Purpose)
	}
}

func TestFileStoreClient_ListFiles(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/files" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"file-1","object":"file","bytes":1024,"created_at":1234567890,"filename":"test1.pdf","purpose":"assistants"}],"has_more":false}`))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	list, err := client.ListFiles(context.Background(), "assistants")
	if err != nil {
		t.Fatalf("ListFiles failed: %v", err)
	}

	if len(list.Data) != 1 {
		t.Errorf("Expected 1 file, got %d", len(list.Data))
	}
	if list.Data[0].ID != "file-1" {
		t.Errorf("Expected file ID 'file-1', got '%s'", list.Data[0].ID)
	}
}

func TestFileStoreClient_GetFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/files/file-123" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"file-123","object":"file","bytes":1024,"created_at":1234567890,"filename":"test.pdf","purpose":"assistants"}`))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	file, err := client.GetFile(context.Background(), "file-123")
	if err != nil {
		t.Fatalf("GetFile failed: %v", err)
	}

	if file.ID != "file-123" {
		t.Errorf("Expected file ID 'file-123', got '%s'", file.ID)
	}
}

func TestFileStoreClient_DeleteFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "DELETE" || r.URL.Path != "/v1/files/file-123" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"file-123","object":"file","deleted":true}`))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	err := client.DeleteFile(context.Background(), "file-123")
	if err != nil {
		t.Fatalf("DeleteFile failed: %v", err)
	}
}

func TestFileStoreClient_DownloadFile(t *testing.T) {
	expectedContent := "test file content"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/files/file-123/content" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(expectedContent))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	content, err := client.DownloadFile(context.Background(), "file-123")
	if err != nil {
		t.Fatalf("DownloadFile failed: %v", err)
	}

	if string(content) != expectedContent {
		t.Errorf("Expected content '%s', got '%s'", expectedContent, string(content))
	}
}

func TestFileStoreClient_ErrorHandling(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid request"}}`))
	}))
	defer server.Close()

	client := NewFileStoreClient(server.URL, "test-key")
	fileReader := bytes.NewReader([]byte("test"))

	_, err := client.UploadFile(context.Background(), fileReader, "test.pdf", "assistants")
	if err == nil {
		t.Error("Expected error for bad request, got nil")
	}
}
