package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestVectorStoreClient_CreateVectorStore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/vector_stores" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"vs_123","object":"vector_store","created_at":1234567890,"name":"test-store","file_counts":{"total":0,"completed":0,"in_progress":0,"failed":0,"cancelled":0}}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	req := &CreateVectorStoreRequest{
		Name:    "test-store",
		FileIDs: []string{"file-1", "file-2"},
	}

	store, err := client.CreateVectorStore(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateVectorStore failed: %v", err)
	}

	if store.ID != "vs_123" {
		t.Errorf("Expected vector store ID 'vs_123', got '%s'", store.ID)
	}
	if store.Name != "test-store" {
		t.Errorf("Expected name 'test-store', got '%s'", store.Name)
	}
}

func TestVectorStoreClient_SearchVectorStore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/vector_stores/vs_123/search" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}

		// Verify request body
		var reqBody map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		if reqBody["query"] != "test query" {
			t.Errorf("Expected query 'test query', got '%v'", reqBody["query"])
		}

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"object":"list",
			"data":[
				{"content":"Test content 1","filename":"test1.pdf","score":0.95},
				{"content":"Test content 2","filename":"test2.pdf","score":0.90}
			]
		}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	results, err := client.SearchVectorStore(context.Background(), "vs_123", "test query", 10, nil)
	if err != nil {
		t.Fatalf("SearchVectorStore failed: %v", err)
	}

	if len(results.Data) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results.Data))
	}
	if results.Data[0].Content != "Test content 1" {
		t.Errorf("Expected content 'Test content 1', got '%s'", results.Data[0].Content)
	}
	if results.Data[0].Score != 0.95 {
		t.Errorf("Expected score 0.95, got %f", results.Data[0].Score)
	}
}

func TestVectorStoreClient_ListVectorStores(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/vector_stores" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"vs_1","object":"vector_store","created_at":1234567890,"name":"store1","file_counts":{"total":0,"completed":0,"in_progress":0,"failed":0,"cancelled":0}}],"has_more":false}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	list, err := client.ListVectorStores(context.Background(), 10, "desc", "", "")
	if err != nil {
		t.Fatalf("ListVectorStores failed: %v", err)
	}

	if len(list.Data) != 1 {
		t.Errorf("Expected 1 vector store, got %d", len(list.Data))
	}
	if list.Data[0].ID != "vs_1" {
		t.Errorf("Expected vector store ID 'vs_1', got '%s'", list.Data[0].ID)
	}
}

func TestVectorStoreClient_GetVectorStore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/vector_stores/vs_123" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"vs_123","object":"vector_store","created_at":1234567890,"name":"test-store","file_counts":{"total":5,"completed":5,"in_progress":0,"failed":0,"cancelled":0}}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	store, err := client.GetVectorStore(context.Background(), "vs_123")
	if err != nil {
		t.Fatalf("GetVectorStore failed: %v", err)
	}

	if store.ID != "vs_123" {
		t.Errorf("Expected vector store ID 'vs_123', got '%s'", store.ID)
	}
	if store.FileCounts.Total != 5 {
		t.Errorf("Expected 5 total files, got %d", store.FileCounts.Total)
	}
}

func TestVectorStoreClient_CreateVectorStoreFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/vector_stores/vs_123/files" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"vsf_123","object":"vector_store.file","created_at":1234567890,"vector_store_id":"vs_123","status":"completed"}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	file, err := client.CreateVectorStoreFile(context.Background(), "vs_123", "file-123")
	if err != nil {
		t.Fatalf("CreateVectorStoreFile failed: %v", err)
	}

	if file.ID != "vsf_123" {
		t.Errorf("Expected file ID 'vsf_123', got '%s'", file.ID)
	}
	if file.VectorStoreID != "vs_123" {
		t.Errorf("Expected vector store ID 'vs_123', got '%s'", file.VectorStoreID)
	}
}

func TestVectorStoreClient_ErrorHandling(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"Vector store not found"}}`))
	}))
	defer server.Close()

	client := NewVectorStoreClient(server.URL, "test-key")
	_, err := client.GetVectorStore(context.Background(), "vs_invalid")
	if err == nil {
		t.Error("Expected error for not found, got nil")
	}
}
