//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func TestHandleUploadFileStreamsToFileStore(t *testing.T) {
	fileStore, err := vectorstore.NewFileStore(t.TempDir(), vectorstore.NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatalf("failed to create file store: %v", err)
	}

	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{FileStore: fileStore})
	apiServer := &ClassificationAPIServer{runtimeRegistry: registry}

	payload := []byte("hello from streamed multipart upload")
	body, contentType := buildMultipartUpload(t, "file", "notes.txt", payload, map[string]string{
		"purpose": "assistants",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/files", body)
	req.Header.Set("Content-Type", contentType)

	rr := httptest.NewRecorder()
	apiServer.handleUploadFile(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var record vectorstore.FileRecord
	if decodeErr := json.Unmarshal(rr.Body.Bytes(), &record); decodeErr != nil {
		t.Fatalf("failed to decode response: %v", decodeErr)
	}
	if record.Bytes != int64(len(payload)) {
		t.Fatalf("expected uploaded byte count %d, got %d", len(payload), record.Bytes)
	}

	stored, err := fileStore.Read(record.ID)
	if err != nil {
		t.Fatalf("failed to read stored file: %v", err)
	}
	if !bytes.Equal(stored, payload) {
		t.Fatalf("stored file content mismatch: got %q, want %q", stored, payload)
	}
}

func buildMultipartUpload(
	t *testing.T,
	fieldName string,
	filename string,
	content []byte,
	fields map[string]string,
) (*bytes.Buffer, string) {
	t.Helper()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		t.Fatalf("failed to create multipart file: %v", err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatalf("failed to write multipart content: %v", err)
	}
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			t.Fatalf("failed to write multipart field %s: %v", key, err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close multipart writer: %v", err)
	}

	return body, writer.FormDataContentType()
}

func newFileUploadServer(t *testing.T) (*ClassificationAPIServer, *vectorstore.FileStore) {
	t.Helper()
	fileStore, err := vectorstore.NewFileStore(t.TempDir(), vectorstore.NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatalf("failed to create file store: %v", err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{FileStore: fileStore})
	return &ClassificationAPIServer{runtimeRegistry: registry}, fileStore
}

func uploadFile(t *testing.T, apiServer *ClassificationAPIServer, filename string, content []byte, purpose string) *httptest.ResponseRecorder {
	t.Helper()
	body, contentType := buildMultipartUpload(t, "file", filename, content, map[string]string{"purpose": purpose})
	req := httptest.NewRequest(http.MethodPost, "/v1/files", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	apiServer.handleUploadFile(rr, req)
	return rr
}

func TestHandleUploadFileAcceptsImagesForVisionPurpose(t *testing.T) {
	apiServer, fileStore := newFileUploadServer(t)
	png := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

	rr := uploadFile(t, apiServer, "cat.png", png, "vision")
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var record vectorstore.FileRecord
	if err := json.Unmarshal(rr.Body.Bytes(), &record); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if record.Purpose != "vision" {
		t.Fatalf("expected purpose vision, got %q", record.Purpose)
	}
	stored, err := fileStore.Read(record.ID)
	if err != nil {
		t.Fatalf("failed to read stored file: %v", err)
	}
	if !bytes.Equal(stored, png) {
		t.Fatalf("stored image mismatch")
	}
}

func TestHandleUploadFileRejectsImagesForDocumentPurposes(t *testing.T) {
	apiServer, _ := newFileUploadServer(t)

	rr := uploadFile(t, apiServer, "cat.png", []byte("\x89PNG\r\n\x1a\n"), "assistants")
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
	if !bytes.Contains(rr.Body.Bytes(), []byte("unsupported file type")) {
		t.Fatalf("expected unsupported file type error, got %s", rr.Body.String())
	}
}

func TestHandleUploadFileRejectsDocumentsForVisionPurpose(t *testing.T) {
	apiServer, _ := newFileUploadServer(t)

	rr := uploadFile(t, apiServer, "notes.txt", []byte("not an image"), "vision")
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}
