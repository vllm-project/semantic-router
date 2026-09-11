//go:build !windows && cgo

/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package apiserver

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"net/http"
	"path/filepath"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

const (
	megabyte             int64 = 1024 * 1024
	maxUploadSize              = 50 * megabyte
	multipartMemoryLimit       = 8 * 1024 * 1024
)

// allowedExtensions defines the document types that can be uploaded for
// vector-store ingestion.
var allowedExtensions = map[string]bool{
	".txt":  true,
	".md":   true,
	".json": true,
	".csv":  true,
	".html": true,
	".htm":  true,
}

// imageExtensions defines the image types accepted for purpose "vision". They
// exist so Response API input_image parts can reference an uploaded image by
// file_id; the router inlines the image for the selected backend.
var imageExtensions = map[string]bool{
	".png":  true,
	".jpg":  true,
	".jpeg": true,
	".gif":  true,
	".webp": true,
}

// purposeVision is the upload purpose for images used as model input, matching
// the OpenAI Files API value.
const purposeVision = "vision"

// uploadExtensionAllowed reports whether a file extension may be uploaded for
// the given purpose: images for vision, documents for everything else.
func uploadExtensionAllowed(ext, purpose string, documentExtensions map[string]bool) bool {
	if purpose == purposeVision {
		return imageExtensions[ext]
	}
	return documentExtensions[ext]
}

func allowedExtensionList(purpose string, documentExtensions map[string]bool) string {
	if purpose == purposeVision {
		return ".png, .jpg, .jpeg, .gif, .webp"
	}
	return strings.Join(slices.Sorted(maps.Keys(documentExtensions)), ", ")
}

func uploadMaxBytes(maxFileSizeMB int) int64 {
	if maxFileSizeMB <= 0 || int64(maxFileSizeMB) > math.MaxInt64/megabyte {
		return maxUploadSize
	}
	return int64(maxFileSizeMB) * megabyte
}

func (s *ClassificationAPIServer) uploadLimits() (int64, map[string]bool) {
	cfg := s.currentConfig()
	if cfg == nil || cfg.VectorStore == nil {
		return maxUploadSize, allowedExtensions
	}
	documentExtensions := allowedExtensions
	if len(cfg.VectorStore.SupportedFormats) > 0 {
		documentExtensions = make(map[string]bool, len(cfg.VectorStore.SupportedFormats))
		for _, format := range cfg.VectorStore.SupportedFormats {
			ext := strings.ToLower(strings.TrimSpace(format))
			if ext != "" && !strings.HasPrefix(ext, ".") {
				ext = "." + ext
			}
			documentExtensions[ext] = true
		}
	}
	return uploadMaxBytes(cfg.VectorStore.MaxFileSizeMB), documentExtensions
}

// SetFileStore sets the global file store for the API server.
func SetFileStore(fs *vectorstore.FileStore) {
	globalRuntimeDeps.setFileStore(fs)
}

func (s *ClassificationAPIServer) handleUploadFile(w http.ResponseWriter, r *http.Request) {
	fileStore := s.currentFileStore()
	if fileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	maxBytes, documentExtensions := s.uploadLimits()
	// Limit upload size.
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)

	if err := r.ParseMultipartForm(multipartMemoryLimit); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			s.writeErrorResponse(w, http.StatusRequestEntityTooLarge, "REQUEST_BODY_TOO_LARGE", errRequestBodyTooLarge.Error())
			return
		}
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT",
			fmt.Sprintf("failed to parse multipart form (max size: %dMB): %s", maxBytes/megabyte, err.Error()))
		return
	}
	if r.MultipartForm != nil {
		defer func() {
			if err := r.MultipartForm.RemoveAll(); err != nil {
				logging.ComponentWarnEvent("apiserver", "file_upload_cleanup_failed", map[string]interface{}{
					"error": err.Error(),
				})
			}
		}()
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file field is required")
		return
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			logging.ComponentWarnEvent("apiserver", "file_upload_close_failed", map[string]interface{}{
				"filename": header.Filename,
				"error":    closeErr.Error(),
			})
		}
	}()

	purpose := r.FormValue("purpose")
	if purpose == "" {
		purpose = "assistants"
	}

	// Validate extension against the purpose.
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !uploadExtensionAllowed(ext, purpose, documentExtensions) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_FILE_TYPE",
			fmt.Sprintf("unsupported file type for purpose %q: %s (allowed: %s)", purpose, ext, allowedExtensionList(purpose, documentExtensions)))
		return
	}

	record, err := fileStore.SaveFromReader(header.Filename, file, purpose)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "SAVE_ERROR", "failed to save file")
		return
	}

	s.writeJSONResponse(w, http.StatusOK, record)
}

func (s *ClassificationAPIServer) handleListFiles(w http.ResponseWriter, r *http.Request) {
	fileStore := s.currentFileStore()
	if fileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	purposeFilter := r.URL.Query().Get("purpose")
	records := fileStore.List()

	// Filter by purpose if specified.
	if purposeFilter != "" {
		filtered := make([]*vectorstore.FileRecord, 0)
		for _, r := range records {
			if r.Purpose == purposeFilter {
				filtered = append(filtered, r)
			}
		}
		records = filtered
	}

	response := map[string]interface{}{
		"object": "list",
		"data":   records,
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleGetFile(w http.ResponseWriter, r *http.Request) {
	fileStore := s.currentFileStore()
	if fileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/files/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	record, err := fileStore.Get(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, record)
}

func (s *ClassificationAPIServer) handleDeleteFile(w http.ResponseWriter, r *http.Request) {
	fileStore := s.currentFileStore()
	if fileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/files/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	if err := fileStore.Delete(id); err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"id":      id,
		"object":  "file",
		"deleted": true,
	})
}

func (s *ClassificationAPIServer) handleGetFileContent(w http.ResponseWriter, r *http.Request) {
	fileStore := s.currentFileStore()
	if fileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	// Extract file ID from /v1/files/{id}/content
	path := strings.TrimPrefix(r.URL.Path, "/v1/files/")
	id := strings.TrimSuffix(path, "/content")
	if id == "" || id == path {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	record, err := fileStore.Get(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", "file not found")
		return
	}

	content, err := fileStore.Read(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", "failed to read file content")
		return
	}

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", record.Filename))
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(content)
}
