package systemone

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// zstdMaxWindow is the largest window a zstd request body may use, as on
// System One.
const zstdMaxWindow = 8 << 20

// decodeSlots bounds the request bodies being decompressed at once: each may
// grow to RequestLimit from a few kilobytes on the wire.
var decodeSlots = make(chan struct{}, 4)

// wireError is a public answer for a body the handler cannot read.
type wireError struct {
	status  int
	code    string
	reason  string
	message string
}

func (e *wireError) Error() string { return e.message }

func malformedBody(reason, format string, args ...any) *wireError {
	return &wireError{status: http.StatusBadRequest, code: "invalid_request", reason: reason, message: fmt.Sprintf(format, args...)}
}

// requestEncoding reads Content-Encoding as System One does: "" for none or
// identity, "gzip" for gzip or x-gzip, "zstd"; another coding, or more than
// one, is 415.
func requestEncoding(h http.Header) (string, *wireError) {
	var codings []string
	for _, value := range h.Values("Content-Encoding") {
		for _, coding := range strings.Split(value, ",") {
			if coding = strings.ToLower(strings.TrimSpace(coding)); coding != "" && coding != "identity" {
				codings = append(codings, coding)
			}
		}
	}
	unsupported := &wireError{
		status: http.StatusUnsupportedMediaType, code: "unsupported_media_type", reason: "encoding_unsupported",
		message: "send the body uncompressed, or with one Content-Encoding: gzip or zstd",
	}
	switch {
	case len(codings) == 0:
		return "", nil
	case len(codings) > 1:
		return "", unsupported
	case codings[0] == "gzip" || codings[0] == "x-gzip":
		return "gzip", nil
	case codings[0] == "zstd":
		return "zstd", nil
	}
	return "", unsupported
}

// decodeBody decompresses one layer of encoding into at most RequestLimit bytes.
func decodeBody(ctx interface{ Done() <-chan struct{} }, encoding string, wire []byte) ([]byte, *wireError) {
	select {
	case decodeSlots <- struct{}{}:
	case <-ctx.Done():
		return nil, malformedBody("encoding_malformed", "the client closed the request")
	}
	defer func() { <-decodeSlots }()
	tooLarge := &wireError{status: http.StatusRequestEntityTooLarge, code: "request_too_large", message: "request too large once decompressed"}
	var reader io.Reader
	var err error
	switch encoding {
	case "gzip":
		var gz *gzip.Reader
		if gz, err = gzip.NewReader(bytes.NewReader(wire)); err == nil {
			defer gz.Close()
			reader = gz
		}
	case "zstd":
		var header zstd.Header
		if header.Decode(wire) == nil && header.HasFCS && header.FrameContentSize > RequestLimit {
			return nil, tooLarge
		}
		var zr *zstd.Decoder
		zr, err = zstd.NewReader(bytes.NewReader(wire), zstd.WithDecoderConcurrency(1), zstd.WithDecoderLowmem(true),
			zstd.WithDecoderMaxWindow(zstdMaxWindow), zstd.WithDecoderMaxMemory(max(RequestLimit, zstdMaxWindow)))
		if zr != nil {
			defer zr.Close()
		}
		reader = zr
	}
	var out bytes.Buffer
	var n int64
	if err == nil {
		n, err = out.ReadFrom(io.LimitReader(reader, RequestLimit+1))
	}
	switch {
	case err == nil && n <= RequestLimit:
		return out.Bytes(), nil
	case err == nil, errors.Is(err, zstd.ErrDecoderSizeExceeded):
		return nil, tooLarge
	case errors.Is(err, zstd.ErrWindowSizeExceeded):
		return nil, malformedBody("encoding_malformed", "zstd request bodies may use a window of at most 8 MiB")
	}
	return nil, malformedBody("encoding_malformed", "the request body is not valid %s data", encoding)
}

// multipartBoundary is the boundary of a multipart/form-data body, "" for
// any other body.
func multipartBoundary(contentType string) (string, *wireError) {
	if contentType == "" {
		return "", nil
	}
	media, params, err := mime.ParseMediaType(contentType)
	if err != nil || media != "multipart/form-data" {
		return "", nil
	}
	boundary := params["boundary"]
	if boundary == "" || len(boundary) > 70 {
		return "", malformedBody("multipart_malformed", "a multipart/form-data body needs a boundary")
	}
	return boundary, nil
}

var imagePartTypes = map[string]bool{"image/png": true, "image/jpeg": true, "image/jpg": true, "application/octet-stream": true}

// canonicalMultipart turns a System One multipart body into the JSON request
// the runtime reads: the first part, request, with each asset that names a
// part holding that part's bytes as data_base64. Every image part must be
// claimed by exactly one asset.
func canonicalMultipart(body []byte, boundary string) ([]byte, *wireError) {
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var request []byte
	parts := map[string][]byte{}
	for index := 0; ; index++ {
		part, err := reader.NextRawPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, malformedBody("multipart_malformed", "the multipart body is malformed")
		}
		name := part.FormName()
		if part.Header.Get("Content-Disposition") == "" || name == "" {
			return nil, malformedBody("multipart_malformed", "every multipart part needs a form-data name")
		}
		data, err := io.ReadAll(part)
		if err != nil {
			return nil, malformedBody("multipart_malformed", "the multipart body is malformed")
		}
		contentType := ""
		if value := part.Header.Get("Content-Type"); value != "" {
			if contentType, _, err = mime.ParseMediaType(value); err != nil {
				return nil, malformedBody("multipart_malformed", "a multipart part has a malformed content type")
			}
		}
		if index == 0 {
			if name != "request" || (contentType != "" && contentType != "application/json") {
				return nil, malformedBody("multipart_malformed", "the first multipart part must be request, application/json")
			}
			request = data
			continue
		}
		if name == "request" {
			return nil, malformedBody("multipart_malformed", "multipart part request appears more than once")
		}
		if _, seen := parts[name]; seen {
			return nil, malformedBody("multipart_malformed", "a multipart part name appears more than once")
		}
		if contentType != "" && !imagePartTypes[contentType] {
			return nil, malformedBody("multipart_malformed", "image parts are image/png, image/jpeg or application/octet-stream")
		}
		parts[name] = data
	}
	if request == nil {
		return nil, malformedBody("multipart_malformed", "the first multipart part must be request, application/json")
	}
	var members map[string]json.RawMessage
	if json.Unmarshal(request, &members) != nil || members == nil {
		return nil, malformedBody("multipart_malformed", "the request part is not a JSON object")
	}
	var assets map[string]map[string]json.RawMessage
	if raw, present := members["assets"]; present && json.Unmarshal(raw, &assets) != nil {
		return nil, malformedBody("asset_malformed", "assets must be an object of references to images")
	}
	claimed := map[string]bool{}
	for reference, entry := range assets {
		raw, named := entry["part"]
		if !named {
			continue
		}
		var name string
		if json.Unmarshal(raw, &name) != nil || !hasPart(parts, name) {
			return nil, malformedBody("multipart_malformed", "an asset names a part the body does not have")
		}
		if claimed[name] {
			return nil, malformedBody("multipart_malformed", "a multipart part is claimed twice")
		}
		claimed[name] = true
		encoded, _ := json.Marshal(base64.StdEncoding.EncodeToString(parts[name]))
		delete(entry, "part")
		entry["data_base64"] = encoded
		assets[reference] = entry
	}
	if len(claimed) != len(parts) {
		return nil, malformedBody("multipart_malformed", "a multipart part is claimed by no asset")
	}
	if assets != nil {
		encoded, err := encodeJSON(assets)
		if err != nil {
			return nil, malformedBody("asset_malformed", "assets must be an object of references to images")
		}
		members["assets"] = encoded
	}
	canonical, err := encodeJSON(members)
	if err != nil {
		return nil, malformedBody("multipart_malformed", "the request part is not a JSON object")
	}
	return canonical, nil
}

// encodeJSON is json.Marshal without HTML escaping, so references such as
// <image:...> keep their bytes.
func encodeJSON(value any) ([]byte, error) {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimRight(out.Bytes(), "\n"), nil
}

func hasPart(parts map[string][]byte, name string) bool {
	_, present := parts[name]
	return present
}
