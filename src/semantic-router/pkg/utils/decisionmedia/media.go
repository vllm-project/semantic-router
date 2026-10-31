// Package decisionmedia validates inline attachments without fetching URLs or
// decoding image pixels or video frames. Sources are private to one request.
package decisionmedia

import (
	"bytes"
	"encoding/base64"
	"strings"
	"sync"
)

const (
	ImageMaxBytes = 8_000_000
	VideoMaxBytes = 32_000_000
	// Reserve space for state, questions and the bundle envelope. The transport
	// also checks the complete serialized body before sending it.
	EncodedBudget = 47 << 20
)

// Source references neutral content. Its fields are omitted from JSON to keep
// attachments out of accidental diagnostic and replay serialization.
type Source struct {
	Kind      string `json:"-"`
	URL       string `json:"-"`
	MediaType string `json:"-"`
	Data      string `json:"-"`
	once      sync.Once
	value     string
	size      int
	reason    string
}

type (
	Limit  struct{ Count, Bytes int }
	Policy struct {
		Images, Videos *Limit
		EncodedBytes   int
	}
)

type (
	Selection struct{ Images, Videos []string }
	Skip      func(kind, reason string)
)

// Select keeps the most recent valid attachments, restoring their input order
// within each modality. A Source is decoded at most once across deployments.
func Select(sources []*Source, policy Policy, skip Skip) Selection {
	var result Selection
	budget := policy.EncodedBytes
	if budget <= 0 || budget > EncodedBudget {
		budget = EncodedBudget
	}
	for i := len(sources) - 1; i >= 0; i-- {
		source := sources[i]
		var limit *Limit
		var target *[]string
		switch source.Kind {
		case "image":
			limit, target = policy.Images, &result.Images
		case "video":
			limit, target = policy.Videos, &result.Videos
		default:
			continue
		}
		if limit == nil {
			continue
		}
		reason := ""
		if len(*target) >= limit.Count {
			reason = "over_count"
		} else if source.exceedsEncodedLimit(limit.Bytes) {
			reason = "too_large"
		} else {
			source.once.Do(source.validate)
			reason = source.reason
			if reason == "" && (source.size > limit.Bytes || len(source.value)+3 > budget) {
				reason = "too_large"
			}
		}
		if reason != "" {
			if skip != nil {
				skip(source.Kind, reason)
			}
			continue
		}
		*target = append(*target, source.value)
		budget -= len(source.value) + 3
	}
	reverse(result.Images)
	reverse(result.Videos)
	return result
}

func reverse(values []string) {
	for i, j := 0, len(values)-1; i < j; i, j = i+1, j-1 {
		values[i], values[j] = values[j], values[i]
	}
}

func (s *Source) validate() {
	mime, payload := strings.ToLower(strings.TrimSpace(s.MediaType)), s.Data
	if s.URL != "" {
		header, body, ok := strings.Cut(s.URL, ",")
		lower := strings.ToLower(header)
		if strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://") {
			s.reason = "remote_url"
			return
		}
		if !ok || !strings.HasPrefix(lower, "data:") || !strings.HasSuffix(lower, ";base64") {
			s.reason = "malformed"
			return
		}
		mime, payload = strings.TrimSuffix(strings.TrimPrefix(lower, "data:"), ";base64"), body
	}
	if !allowed(s.Kind, mime) {
		s.reason = "unsupported_type"
		return
	}
	capBytes := ImageMaxBytes
	if s.Kind == "video" {
		capBytes = VideoMaxBytes
	}
	// Reject excessive encoded input before allocating a decoded buffer.
	if len(payload) > base64.StdEncoding.EncodedLen(capBytes) {
		s.reason = "too_large"
		return
	}
	data, err := base64.StdEncoding.Strict().DecodeString(payload)
	if err != nil || len(data) == 0 {
		s.reason = "malformed"
		return
	}
	if len(data) > capBytes {
		s.reason = "too_large"
		return
	}
	if !signature(mime, data) {
		s.reason = "malformed"
		return
	}
	// Canonical base64 prevents accepted CR/LF from expanding the JSON body.
	s.size = len(data)
	s.value = "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func allowed(kind, mime string) bool {
	if kind == "image" {
		return mime == "image/png" || mime == "image/jpeg" || mime == "image/webp"
	}
	return kind == "video" && (mime == "video/mp4" || mime == "video/webm" || mime == "video/quicktime" || mime == "video/x-matroska")
}

func signature(mime string, data []byte) bool {
	switch mime {
	case "image/png":
		return bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n"))
	case "image/jpeg":
		return bytes.HasPrefix(data, []byte{0xff, 0xd8, 0xff})
	case "image/webp":
		return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
	case "video/mp4", "video/quicktime":
		if len(data) < 12 || string(data[4:8]) != "ftyp" {
			return false
		}
		if mime == "video/quicktime" {
			return string(data[8:12]) == "qt  "
		}
		return string(data[8:12]) != "qt  "
	case "video/webm", "video/x-matroska":
		if !bytes.HasPrefix(data, []byte{0x1a, 0x45, 0xdf, 0xa3}) {
			return false
		}
		// The EBML DocType element identifies the container without decoding frames.
		header := data[:min(len(data), 4096)]
		if mime == "video/webm" {
			return bytes.Contains(header, []byte{0x42, 0x82, 0x84, 'w', 'e', 'b', 'm'})
		}
		return bytes.Contains(header, []byte{0x42, 0x82, 0x88, 'm', 'a', 't', 'r', 'o', 's', 'k', 'a'})
	}
	return false
}

// Apply lower deployment caps before decoding, without poisoning the shared
// validation result for another deployment with a larger allowance.
func (s *Source) exceedsEncodedLimit(limit int) bool {
	capBytes := ImageMaxBytes
	if s.Kind == "video" {
		capBytes = VideoMaxBytes
	}
	if limit < 0 || limit >= capBytes {
		return false
	}
	payload := s.Data
	if s.URL != "" {
		header, body, ok := strings.Cut(s.URL, ",")
		if !ok || !strings.HasPrefix(strings.ToLower(header), "data:") {
			return false
		}
		payload = body
	}
	return len(payload) > base64.StdEncoding.EncodedLen(limit)
}
