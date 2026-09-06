package logging

import (
	"crypto/sha256"
	"fmt"
)

// This file defines the single content-logging policy used across the router
// (issue #2464). Normal diagnostics carry correlation metadata, counts, timing,
// model identity, and approved hashes — never prompt, response, query, tool, or
// provider content. Log level is not a confidentiality boundary: a DEBUG-only
// dump of a prompt is still a content leak wherever debug logging is enabled.
//
// Call ContentDescriptor (or ContentDescriptorBytes) instead of interpolating
// user or provider content into a log line. The descriptor keeps the two things
// that make such a line useful for debugging — how large the content was, and
// whether two occurrences were identical — without disclosing the content.

// contentDescriptorPrefix marks a redacted content field so a log reader can
// tell withheld content from a value that was genuinely absent.
const contentDescriptorPrefix = "[content]"

// contentHashBytes is the number of SHA-256 bytes rendered in a descriptor.
// Eight bytes (16 hex chars) is enough to correlate repeated content within a
// trace without functioning as a practical content oracle.
const contentHashBytes = 8

// ContentDescriptor returns a content-free descriptor for user or provider
// content: its byte length and a truncated SHA-256 digest. Identical content
// yields an identical descriptor, so cache hits, retries, and duplicate
// requests remain correlatable in logs without exposing the content itself.
//
// Use this for prompts, queries, tool arguments, completions, and provider
// bodies. Do not use it for credentials: those must not be logged at all, not
// even as a hash.
func ContentDescriptor(content string) string {
	if content == "" {
		return contentDescriptorPrefix + " empty"
	}
	sum := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%s len=%d sha256=%x", contentDescriptorPrefix, len(content), sum[:contentHashBytes])
}

// ContentDescriptorBytes is ContentDescriptor for byte slices, so callers do
// not have to materialize a string copy of a body purely to log it.
func ContentDescriptorBytes(content []byte) string {
	if len(content) == 0 {
		return contentDescriptorPrefix + " empty"
	}
	sum := sha256.Sum256(content)
	return fmt.Sprintf("%s len=%d sha256=%x", contentDescriptorPrefix, len(content), sum[:contentHashBytes])
}
