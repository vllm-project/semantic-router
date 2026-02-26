package memory

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSanitizeMemoryContent_Clean(t *testing.T) {
	content := "User budget for Hawaii trip is $10,000"
	result, err := sanitizeMemoryContent(content)
	require.NoError(t, err)
	assert.Equal(t, content, result)
}

func TestSanitizeMemoryContent_Empty(t *testing.T) {
	_, err := sanitizeMemoryContent("")
	assert.Error(t, err)

	_, err = sanitizeMemoryContent("   ")
	assert.Error(t, err)
}

func TestSanitizeMemoryContent_InvalidUTF8(t *testing.T) {
	_, err := sanitizeMemoryContent("hello \xff world")
	assert.Error(t, err)
}

func TestSanitizeMemoryContent_Truncation(t *testing.T) {
	long := strings.Repeat("a", maxMemoryContentBytes+1000)
	result, err := sanitizeMemoryContent(long)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(result), maxMemoryContentBytes)
}

func TestTruncateUTF8(t *testing.T) {
	s := "Hello, 世界!"
	truncated := truncateUTF8(s, 10)
	assert.LessOrEqual(t, len(truncated), 10)
	assert.True(t, strings.HasPrefix(s, truncated))
}
