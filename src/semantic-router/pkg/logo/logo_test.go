package logo

import (
	"io"
	"os"
	"strings"
	"testing"
)

func TestBuildVLLMLogoLinesMatchesCurrentWordmark(t *testing.T) {
	lines := buildVLLMLogoLines()
	rendered := strings.Join(lines, "\n")

	if !strings.Contains(rendered, "Semantic Router") {
		t.Fatalf("expected Semantic Router wordmark in logo, got %q", rendered)
	}
	if !strings.Contains(rendered, "local runtime") {
		t.Fatalf("expected local runtime tagline in logo, got %q", rendered)
	}
	if strings.Contains(rendered, "########") {
		t.Fatalf("expected legacy ASCII logo to be removed, got %q", rendered)
	}
	if !strings.Contains(rendered, "█") {
		t.Fatalf("expected block glyph wordmark in logo, got %q", rendered)
	}
}

func TestPrintVLLMLogoWritesBanner(t *testing.T) {
	output := captureStdout(t, PrintVLLMLogo)

	if !strings.Contains(output, "Semantic Router") {
		t.Fatalf("expected Semantic Router wordmark in output, got %q", output)
	}
	if !strings.Contains(output, "local runtime") {
		t.Fatalf("expected local runtime tagline in output, got %q", output)
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	originalStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}

	os.Stdout = writer
	fn()

	if closeErr := writer.Close(); closeErr != nil {
		t.Fatalf("close stdout writer: %v", closeErr)
	}
	os.Stdout = originalStdout

	output, readErr := io.ReadAll(reader)
	if readErr != nil {
		t.Fatalf("read stdout output: %v", readErr)
	}
	if closeErr := reader.Close(); closeErr != nil {
		t.Fatalf("close stdout reader: %v", closeErr)
	}

	return string(output)
}
