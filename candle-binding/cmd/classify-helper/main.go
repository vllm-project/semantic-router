// Command classify-helper is a persistent NDJSON worker wrapping the Candle
// mmBERT-32K modality classifier for the #3856 same-run harness
// (same_run_harness.py's --binding candle path).
//
// Protocol: one JSON object per line on stdin, one JSON object per line on
// stdout, in request order.
//
//	request:  {"text": "...", "max_length": 256}
//	response: {"label": "AR", "seq_len": 0, "tokenize_ns": 0,
//	           "forward_ns": 5000000, "helper_cpu_s": 0.482,
//	           "helper_rss_mb": 412.3}
//	error:    {"error": "..."}
//
// helper_cpu_s and helper_rss_mb are this process's own cumulative
// getrusage(RUSAGE_SELF) readings (CPU seconds, lifetime-peak RSS in MB),
// reported on every response so the Python harness can attribute this
// process's real resource cost instead of only measuring its own pipe I/O.
//
// seq_len and tokenize_ns are always 0: ClassifyMmBert32KModality does not
// expose a token count or a separate tokenize/forward split, so this helper
// does not fabricate one — forward_ns covers the full classify call.
//
// Build:
//
//	cd candle-binding && go build -o candle-classify ./cmd/classify-helper/
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
)

var hubIDPattern = regexp.MustCompile(`^[\w.-]+/[\w.-]+$`)

// checkLocalModelDir gives an actionable error before hitting the Rust FFI
// boundary. Unlike --binding hf (which auto-downloads via huggingface_hub),
// the candle C ABI only accepts a local directory — passing a Hub ID like
// "org/model" fails deep inside Rust with a bare "file not found:
// org/model/config.json", which does not point at the actual fix.
func checkLocalModelDir(model string) error {
	if _, err := os.Stat(filepath.Join(model, "config.json")); err == nil {
		return nil
	}
	hint := ""
	if hubIDPattern.MatchString(model) {
		hint = fmt.Sprintf(
			"\n%q looks like a Hugging Face Hub ID. --binding candle requires a "+
				"local directory, not a Hub ID (it does not auto-download).\n"+
				"Download one first:\n"+
				"  huggingface-cli download %s --local-dir /path/to/local-model\n"+
				"Then pass: --model /path/to/local-model",
			model, model,
		)
	}
	return fmt.Errorf(
		"classify-helper: %s/config.json not found%s",
		model, hint,
	)
}

type request struct {
	Text      string `json:"text"`
	MaxLength int    `json:"max_length"`
}

type response struct {
	Label       string  `json:"label,omitempty"`
	SeqLen      int     `json:"seq_len"`
	TokenizeNs  int64   `json:"tokenize_ns"`
	ForwardNs   int64   `json:"forward_ns,omitempty"`
	HelperCPUs  float64 `json:"helper_cpu_s"`
	HelperRSSMB float64 `json:"helper_rss_mb"`
	Error       string  `json:"error,omitempty"`
}

func helperResourceUsage() (cpuSeconds float64, rssMB float64) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0, 0
	}
	cpuSeconds = float64(ru.Utime.Sec) + float64(ru.Utime.Usec)/1e6 +
		float64(ru.Stime.Sec) + float64(ru.Stime.Usec)/1e6

	divisor := 1024.0 // Linux: ru_maxrss is in KB.
	if runtime.GOOS == "darwin" {
		divisor = 1024.0 * 1024.0 // Darwin: ru_maxrss is in bytes.
	}
	rssMB = float64(ru.Maxrss) / divisor
	return cpuSeconds, rssMB
}

func emit(w *bufio.Writer, resp response) {
	data, err := json.Marshal(resp)
	if err != nil {
		// Marshalling our own struct should never fail; if it does, there is
		// nothing more useful to report than the error itself.
		fmt.Fprintf(os.Stderr, "failed to marshal response: %v\n", err)
		return
	}
	w.Write(data)
	w.WriteByte('\n')
	w.Flush()
}

func main() {
	model := flag.String("model", "", "Local directory holding the mmBERT-32K modality model (config.json, model.safetensors, tokenizer files) — required")
	flag.Int("max-length", 256, "Default max tokenisation length (accepted for CLI parity; per-request max_length is not forwarded to the C ABI, which manages its own context window)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		os.Exit(1)
	}

	if err := checkLocalModelDir(*model); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "classify-helper: initializing %s\n", *model)
	if err := candle.InitMmBert32KModalityClassifier(*model, true); err != nil {
		fmt.Fprintf(os.Stderr, "classify-helper: init failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "classify-helper: ready")

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var req request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			emit(writer, response{Error: fmt.Sprintf("invalid request: %v", err)})
			continue
		}

		t0 := time.Now()
		result, err := candle.ClassifyMmBert32KModality(req.Text)
		forwardNs := time.Since(t0).Nanoseconds()
		cpuSeconds, rssMB := helperResourceUsage()

		if err != nil {
			emit(writer, response{
				Error:       err.Error(),
				HelperCPUs:  cpuSeconds,
				HelperRSSMB: rssMB,
			})
			continue
		}

		emit(writer, response{
			Label:       result.Modality,
			SeqLen:      0,
			TokenizeNs:  0,
			ForwardNs:   forwardNs,
			HelperCPUs:  cpuSeconds,
			HelperRSSMB: rssMB,
		})
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "classify-helper: stdin read error: %v\n", err)
		os.Exit(1)
	}
}
