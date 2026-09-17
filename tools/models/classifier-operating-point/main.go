// classifier-operating-point binds an explicit score policy to final native files.
package main

import (
	"context"
	"crypto/sha256"
	"flag"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
)

func main() {
	model := flag.String("model", "", "Directory containing final config, tokenizer and model.safetensors")
	source := flag.String("policy", "", "Runtime-only v1/v2 score policy; thresholds are never selected here")
	output := flag.String("output", "", "New version-2 sidecar path (must not already exist)")
	flag.Parse()
	if err := run(*model, *source, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(model, source, output string) error {
	if model == "" || source == "" || output == "" {
		return fmt.Errorf("--model, --policy and --output are required")
	}
	data, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	data, err = operatingpoint.BindArtifact(context.Background(), data, model)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(data)
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	fmt.Printf("%x  %s\n", sha256.Sum256(data), output)
	return nil
}
