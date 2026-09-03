// Command modelcompat generates and validates Router Model compatibility evidence.
// It is an explicit offline test/release tool and is not used by router startup.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/compatibility"
)

const usage = `Usage: modelcompat <command> [options]

Commands:
  qualify-candle-cpu  Run offline conformance against a local Candle classifier
  validate            Validate a compatibility receipt without running a model

qualify-candle-cpu requires:
  --model-path PATH          Local directory containing config, tokenizer, and weights
  --artifact-revision REV    Exact external model revision; never inferred from PATH
  --router-revision REV      Exact router revision under test
  --labels A,B               Labels in class-index order
  --suite PATH               Versioned qualification suite JSON
  [--device-profile PROFILE] Defaults to GOOS/GOARCH/cpu
  [--output PATH]            Defaults to stdout

validate requires one receipt path.
`

type nativeCandleRuntime struct{}

func (nativeCandleRuntime) Initialize(modelPath string, numClasses int, useCPU bool) error {
	return candle.InitGenericClassifier(modelPath, numClasses, useCPU)
}

func (nativeCandleRuntime) Classify(input string) (compatibility.ClassificationResult, error) {
	result, err := candle.ClassifyTextWithProbabilities(input)
	return compatibility.ClassificationResult{
		Class:         result.Class,
		Confidence:    result.Confidence,
		Probabilities: append([]float32(nil), result.Probabilities...),
		NumClasses:    result.NumClasses,
	}, err
}

func main() {
	if err := run(os.Args[1:], nativeCandleRuntime{}, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "modelcompat:", err)
		os.Exit(1)
	}
}

func run(args []string, candleRuntime compatibility.CandleRuntime, stdout io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("command is required\n%s", usage)
	}
	switch args[0] {
	case "qualify-candle-cpu":
		return runQualifyCandleCPU(args[1:], candleRuntime, stdout)
	case "validate":
		return runValidate(args[1:], stdout)
	case "help", "-h", "--help":
		_, err := fmt.Fprint(stdout, usage)
		return err
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}

func runQualifyCandleCPU(
	args []string,
	candleRuntime compatibility.CandleRuntime,
	stdout io.Writer,
) error {
	flags := flag.NewFlagSet("qualify-candle-cpu", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	modelPath := flags.String("model-path", "", "local Candle model directory")
	artifactRevision := flags.String("artifact-revision", "", "exact model artifact revision")
	routerRevision := flags.String("router-revision", "", "exact tested router revision")
	labelsText := flags.String("labels", "", "comma-separated labels in class-index order")
	deviceProfile := flags.String(
		"device-profile",
		runtime.GOOS+"/"+runtime.GOARCH+"/cpu",
		"tested device profile",
	)
	suitePath := flags.String("suite", "", "qualification suite JSON path")
	outputPath := flags.String("output", "-", "receipt output path, or - for stdout")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}

	required := []struct {
		name  string
		value string
	}{
		{"--model-path", *modelPath},
		{"--artifact-revision", *artifactRevision},
		{"--router-revision", *routerRevision},
		{"--labels", *labelsText},
		{"--suite", *suitePath},
	}
	for _, value := range required {
		if strings.TrimSpace(value.value) == "" {
			return fmt.Errorf("%s is required", value.name)
		}
	}

	labels, err := parseLabels(*labelsText)
	if err != nil {
		return err
	}
	artifactDigest, err := compatibility.DigestLocalCandleArtifact(*modelPath)
	if err != nil {
		return err
	}
	suiteData, err := os.ReadFile(*suitePath)
	if err != nil {
		return fmt.Errorf("read qualification suite: %w", err)
	}
	suite, err := compatibility.ParseQualificationSuite(suiteData, labels)
	if err != nil {
		return err
	}

	subject := compatibility.CandleClassifierSubject{
		SchemaVersion:    compatibility.SubjectSchemaVersionV1,
		ArtifactRevision: *artifactRevision,
		ArtifactDigest:   artifactDigest,
		TaskContract:     compatibility.LabelDistributionContractV1,
		Connector:        compatibility.CandleConnectorV1,
		Precision:        "float32",
		Provider:         "cpu",
		DeviceProfile:    *deviceProfile,
		RouterRevision:   *routerRevision,
		Labels:           labels,
	}
	receipt, err := compatibility.QualifyLocalCandleCPU(
		subject,
		*modelPath,
		suite,
		candleRuntime,
	)
	if err != nil {
		return err
	}
	if err := writeReceipt(*outputPath, receipt, stdout); err != nil {
		return err
	}
	if failed := compatibility.FailedCheckNames(receipt); len(failed) != 0 {
		return fmt.Errorf("qualification checks failed: %s", strings.Join(failed, ", "))
	}
	return nil
}

func runValidate(args []string, stdout io.Writer) error {
	flags := flag.NewFlagSet("validate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("validate requires exactly one receipt path")
	}
	data, err := os.ReadFile(flags.Arg(0))
	if err != nil {
		return fmt.Errorf("read compatibility receipt: %w", err)
	}
	receipt, err := compatibility.ParseReceipt(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(
		stdout,
		"valid receipt %s; failed checks: %d\n",
		receipt.SubjectDigest,
		len(compatibility.FailedCheckNames(receipt)),
	)
	return err
}

func parseLabels(value string) ([]string, error) {
	parts := strings.Split(value, ",")
	labels := make([]string, 0, len(parts))
	for index, part := range parts {
		label := strings.TrimSpace(part)
		if label == "" {
			return nil, fmt.Errorf("--labels entry %d is empty", index)
		}
		labels = append(labels, label)
	}
	if len(labels) < 2 {
		return nil, fmt.Errorf("--labels must contain at least two entries")
	}
	return labels, nil
}

func writeReceipt(path string, receipt compatibility.Receipt, stdout io.Writer) error {
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return fmt.Errorf("encode compatibility receipt: %w", err)
	}
	data = append(data, '\n')
	if path == "-" {
		_, err = stdout.Write(data)
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write compatibility receipt: %w", err)
	}
	return nil
}
