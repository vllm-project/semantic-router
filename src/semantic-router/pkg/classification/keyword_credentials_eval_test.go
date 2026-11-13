package classification

// Precision and recall of the credential keyword rules in
// config/fragments/signal/keyword/credentials.yaml, measured through the real
// keyword matcher on Samsung CredData (https://github.com/Samsung/CredData,
// Apache-2.0) and on a negative corpus such as this repository's own tree.
//
//	SR_CREDDATA_DIR=/path/to/CredData \
//	SR_CREDENTIAL_NEGATIVE_DIR=/path/to/semantic-router \
//	SR_CREDENTIAL_EVAL_REPORT=/tmp/credential-eval.md \
//	go test ./pkg/classification/ -run TestCredentialKeywordRulesAccuracy -v
//
// The test is skipped when SR_CREDDATA_DIR is unset. CredData's labeled values
// are obfuscated with same-shape replacements; the report counts how many
// labeled AWS keys stopped matching only because the replacement left the
// key's alphabet, so that loss is not read as a detector miss.

import (
	"encoding/csv"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type credDataRow struct {
	file                 string
	lineStart, lineEnd   int
	valueStart, valueEnd int
	truth                string
	category             string
}

type lineSpan struct{ start, end int }

func (s lineSpan) intersects(o lineSpan) bool { return s.start <= o.end && o.start <= s.end }

// credentialRuleForCategory maps a CredData category to the rule expected to
// detect it. Categories outside the rules' reach return "".
func credentialRuleForCategory(category string) string {
	switch {
	case strings.Contains(category, "PEM Private Key") && !strings.Contains(category, "BASE64 encoded PEM"):
		return "credential_private_keys"
	case strings.Contains(category, "AWS Client ID"), strings.Contains(category, "Google API Key"):
		return "credential_cloud_keys"
	case strings.Contains(category, "Slack Token"), strings.Contains(category, "Stripe Credentials"),
		strings.Contains(category, "Github Classic Token"), strings.Contains(category, "Anthropic API Key"):
		return "credential_access_tokens"
	}
	return ""
}

// awsAccessKeyPrefixes are the prefixes gitleaks' aws-access-token rule
// accepts. CredData also labels IAM role, policy and user identifiers
// (AROA, ANPA, AIDA, ...) as "AWS Client ID"; those are not secrets.
var awsAccessKeyPrefixes = regexp.MustCompile(`^(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)`)

func loadCredentialFragmentRules(t *testing.T) []config.KeywordRule {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test filename")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(filename), "../../../.."))
	data, err := os.ReadFile(filepath.Join(root, "config", "fragments", "signal", "keyword", "credentials.yaml"))
	if err != nil {
		t.Fatalf("read credentials fragment: %v", err)
	}
	var fragment struct {
		Recipes []struct {
			Signals struct {
				Keywords []config.KeywordRule `yaml:"keywords"`
			} `yaml:"signals"`
		} `yaml:"recipes"`
	}
	if err := yaml.Unmarshal(data, &fragment); err != nil {
		t.Fatalf("parse credentials fragment: %v", err)
	}
	if len(fragment.Recipes) != 1 || len(fragment.Recipes[0].Signals.Keywords) == 0 {
		t.Fatal("credentials fragment declares no keyword rules in its one recipe")
	}
	return fragment.Recipes[0].Signals.Keywords
}

// TestCredentialFragmentRulesCompileAndReadToolResults is the always-on part
// of the evaluation: every fragment rule compiles and is scoped to tool results.
func TestCredentialFragmentRulesCompileAndReadToolResults(t *testing.T) {
	rules := loadCredentialFragmentRules(t)
	for _, rule := range rules {
		if !rule.Input.ToolResults || rule.Input.ReadsRequest() {
			t.Fatalf("rule %q input = %#v, want tool_results true and request false", rule.Name, rule.Input)
		}
		if rule.Method != "regex" || !rule.CaseSensitive || rule.Operator != "OR" {
			t.Fatalf("rule %q must be a case-sensitive regex OR rule, got %+v", rule.Name, rule)
		}
	}
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = rules
	classifier := &Classifier{Config: cfg, keywordClassifier: kc}
	// Public example values, not live credentials: the AWS documentation key
	// id, a GitHub token of the documented shape, and a truncated PEM block.
	samples := map[string]string{
		"credential_cloud_keys":    "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
		"credential_access_tokens": "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123",
		"credential_private_keys":  "-----BEGIN RSA PRIVATE KEY-----\n" + strings.Repeat("MIIEowIBAAKCAQEA0123456789abcdefghijklmnopqrstuvwxyzABCDEFGH\n", 2) + "-----END RSA PRIVATE KEY-----",
	}
	for want, text := range samples {
		results, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
			Text: "cat the file", CurrentUserText: "cat the file", ForceEvaluateAll: true,
			ToolResultTexts: []string{text},
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(results.MatchedKeywordRules) != 1 || results.MatchedKeywordRules[0] != want {
			t.Fatalf("sample for %s: MatchedKeywordRules = %v", want, results.MatchedKeywordRules)
		}
		if got, _ := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{Text: text, CurrentUserText: text, ForceEvaluateAll: true}); len(got.MatchedKeywordRules) != 0 {
			t.Fatalf("sample for %s in the prompt must not match, got %v", want, got.MatchedKeywordRules)
		}
	}
}

func readCredDataMeta(t *testing.T, metaDir string) []credDataRow {
	t.Helper()
	entries, err := os.ReadDir(metaDir)
	if err != nil {
		t.Fatalf("read CredData meta dir: %v", err)
	}
	var rows []credDataRow
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".csv") {
			continue
		}
		f, err := os.Open(filepath.Join(metaDir, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		records, err := csv.NewReader(f).ReadAll()
		f.Close()
		if err != nil {
			t.Fatalf("parse %s: %v", entry.Name(), err)
		}
		if len(records) == 0 {
			continue
		}
		col := map[string]int{}
		for i, name := range records[0] {
			col[name] = i
		}
		atoi := func(s string) int {
			n, err := strconv.Atoi(s)
			if err != nil {
				return -1
			}
			return n
		}
		for _, rec := range records[1:] {
			rows = append(rows, credDataRow{
				file:       rec[col["FilePath"]],
				lineStart:  atoi(rec[col["LineStart"]]),
				lineEnd:    atoi(rec[col["LineEnd"]]),
				valueStart: atoi(rec[col["ValueStart"]]),
				valueEnd:   atoi(rec[col["ValueEnd"]]),
				truth:      rec[col["GroundTruth"]],
				category:   rec[col["Category"]],
			})
		}
	}
	return rows
}

// lineSpanOf converts a byte range of content into 1-based line numbers.
func lineSpanOf(content string, start, end int) lineSpan {
	first := strings.Count(content[:start], "\n") + 1
	last := first + strings.Count(content[start:end], "\n")
	return lineSpan{first, last}
}

func lineOf(content string, line int) string {
	lines := strings.Split(content, "\n")
	if line < 1 || line > len(lines) {
		return ""
	}
	return lines[line-1]
}

// credDataFileResult is one labeled file's contribution to the evaluation,
// computed in a worker and summed by the test.
type credDataFileResult struct {
	flagged              map[string]int // rule -> flagged lines
	truePositive         map[string]int // rule -> flagged lines that carry a true label
	fileTruePositive     map[string]bool
	recallRows           map[string]int // rule -> in-scope true rows
	recallDetected       map[string]int // rule -> in-scope true rows the rule flagged
	trueRows             int
	trueRowsFlagged      int
	awsKeyRows           int
	awsKeyDetected       int
	awsKeyAlphabetLoss   int
	awsIdentifierRows    int
	falsePositiveSamples []string
	err                  error
}

var looseAWSKey = regexp.MustCompile(`(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Za-z0-9]{16}`)

func evaluateCredDataFile(credDataDir, file string, fileRows []credDataRow, kc *KeywordClassifier, patterns map[string][]*regexp.Regexp, ruleNames []string) credDataFileResult {
	result := credDataFileResult{
		flagged: map[string]int{}, truePositive: map[string]int{}, fileTruePositive: map[string]bool{},
		recallRows: map[string]int{}, recallDetected: map[string]int{},
	}
	data, err := os.ReadFile(filepath.Join(credDataDir, file))
	if err != nil {
		result.err = fmt.Errorf("read %s: %w", file, err)
		return result
	}
	content := string(data)
	matches, _, err := kc.MatchAllWithSources(KeywordMatchInput{ToolResultTexts: []string{content}})
	if err != nil {
		result.err = err
		return result
	}
	fired := map[string]bool{}
	for _, match := range matches {
		fired[match.RuleName] = true
	}

	trueRowSpans := make([]lineSpan, 0, len(fileRows))
	for _, row := range fileRows {
		if row.truth == "T" {
			trueRowSpans = append(trueRowSpans, lineSpan{row.lineStart, row.lineEnd})
		}
	}
	// Attribute the classifier's verdict to lines with the same compiled
	// patterns. A rule that did not fire has no spans by construction.
	flaggedByRule := map[string][]lineSpan{}
	for _, name := range ruleNames {
		if !fired[name] {
			continue
		}
		seen := map[int]bool{}
		for _, re := range patterns[name] {
			for _, loc := range re.FindAllStringIndex(content, -1) {
				span := lineSpanOf(content, loc[0], loc[1])
				if seen[span.start] {
					continue
				}
				seen[span.start] = true
				flaggedByRule[name] = append(flaggedByRule[name], span)
			}
		}
		if len(flaggedByRule[name]) == 0 {
			result.err = fmt.Errorf("%s: rule %s fired but no pattern matched", file, name)
			return result
		}
		for _, span := range flaggedByRule[name] {
			result.flagged[name]++
			tp := false
			for _, truth := range trueRowSpans {
				if span.intersects(truth) {
					tp = true
					break
				}
			}
			if tp {
				result.truePositive[name]++
				result.fileTruePositive[name] = true
			} else if len(result.falsePositiveSamples) < 3 {
				result.falsePositiveSamples = append(result.falsePositiveSamples,
					fmt.Sprintf("%s %s:%d %q", name, file, span.start, truncateForReport(lineOf(content, span.start), 100)))
			}
		}
	}

	for _, row := range fileRows {
		if row.truth != "T" {
			continue
		}
		result.trueRows++
		rowSpan := lineSpan{row.lineStart, row.lineEnd}
		flaggedBy := func(rule string) bool {
			for _, span := range flaggedByRule[rule] {
				if span.intersects(rowSpan) {
					return true
				}
			}
			return false
		}
		for _, name := range ruleNames {
			if flaggedBy(name) {
				result.trueRowsFlagged++
				break
			}
		}
		expected := credentialRuleForCategory(row.category)
		if expected == "" {
			continue
		}
		if strings.Contains(row.category, "AWS Client ID") {
			line := lineOf(content, row.lineStart)
			value := line
			if row.valueStart >= 0 && row.valueEnd > row.valueStart && row.valueEnd <= len(line) {
				value = line[row.valueStart:row.valueEnd]
			}
			if !awsAccessKeyPrefixes.MatchString(value) {
				result.awsIdentifierRows++
				continue // an IAM identifier, not an access key: out of scope
			}
			result.awsKeyRows++
			if flaggedBy(expected) {
				result.awsKeyDetected++
			} else if looseAWSKey.MatchString(line) {
				result.awsKeyAlphabetLoss++
			}
		}
		result.recallRows[expected]++
		if flaggedBy(expected) {
			result.recallDetected[expected]++
		}
	}
	return result
}

func TestCredentialKeywordRulesAccuracy(t *testing.T) {
	credDataDir := os.Getenv("SR_CREDDATA_DIR")
	if credDataDir == "" {
		t.Skip("SR_CREDDATA_DIR not set; see the file comment for how to run the evaluation")
	}
	rules := loadCredentialFragmentRules(t)
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatal(err)
	}
	patterns := map[string][]*regexp.Regexp{}
	ruleNames := make([]string, 0, len(rules))
	for _, rule := range rules {
		ruleNames = append(ruleNames, rule.Name)
		for _, keyword := range rule.Keywords {
			patterns[rule.Name] = append(patterns[rule.Name], regexp.MustCompile(keyword))
		}
	}

	rows := readCredDataMeta(t, filepath.Join(credDataDir, "meta"))
	byFile := map[string][]credDataRow{}
	for _, row := range rows {
		byFile[row.file] = append(byFile[row.file], row)
	}
	files := make([]string, 0, len(byFile))
	for file := range byFile {
		files = append(files, file)
	}
	sort.Strings(files)

	// Files are independent; scan them on every CPU and sum in order.
	results := make([]credDataFileResult, len(files))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i, file := range files {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, file string) {
			defer wg.Done()
			defer func() { <-sem }()
			results[i] = evaluateCredDataFile(credDataDir, file, byFile[file], kc, patterns, ruleNames)
		}(i, file)
	}
	wg.Wait()

	type counts struct{ flagged, truePositive, filesFlagged, filesTruePositive int }
	perRule := map[string]*counts{}
	type categoryRecall struct{ rows, detected int }
	recall := map[string]*categoryRecall{}
	for _, name := range ruleNames {
		perRule[name] = &counts{}
		recall[name] = &categoryRecall{}
	}
	var falsePositiveSamples []string
	totalTrueRows, totalTrueRowsFlagged := 0, 0
	awsKeyRows, awsKeyDetected, awsKeyAlphabetLoss, awsIdentifierRows := 0, 0, 0, 0
	for _, r := range results {
		if r.err != nil {
			t.Fatal(r.err)
		}
		for _, name := range ruleNames {
			c := perRule[name]
			c.flagged += r.flagged[name]
			c.truePositive += r.truePositive[name]
			if r.flagged[name] > 0 {
				c.filesFlagged++
			}
			if r.fileTruePositive[name] {
				c.filesTruePositive++
			}
			recall[name].rows += r.recallRows[name]
			recall[name].detected += r.recallDetected[name]
		}
		totalTrueRows += r.trueRows
		totalTrueRowsFlagged += r.trueRowsFlagged
		awsKeyRows += r.awsKeyRows
		awsKeyDetected += r.awsKeyDetected
		awsKeyAlphabetLoss += r.awsKeyAlphabetLoss
		awsIdentifierRows += r.awsIdentifierRows
		if len(falsePositiveSamples) < 60 {
			falsePositiveSamples = append(falsePositiveSamples, r.falsePositiveSamples...)
		}
	}

	var report strings.Builder
	fmt.Fprintf(&report, "CredData: %d labeled files, %d labeled lines, %d labeled true\n\n", len(files), len(rows), totalTrueRows)
	fmt.Fprintf(&report, "| rule | flagged lines | labeled true | precision | flagged files | files with a true line |\n|---|---|---|---|---|---|\n")
	for _, name := range ruleNames {
		c := perRule[name]
		fmt.Fprintf(&report, "| %s | %d | %d | %s | %d | %d |\n", name, c.flagged, c.truePositive, ratio(c.truePositive, c.flagged), c.filesFlagged, c.filesTruePositive)
	}
	fmt.Fprintf(&report, "\n| rule | in-scope true rows | detected | recall |\n|---|---|---|---|\n")
	for _, name := range ruleNames {
		r := recall[name]
		fmt.Fprintf(&report, "| %s | %d | %d | %s |\n", name, r.rows, r.detected, ratio(r.detected, r.rows))
	}
	fmt.Fprintf(&report, "\nAWS access keys (AKIA/ASIA/ABIA/ACCA/A3T prefixes): %d labeled, %d detected, %d undetected only because the obfuscated value left the key alphabet; %d further rows are IAM identifiers (AROA/ANPA/AIDA/...) that gitleaks excludes on purpose.\n",
		awsKeyRows, awsKeyDetected, awsKeyAlphabetLoss, awsIdentifierRows)
	fmt.Fprintf(&report, "All labeled-true lines flagged by any rule: %d of %d (%s); the rules target prefixed provider keys and PEM blocks, not generic passwords.\n", totalTrueRowsFlagged, totalTrueRows, ratio(totalTrueRowsFlagged, totalTrueRows))
	if len(falsePositiveSamples) > 0 {
		fmt.Fprintf(&report, "\nFlagged lines without a true label (first %d):\n", len(falsePositiveSamples))
		for _, sample := range falsePositiveSamples {
			fmt.Fprintf(&report, "  %s\n", sample)
		}
	}

	if negativeDir := os.Getenv("SR_CREDENTIAL_NEGATIVE_DIR"); negativeDir != "" {
		filesScanned, hits := scanNegativeCorpus(t, kc, patterns, negativeDir)
		fmt.Fprintf(&report, "\nNegative corpus %s: %d text files scanned as one tool result each, %d flagged\n", negativeDir, filesScanned, len(hits))
		for _, hit := range hits {
			fmt.Fprintf(&report, "  %s\n", hit)
		}
	}

	t.Logf("\n%s", report.String())
	if path := os.Getenv("SR_CREDENTIAL_EVAL_REPORT"); path != "" {
		if err := os.WriteFile(path, []byte(report.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func scanNegativeCorpus(t *testing.T, kc *KeywordClassifier, patterns map[string][]*regexp.Regexp, dir string) (int, []string) {
	t.Helper()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatalf("open negative corpus: %v", err)
	}
	defer root.Close()
	var hits []string
	scanned := 0
	err = fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", ".venv", ".venv-agent", "__pycache__":
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil || !info.Mode().IsRegular() || info.Size() > 2<<20 {
			return nil
		}
		data, err := root.ReadFile(path)
		if err != nil || !utf8.Valid(data) {
			return nil
		}
		scanned++
		content := string(data)
		matches, _, err := kc.MatchAllWithSources(KeywordMatchInput{ToolResultTexts: []string{content}})
		if err != nil {
			return err
		}
		for _, match := range matches {
			for _, re := range patterns[match.RuleName] {
				if loc := re.FindStringIndex(content); loc != nil {
					span := lineSpanOf(content, loc[0], loc[1])
					hits = append(hits, fmt.Sprintf("%s %s:%d %q", match.RuleName, path, span.start, truncateForReport(lineOf(content, span.start), 100)))
					break
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk negative corpus: %v", err)
	}
	return scanned, hits
}

func ratio(num, den int) string {
	if den == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(num)/float64(den))
}

func truncateForReport(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
