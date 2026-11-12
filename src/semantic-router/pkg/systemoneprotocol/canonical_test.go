package systemoneprotocol

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"go/format"
	"os"
	"os/exec"
	"sync"
	"testing"
)

type goldenCase struct {
	Name      string `json:"name"`
	Input     string `json:"input"`
	Canonical string `json:"canonical"`
}

func goldenCases(t testing.TB) []goldenCase {
	t.Helper()
	data, err := os.ReadFile("testdata/canonical_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []goldenCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	return golden.Cases
}

// The golden outputs are Python's json.dumps(json.loads(input),
// ensure_ascii=False), written by the schema generator.
func TestCanonicalMatchesPythonGolden(t *testing.T) {
	for _, c := range goldenCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			got, err := CanonicalJSON([]byte(c.Input))
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Canonical {
				t.Fatalf("canonical\n got %q\nwant %q", got, c.Canonical)
			}
		})
	}
}

func TestParseKeepsOrderAndPythonDuplicates(t *testing.T) {
	value, err := Parse([]byte(`{"b": 1, "a": {"z": true, "y": null}, "b": 2}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := Canonical(value); got != `{"b": 2, "a": {"z": true, "y": null}}` {
		t.Fatalf("got %s", got)
	}
	compact, _ := value.MarshalJSON()
	if string(compact) != `{"b":2,"a":{"z":true,"y":null}}` {
		t.Fatalf("compact %s", compact)
	}
}

func TestObjectEditsKeepPlaces(t *testing.T) {
	object := Object(Member{"a", Int(1)}, Member{"b", Int(2)}, Member{"c", Int(3)})
	object = object.With("b", String("two")).With("d", Bool(false)).Without("a")
	if got := Canonical(object); got != `{"b": "two", "c": 3, "d": false}` {
		t.Fatalf("got %s", got)
	}
	if !object.Equal(Object(Member{"b", String("two")}, Member{"c", Int(3)}, Member{"d", Bool(false)})) {
		t.Fatal("equal objects compare unequal")
	}
}

func TestFloatFormsAShortestLiteral(t *testing.T) {
	value, err := Float(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if literal, _ := value.Literal(); literal != "0.1" {
		t.Fatalf("literal %s", literal)
	}
	if _, err := Float(1.0 / zero()); err == nil {
		t.Fatal("infinity is not a JSON number")
	}
}

func zero() float64 { return 0 }

func TestGeneratedTablesAreGofmtClean(t *testing.T) {
	data, err := os.ReadFile("zz_generated_fields.go")
	if err != nil {
		t.Fatal(err)
	}
	formatted, err := format.Source(data)
	if err != nil {
		t.Fatal(err)
	}
	if string(formatted) != string(data) {
		t.Fatal("zz_generated_fields.go is not gofmt-clean; fix the generator")
	}
}

func TestTheFieldRegistry(t *testing.T) {
	roles, ok := ExtensionField(LevelRequest, "roles")
	if !ok || roles.StandardTarget != TargetStrip || roles.HonouredBy("decision3") {
		t.Fatalf("roles: %+v", roles)
	}
	head, ok := ExtensionField(LevelQuestion, "head")
	if !ok || head.StandardTarget != TargetRefuse || !head.AppliesToQuestionType("span") || head.AppliesToQuestionType("choice") {
		t.Fatalf("head: %+v", head)
	}
	if !IsStandardField(LevelRequest, "state") || IsStandardField(LevelRequest, "roles") {
		t.Fatal("standard fields")
	}
	if !IsStandardField(LevelQuestionType, "box") || IsStandardField(LevelQuestionType, "span") {
		t.Fatal("standard question types")
	}
}

// pythonOracle answers json.dumps(json.loads(text), ensure_ascii=False) for
// one input at a time, from one python3 process per test binary.
type pythonOracle struct {
	mu     sync.Mutex
	input  *bufio.Writer
	output *bufio.Scanner
}

const oracleScript = `
import base64, json, sys
for line in sys.stdin:
    try:
        text = base64.b64decode(line).decode("utf-8")
        value = json.loads(text)
        out = json.dumps(value, ensure_ascii=False)
        out.encode("utf-8")
        print(json.dumps({"ok": True, "canonical": out}), flush=True)
    except Exception:
        print(json.dumps({"ok": False}), flush=True)
`

var (
	oracleOnce sync.Once
	oracle     *pythonOracle
)

func startOracle(t testing.TB) *pythonOracle {
	oracleOnce.Do(func() {
		python, err := exec.LookPath("python3")
		if err != nil {
			return
		}
		command := exec.Command(python, "-c", oracleScript)
		stdin, err := command.StdinPipe()
		if err != nil {
			return
		}
		stdout, err := command.StdoutPipe()
		if err != nil || command.Start() != nil {
			return
		}
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 1<<20), 64<<20)
		oracle = &pythonOracle{input: bufio.NewWriter(stdin), output: scanner}
	})
	if oracle == nil {
		t.Skip("python3 is not available as the oracle")
	}
	return oracle
}

func (o *pythonOracle) canonical(text []byte) (string, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	_, _ = o.input.WriteString(base64.StdEncoding.EncodeToString(text) + "\n")
	_ = o.input.Flush()
	if !o.output.Scan() {
		return "", false
	}
	var answer struct {
		OK        bool   `json:"ok"`
		Canonical string `json:"canonical"`
	}
	if json.Unmarshal(o.output.Bytes(), &answer) != nil {
		return "", false
	}
	return answer.Canonical, answer.OK
}

// FuzzCanonicalMatchesPython checks the encoder against Python itself on
// every input both read: the seeds are the golden inputs, and go test -fuzz
// explores from them.
func FuzzCanonicalMatchesPython(f *testing.F) {
	for _, c := range goldenCases(f) {
		f.Add([]byte(c.Input))
	}
	f.Add([]byte(`{"k": "\u2028\u0085\ud83d\ude00", "n": [1e22, 1e-7, 123.456e5]}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		value, err := Parse(input)
		if err != nil || !json.Valid(input) {
			return
		}
		want, ok := startOracle(t).canonical(input)
		if !ok {
			return
		}
		if got := Canonical(value); got != want {
			t.Fatalf("input %q\n got %q\nwant %q", input, got, want)
		}
	})
}
