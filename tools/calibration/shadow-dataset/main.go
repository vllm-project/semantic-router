// shadow-dataset publishes a shadow comparison manifest, and optionally the
// judgments returned for it, to a storage destination.
//
// Everything is validated before anything is written, and everything written is
// named by its content, so a published file is never replaced by a different
// one. Judge tasks are read to check the judgments against but never written:
// they carry prompt and answer text, and the manifest and judgments do not. The
// blinding key the tasks were built with is read to check them against the
// manifest, and never written either.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/shadowdataset"
)

// Destination stores published files. Names are slash-separated and derived
// from content, so an implementation only has to refuse replacing a name with
// different bytes.
type Destination interface {
	Put(name string, body []byte) error
}

// DirectoryDestination writes under a root directory, which covers a local
// disk and any object store mounted as a filesystem.
type DirectoryDestination struct {
	Root string
}

// Put writes body under name. The file is linked into place from a temporary
// file, so a reader never sees a partial file and a concurrent writer cannot
// replace it. Publishing the same bytes twice is a no-op.
func (d DirectoryDestination) Put(name string, body []byte) error {
	target := filepath.Join(d.Root, filepath.FromSlash(name))
	if existing, err := os.ReadFile(target); err == nil {
		if bytes.Equal(existing, body) {
			return nil
		}
		return fmt.Errorf("%s already holds different content", target)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(target), ".publish-*")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name())
	if _, err = temp.Write(body); err != nil {
		temp.Close()
		return err
	}
	if err = temp.Close(); err != nil {
		return err
	}
	return os.Link(temp.Name(), target)
}

func main() {
	dest := flag.String("dest", "", "Directory to publish into")
	manifestPath := flag.String("manifest", "", "Manifest from GET /api/v1/observability/replays/dataset")
	tasksPath := flag.String("tasks", "", "Judge tasks the judgments answer, read for validation and never published")
	judgmentsPath := flag.String("judgments", "", "Judgment set to validate and publish with its report")
	key := flag.String("blinding-key", "", "Key the judge tasks were built with, used to check them against the manifest and never published")
	flag.Parse()
	if *dest == "" {
		fmt.Fprintln(os.Stderr, "--dest is required")
		os.Exit(1)
	}
	names, err := publish(DirectoryDestination{Root: *dest}, *manifestPath, *tasksPath, *judgmentsPath, *key)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, name := range names {
		fmt.Println(name)
	}
}

// publishedFile is one validated file and the name it is published under.
type publishedFile struct {
	name string
	body []byte
}

// publish validates the inputs and writes them, returning the names written.
// Nothing is written unless every input validates.
func publish(dest Destination, manifestPath, tasksPath, judgmentsPath, key string) ([]string, error) {
	if manifestPath == "" {
		return nil, fmt.Errorf("--manifest is required")
	}
	if (tasksPath == "") != (judgmentsPath == "") || (tasksPath == "") != (key == "") {
		return nil, fmt.Errorf("--tasks, --judgments and --blinding-key are given together")
	}
	var manifest shadowdataset.Manifest
	if err := readJSON(manifestPath, &manifest); err != nil {
		return nil, err
	}
	if err := shadowdataset.Validate(manifest); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	body, err := encode(manifest)
	if err != nil {
		return nil, err
	}
	// The digest leaves out the counts, so two exports of the same examples can
	// differ in what they excluded. The file is named by its bytes as well.
	files := []publishedFile{{name: "manifests/" + manifest.Digest + "/" + digestOf(body) + ".json", body: body}}

	if judgmentsPath != "" {
		judged, judgedErr := judgedFiles(manifest, tasksPath, judgmentsPath, key)
		if judgedErr != nil {
			return nil, judgedErr
		}
		files = append(files, judged...)
	}

	names := make([]string, 0, len(files))
	for _, file := range files {
		if err = dest.Put(file.name, file.body); err != nil {
			return names, err
		}
		names = append(names, file.name)
	}
	return names, nil
}

// judgedFiles validates a judgment set against the tasks it answers and the
// manifest they came from. The set is named by the digest of its bytes, under
// its manifest, and its report is named after the set.
func judgedFiles(manifest shadowdataset.Manifest, tasksPath, judgmentsPath, key string) ([]publishedFile, error) {
	var tasks shadowdataset.JudgeTaskSet
	if err := readJSON(tasksPath, &tasks); err != nil {
		return nil, err
	}
	if tasks.ManifestDigest != manifest.Digest {
		return nil, fmt.Errorf("tasks came from manifest %q, not %q", tasks.ManifestDigest, manifest.Digest)
	}
	if err := checkTasks(manifest, tasks, key); err != nil {
		return nil, err
	}
	file, err := os.Open(judgmentsPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	set, err := shadowdataset.DecodeJudgments(file)
	if err != nil {
		return nil, err
	}
	if err = shadowdataset.ValidateJudgments(tasks, set); err != nil {
		return nil, fmt.Errorf("judgments: %w", err)
	}

	setBody, err := encode(set)
	if err != nil {
		return nil, err
	}
	reportBody, err := encode(shadowdataset.ReportJudgments(tasks, set))
	if err != nil {
		return nil, err
	}
	base := "judgments/" + manifest.Digest + "/" + digestOf(setBody)
	return []publishedFile{
		{name: base + ".json", body: setBody},
		{name: base + ".report.json", body: reportBody},
	}, nil
}

// checkTasks refuses tasks that the manifest and the blinding key did not
// produce. A manifest digest only names a manifest, so the tasks are built again
// from the manifest, the key and the text each arm carries, and must come out
// the same: text that does not hash back to the manifest leaves its pair out of
// the rebuild, and an ID the key does not derive from the manifest has no
// counterpart in it.
func checkTasks(manifest shadowdataset.Manifest, tasks shadowdataset.JudgeTaskSet, key string) error {
	armText := map[string]string{}
	armInput := map[string]string{}
	for _, task := range tasks.Tasks {
		for _, side := range []shadowdataset.Candidate{task.First, task.Second} {
			armText[side.Arm] = side.Text
			armInput[side.Arm] = task.Input
		}
	}
	texts := make(map[string]shadowdataset.ExampleText, len(manifest.Examples))
	for _, example := range manifest.Examples {
		text := shadowdataset.ExampleText{Shadows: make([]string, len(example.Shadows))}
		for index := range example.Shadows {
			primary := shadowdataset.ArmID(key, example.ID, index, true)
			if text.Primary == "" {
				text.Input, text.Primary = armInput[primary], armText[primary]
			}
			text.Shadows[index] = armText[shadowdataset.ArmID(key, example.ID, index, false)]
		}
		texts[example.ID] = text
	}
	rebuilt, err := shadowdataset.BuildJudgeTasks(manifest, texts, key)
	if err != nil {
		return err
	}
	imported := slices.Clone(tasks.Tasks)
	sort.Slice(imported, func(i, j int) bool { return imported[i].ID < imported[j].ID })
	if tasks.Version != rebuilt.Version || !slices.Equal(imported, rebuilt.Tasks) {
		return fmt.Errorf("tasks are not the ones manifest %q and the blinding key produce from their text", manifest.Digest)
	}
	return nil
}

func encode(value any) ([]byte, error) {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(body, '\n'), nil
}

func readJSON(path string, into any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err = json.Unmarshal(data, into); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func digestOf(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}
