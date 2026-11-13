// Package tokenizerspec reads the tokenizer a model's prompts are counted
// with. It stays free of the tokenizers themselves, so configuration can
// check a spec without carrying their vocabularies.
package tokenizerspec

import (
	"fmt"
	"regexp"
	"strings"
)

// Built-in encodings.
const (
	O200kBase  = "o200k_base"
	Cl100kBase = "cl100k_base"
)

// Builtins are the encodings the Router carries.
var Builtins = []string{O200kBase, Cl100kBase}

// IsBuiltin reports whether name is a built-in encoding.
func IsBuiltin(name string) bool {
	for _, builtin := range Builtins {
		if name == builtin {
			return true
		}
	}
	return false
}

// Kind is where a tokenizer comes from.
type Kind int

const (
	// Builtin names a built-in encoding, such as o200k_base.
	Builtin Kind = iota
	// File is a path to a tokenizer.json, or to the directory that holds
	// one.
	File
	// Hub is a Hugging Face repo id, [org/]name[@revision].
	Hub
)

// Spec says which tokenizer counts a model's prompts.
type Spec struct {
	Kind     Kind
	Value    string
	Revision string
}

var (
	hubRepo = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$`)
	// The Router keeps a revision's tokenizer in a directory named after it,
	// so each part of one starts with a letter or digit: none is empty, "."
	// or "..".
	hubRevision = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*(/[A-Za-z0-9][A-Za-z0-9._+-]*)*$`)
)

// validRevision reports a branch, tag or commit name, or none.
func validRevision(revision string) bool {
	return revision == "" || hubRevision.MatchString(revision)
}

// Parse reads a tokenizer spec: a built-in encoding's name, a path to a
// tokenizer.json or to the directory that holds one (absolute, starting with
// ./ or ../, or ending in .json), or a Hugging Face repo id with an optional
// @revision.
func Parse(raw string) (Spec, error) {
	value := strings.TrimSpace(raw)
	switch {
	case value == "":
		return Spec{}, fmt.Errorf("a tokenizer cannot be empty")
	case IsBuiltin(value):
		return Spec{Kind: Builtin, Value: value}, nil
	case strings.HasPrefix(value, "/") || strings.HasPrefix(value, "./") || strings.HasPrefix(value, "../") ||
		strings.HasSuffix(value, ".json"):
		return Spec{Kind: File, Value: value}, nil
	}
	repo, revision, _ := strings.Cut(value, "@")
	if !hubRepo.MatchString(repo) || !validRevision(revision) {
		return Spec{}, fmt.Errorf("tokenizer %q is neither a built-in encoding (%s), a tokenizer.json path, nor a Hugging Face repo id",
			raw, strings.Join(Builtins, ", "))
	}
	if revision == "" {
		revision = "main"
	}
	return Spec{Kind: Hub, Value: repo, Revision: revision}, nil
}
