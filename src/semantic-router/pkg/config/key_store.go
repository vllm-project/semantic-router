package config

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// A configuration save, from the Dashboard or the Router's management API,
// moves each literal provider backend api_key and each literal listener
// api_keys entry into a KeyStore and writes the reference in its place. A
// value is literal when it holds no environment reference the Router expands
// (EnvironmentReferenceNames): an uppercase ${NAME} or $NAME, with or without
// a default. Any other text, ${lowercase} included, is a literal key.

// KeyRef is where a saved configuration reads a key from: a file, which a
// backend names in auth.api_key_file and a listener in api_key_files, or an
// environment variable, which a backend names in auth.api_key_env and a
// listener as a "${NAME}" api_keys entry.
type KeyRef struct {
	File string
	Env  string
}

// KeyStore holds the keys a save moves out of a configuration. Ref returns
// the reference to the copy that holds key. With create it stores key first
// when no copy holds it, and reports that it did. Without create, or from a
// store that cannot take the key, it returns the zero KeyRef and the key stays
// in the document. A stored key keeps its reference, so saving or planning
// the same document twice writes the same bytes. Remove deletes a copy Ref
// created.
type KeyStore interface {
	Ref(key string, create bool) (ref KeyRef, created bool, err error)
	Remove(ref KeyRef) error
}

// KeySave is a document whose literal keys a save moved into a store.
type KeySave struct {
	Data []byte
	// Kept are the paths of the literal keys the store could not take, which
	// stay in Data as written.
	Kept    []string
	store   KeyStore
	created []KeyRef
}

// Discard removes the copies the save stored, for a save that did not write
// Data.
func (s KeySave) Discard() {
	for _, ref := range s.created {
		_ = s.store.Remove(ref)
	}
}

// Warnings describe the keys the save kept as written.
func (s KeySave) Warnings() []ConfigWarning {
	warnings := make([]ConfigWarning, 0, len(s.Kept))
	for _, path := range s.Kept {
		warnings = append(warnings, ConfigWarning{
			Code:  "literal_key_kept",
			Field: path,
			Message: "the key stays in the configuration as written: this deployment has no key store it can write, " +
				"such as a Kubernetes ConfigMap without the chart's key Secret; keep the key in a Secret and " +
				"reference it with auth.api_key_env, or as a ${NAME} listener key",
		})
	}
	return warnings
}

// StoreLiteralKeys moves each literal backend key of data, and with listeners
// each literal listener key, into store. Listener keys stay for a gateway that
// renders them itself, such as the Envoy listener of --gateway extproc. The
// save's Data is data unchanged when no key moves.
func StoreLiteralKeys(data []byte, store KeyStore, create, listeners bool) (KeySave, error) {
	save := KeySave{Data: data, store: store}
	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		return save, err
	}
	// A moved key leaves no anchor behind, so its aliases become copies.
	inlineScalarAliases(&document)
	mover := keyMover{store: store, create: create, save: &save}
	root := documentRoot(&document)
	for _, model := range sequenceAt(root, "providers", "models") {
		for index, backend := range sequenceAt(model, "backend_refs") {
			path := fmt.Sprintf("providers.models[%s].backend_refs[%s].api_key", scalarAt(model, "name"), itemLabel(backend, index))
			if err := mover.backend(path, backend); err != nil {
				save.Discard()
				return KeySave{Data: data}, err
			}
		}
	}
	for index, listener := range sequenceAt(root, "listeners") {
		if !listeners {
			break
		}
		if err := mover.listener("listeners["+itemLabel(listener, index)+"].api_keys", listener); err != nil {
			save.Discard()
			return KeySave{Data: data}, err
		}
	}
	if !mover.moved {
		return save, nil
	}
	stored, err := encodeDocument(&document)
	if err != nil {
		save.Discard()
		return KeySave{Data: data}, err
	}
	save.Data = stored
	return save, nil
}

type keyMover struct {
	store  KeyStore
	create bool
	save   *KeySave
	moved  bool
}

// ref stores the key a scalar holds literally, and returns the zero KeyRef
// for a reference, an empty value or any other node.
func (m *keyMover) ref(path string, node *yaml.Node) (KeyRef, error) {
	if node == nil || node.Kind != yaml.ScalarNode || len(EnvironmentReferenceNames(node.Value)) > 0 {
		return KeyRef{}, nil
	}
	// The Router reads the value with $$ as $, and a key file trimmed.
	key := strings.TrimSpace(envExpander{lookup: noEnv}.expandString(node.Value))
	if key == "" {
		return KeyRef{}, nil
	}
	ref, created, err := m.store.Ref(key, m.create)
	if created {
		m.save.created = append(m.save.created, ref)
	}
	if err == nil && ref == (KeyRef{}) && m.create {
		m.save.Kept = append(m.save.Kept, path)
	}
	m.moved = m.moved || ref != (KeyRef{})
	return ref, err
}

func (m *keyMover) backend(path string, backend *yaml.Node) error {
	ref, err := m.ref(path, mappingValue(backend, "api_key"))
	if err != nil || ref == (KeyRef{}) {
		return err
	}
	deleteKey(backend, "api_key")
	auth := ensureMapping(backend, "auth")
	if ref.File != "" {
		setNode(auth, "api_key_file", Scalar(ref.File))
	} else {
		setNode(auth, "api_key_env", Scalar(ref.Env))
	}
	return nil
}

// listener moves a listener's literal api_keys entries: a file joins
// api_key_files, an environment variable stays in api_keys as a reference.
func (m *keyMover) listener(path string, listener *yaml.Node) error {
	keys := mappingValue(listener, "api_keys")
	if keys == nil || keys.Kind != yaml.SequenceNode {
		return nil
	}
	var kept, files []*yaml.Node
	for index, key := range keys.Content {
		ref, err := m.ref(path+"["+strconv.Itoa(index)+"]", key)
		switch {
		case err != nil:
			return err
		case ref.File != "":
			files = append(files, Scalar(ref.File))
		case ref.Env != "":
			kept = append(kept, Scalar("${"+ref.Env+"}"))
		default:
			kept = append(kept, key)
		}
	}
	if slices.Equal(kept, keys.Content) {
		return nil
	}
	if keys.Content = kept; len(kept) == 0 {
		deleteKey(listener, "api_keys")
	}
	if len(files) == 0 {
		return nil
	}
	stored := mappingValue(listener, "api_key_files")
	if stored == nil || stored.Kind != yaml.SequenceNode {
		stored = &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
		setNode(listener, "api_key_files", stored)
	}
	for _, file := range files {
		if !slices.ContainsFunc(stored.Content, func(node *yaml.Node) bool { return node.Value == file.Value }) {
			stored.Content = append(stored.Content, file)
		}
	}
	return nil
}

// inlineScalarAliases replaces each alias of a scalar with a copy of it.
func inlineScalarAliases(node *yaml.Node) {
	for index, child := range node.Content {
		if child.Kind == yaml.AliasNode && child.Alias != nil && child.Alias.Kind == yaml.ScalarNode {
			copied := *child.Alias
			copied.Anchor = ""
			node.Content[index] = &copied
			continue
		}
		inlineScalarAliases(child)
	}
}

// ProviderKeyDir and RouterKeyDir are the directories, beside the
// configuration file, where the Dashboard and the Router's management API
// keep the keys a save moves out of a local configuration. Each writer has
// its own, so neither needs to write into a directory the other created.
const (
	ProviderKeyDir = "provider-keys"
	RouterKeyDir   = "router-keys"
)

// KeyFileGroupEnv shares the key files of a local save with a group:
// `vllm-sr serve` sets it to the group of the stack's state directory, so the
// Dashboard container reads the files the Router container writes as root.
const KeyFileGroupEnv = "VLLM_SR_KEY_FILE_GROUP"

// KeyDir stores each key in its own file in Dir: mode 0600 in a mode 0700
// directory, or with Group set, mode 0640 in a mode 0750 directory, both of
// that group, which reads them and nothing more.
type KeyDir struct {
	Dir   string
	Group int
}

// Ref finds the file in Dir that holds key, or stores key in a new one.
func (d KeyDir) Ref(key string, create bool) (KeyRef, bool, error) {
	dir, err := filepath.Abs(d.Dir)
	if err != nil {
		return KeyRef{}, false, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil && !os.IsNotExist(err) {
		return KeyRef{}, false, err
	}
	for _, entry := range entries {
		if !entry.Type().IsRegular() {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		if stored, readErr := readKeyFile("key file", path); readErr == nil && stored == key {
			return KeyRef{File: path}, false, d.share(path, 0o640)
		}
	}
	if !create {
		return KeyRef{}, false, nil
	}
	path, err := d.write(dir, key)
	return KeyRef{File: path}, err == nil, err
}

// Remove deletes a key file Ref created.
func (d KeyDir) Remove(ref KeyRef) error {
	return os.Remove(ref.File)
}

func (d KeyDir) write(dir, key string) (string, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	// A directory made before sharing was on gets the shared mode too.
	if err := d.share(dir, 0o750); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(dir, "key-*")
	if err != nil {
		return "", err
	}
	err = d.share(file.Name(), 0o640)
	if err == nil {
		_, err = file.WriteString(key)
	}
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}

// share gives path to the key group with mode, when there is one.
func (d KeyDir) share(path string, mode os.FileMode) error {
	if d.Group == 0 {
		return nil
	}
	if err := os.Chown(path, -1, d.Group); err != nil {
		return err
	}
	return os.Chmod(path, mode)
}
