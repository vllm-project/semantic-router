package configsnapshot

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"hash"
	"math"
	"reflect"
	"sort"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// maxFingerprintDepth bounds the walk through nested values. Configuration is
// a tree far shallower than this; only a reference cycle reaches it.
const maxFingerprintDepth = 64

// fingerprint returns a hex digest of the values' content: every field of every
// struct, exported or not and whatever its tags, map entries in key order, and
// nil kept apart from empty. Equal digests mean equal configuration.
func fingerprint(values ...any) string {
	return digest(sha256.New(), values)
}

// secretKey keys the fingerprints of credential values, so a digest that
// leaves the process cannot be used to guess the value it was computed from.
var secretKey = func() []byte {
	key := make([]byte, 32)
	_, _ = rand.Read(key) // crypto/rand.Read never fails since Go 1.24.
	return key
}()

// secretFingerprint is fingerprint keyed by this process's secret key. It is
// stable within a process only.
func secretFingerprint(values ...any) string {
	return digest(hmac.New(sha256.New, secretKey), values)
}

// ownedFields are the configuration fields that resources other than the
// settings own, named by their declaring struct. The settings resource is
// everything else, so a field added to the configuration belongs to it until
// another resource claims it.
var ownedFields = map[string]bool{
	"APIServer.Listeners":                      true, // listeners
	"BackendModels.VLLMEndpoints":              true, // endpoints
	"BackendModels.ProviderProfiles":           true,
	"BackendModels.ModelConfig":                true, // clusters
	"BackendModels.ProviderModelOrder":         true,
	"InlineModels.ModelDeployments":            true, // runtime models
	"RouterConfig.Recipes":                     true, // programs
	"RouterConfig.Entrypoints":                 true, // routes
	"IntelligentRouting.CandidateRequirements": true, // the default program's flat mirror
	"IntelligentRouting.DataPolicy":            true,
	"IntelligentRouting.ModelBindings":         true,
	"IntelligentRouting.Signals":               true,
	"IntelligentRouting.Projections":           true,
	"IntelligentRouting.Decisions":             true,
	"IntelligentRouting.Strategy":              true,
	"IntelligentRouting.Fallback":              true,
	"RouterConfig.DocumentHash":                true, // identity and derived state
	"RouterConfig.EffectiveModelRegistry":      true,
	"RouterConfig.RoutingScope":                true,
	"RouterConfig.SkipExternalAssetValidation": true,
	"RouterConfig.RoutingFragmentOnly":         true,
}

// settingsFingerprint fingerprints, keyed, every field of cfg that no other
// resource owns.
func settingsFingerprint(cfg *config.RouterConfig) string {
	h := hmac.New(sha256.New, secretKey)
	encodeUnowned(&encoder{w: h}, reflect.ValueOf(cfg).Elem())
	return hex.EncodeToString(h.Sum(nil))
}

// encodeUnowned encodes the fields of struct v that ownedFields does not
// name, looking into embedded structs, whose fields the configuration
// promotes.
func encodeUnowned(e *encoder, v reflect.Value) {
	t := v.Type()
	for i := range t.NumField() {
		field := t.Field(i)
		if ownedFields[t.Name()+"."+field.Name] {
			continue
		}
		if field.Anonymous && field.Type.Kind() == reflect.Struct {
			encodeUnowned(e, v.Field(i))
			continue
		}
		e.string(field.Name)
		e.value(v.Field(i), 1)
	}
}

func digest(h hash.Hash, values []any) string {
	e := &encoder{w: h}
	for _, value := range values {
		e.value(reflect.ValueOf(value), 0)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// encoder writes a self-delimiting encoding: each value starts with a tag byte
// and variable-length data carries its length, so distinct values never
// produce the same bytes.
type encoder struct {
	w   hash.Hash
	buf [binary.MaxVarintLen64]byte
}

func (e *encoder) tag(t byte) { e.w.Write([]byte{t}) }

func (e *encoder) uint(u uint64) {
	n := binary.PutUvarint(e.buf[:], u)
	e.w.Write(e.buf[:n])
}

func (e *encoder) length(l int) {
	n := binary.PutVarint(e.buf[:], int64(l))
	e.w.Write(e.buf[:n])
}

func (e *encoder) bytes(b []byte) {
	e.length(len(b))
	e.w.Write(b)
}

func (e *encoder) string(s string) {
	e.length(len(s))
	e.w.Write([]byte(s))
}

func (e *encoder) value(v reflect.Value, depth int) {
	if depth > maxFingerprintDepth {
		e.tag('!')
		return
	}
	if !v.IsValid() {
		e.tag('0')
		return
	}
	switch v.Kind() {
	case reflect.Bool:
		e.tag('b')
		if v.Bool() {
			e.uint(1)
		} else {
			e.uint(0)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		e.tag('i')
		e.string(strconv.FormatInt(v.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		e.tag('u')
		e.uint(v.Uint())
	case reflect.Float32, reflect.Float64:
		e.tag('f')
		e.uint(math.Float64bits(v.Float()))
	case reflect.Complex64, reflect.Complex128:
		e.tag('c')
		c := v.Complex()
		e.uint(math.Float64bits(real(c)))
		e.uint(math.Float64bits(imag(c)))
	case reflect.String:
		e.tag('s')
		e.string(v.String())
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			e.tag('n')
			return
		}
		e.tag('p')
		if v.Kind() == reflect.Interface {
			e.string(v.Elem().Type().String())
		}
		e.value(v.Elem(), depth+1)
	case reflect.Slice:
		if v.IsNil() {
			e.tag('n')
			return
		}
		e.sequence(v, depth)
	case reflect.Array:
		e.sequence(v, depth)
	case reflect.Map:
		e.mapping(v, depth)
	case reflect.Struct:
		e.tag('{')
		e.string(v.Type().String())
		for i := range v.NumField() {
			e.string(v.Type().Field(i).Name)
			e.value(v.Field(i), depth+1)
		}
	default:
		// Functions, channels and unsafe pointers carry no configuration;
		// only whether they are set is recorded.
		e.tag('x')
		e.string(v.Kind().String())
		e.uint(boolBit(!v.IsZero()))
	}
}

func (e *encoder) sequence(v reflect.Value, depth int) {
	e.tag('[')
	e.length(v.Len())
	for i := range v.Len() {
		e.value(v.Index(i), depth+1)
	}
}

// mapping encodes entries sorted by their encoded keys, the only order that
// does not depend on how the map was built.
func (e *encoder) mapping(v reflect.Value, depth int) {
	if v.IsNil() {
		e.tag('n')
		return
	}
	type entry struct{ key, value []byte }
	entries := make([]entry, 0, v.Len())
	iter := v.MapRange()
	for iter.Next() {
		entries = append(entries, entry{
			key:   encodeOne(iter.Key(), depth+1),
			value: encodeOne(iter.Value(), depth+1),
		})
	}
	sort.Slice(entries, func(i, j int) bool { return bytes.Compare(entries[i].key, entries[j].key) < 0 })
	e.tag('m')
	e.length(len(entries))
	for _, en := range entries {
		e.bytes(en.key)
		e.bytes(en.value)
	}
}

// encodeOne returns the encoding of one value on its own.
func encodeOne(v reflect.Value, depth int) []byte {
	var out bytes.Buffer
	sub := &encoder{w: &bufferHash{&out}}
	sub.value(v, depth)
	return out.Bytes()
}

// bufferHash adapts a buffer to hash.Hash so the encoder can write map entries
// to memory before they are sorted.
type bufferHash struct{ *bytes.Buffer }

func (b *bufferHash) Sum(in []byte) []byte { return append(in, b.Bytes()...) }
func (b *bufferHash) Size() int            { return b.Len() }
func (b *bufferHash) BlockSize() int       { return 1 }

func boolBit(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}
