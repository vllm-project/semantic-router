// Package protocolcodec owns wire codecs and composes them through the
// protocol-neutral llmprotocol IR.
package protocolcodec

import (
	"fmt"
	"reflect"
	"regexp"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var formatPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

type CodecMetadata interface {
	Format() llmprotocol.WireFormat
	Capabilities() llmprotocol.CapabilitySet
	// Stateless is an explicit construction contract: one registered Codec may
	// be called concurrently, while all request state belongs to returned stream
	// decoders and encoders.
	Stateless() bool
}

type MessageCodec interface {
	DecodeRequest([]byte, llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error)
	EncodeRequest(llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error)
	DecodeResponse([]byte, llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error)
	EncodeResponse(llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error)
}

type TransportErrorCodec interface {
	DecodeTransportError([]byte, llmprotocol.Policy) (llmprotocol.TransportError, llmprotocol.Diagnostics, error)
	EncodeTransportError(llmprotocol.TransportError) []byte
}

type BufferedCodec interface {
	MessageCodec
	TransportErrorCodec
}

type Codec interface {
	CodecMetadata
	BufferedCodec
}

type StreamCodec interface {
	NewDecoder(llmprotocol.StreamContext, llmprotocol.Policy) llmprotocol.StreamDecoder
	NewEncoder(llmprotocol.StreamContext, llmprotocol.Policy) llmprotocol.StreamEncoder
}

type codecPair struct {
	buffered Codec
	stream   StreamCodec
}

// Registry is immutable after construction. Resolve never exposes the backing
// map and codec implementations must be stateless.
type Registry struct {
	codecs map[llmprotocol.WireFormat]codecPair
}

func NewRegistry(codecs ...Codec) (*Registry, error) {
	entries := make(map[llmprotocol.WireFormat]codecPair, len(codecs))
	for index, codec := range codecs {
		if isNil(codec) {
			return nil, fmt.Errorf("codec %d is nil", index)
		}
		if !codec.Stateless() {
			return nil, fmt.Errorf("codec %q is not safe for immutable registry use", codec.Format())
		}
		format := codec.Format()
		if !formatPattern.MatchString(string(format)) {
			return nil, fmt.Errorf("codec %d has invalid format %q", index, format)
		}
		stream, ok := codec.(StreamCodec)
		if !ok || isNil(stream) {
			return nil, fmt.Errorf("codec %q does not implement streaming", format)
		}
		if _, duplicate := entries[format]; duplicate {
			return nil, fmt.Errorf("codec %q is registered more than once", format)
		}
		entries[format] = codecPair{buffered: codec, stream: stream}
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("codec registry is empty")
	}
	return &Registry{codecs: entries}, nil
}

func NewBuiltinRegistry() *Registry {
	registry, err := NewRegistry(OpenAIChatCodec{}, OpenAIResponsesCodec{}, AnthropicMessagesCodec{})
	if err != nil {
		panic(err)
	}
	return registry
}

func (registry *Registry) resolve(format llmprotocol.WireFormat) (codecPair, bool) {
	if registry == nil {
		return codecPair{}, false
	}
	pair, ok := registry.codecs[format]
	return pair, ok
}

type Capability struct {
	Format       llmprotocol.WireFormat
	Capabilities []string
}

// CapabilitiesFor returns the capability set advertised by the codec bound to
// a wire format, and whether that format is registered. Unknown formats return
// false so callers can distinguish "not registered" from "no capabilities".
func (registry *Registry) CapabilitiesFor(format llmprotocol.WireFormat) (llmprotocol.CapabilitySet, bool) {
	if registry == nil {
		return llmprotocol.CapabilitySet{}, false
	}
	pair, ok := registry.resolve(format)
	if !ok {
		return llmprotocol.CapabilitySet{}, false
	}
	return pair.buffered.Capabilities(), true
}

func (registry *Registry) Capabilities() []Capability {
	if registry == nil {
		return nil
	}
	result := make([]Capability, 0, len(registry.codecs))
	for format, pair := range registry.codecs {
		result = append(result, Capability{Format: format, Capabilities: pair.buffered.Capabilities().Names()})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Format < result[j].Format })
	return result
}

func (registry *Registry) Check(formats []llmprotocol.WireFormat) error {
	for _, format := range formats {
		if _, ok := registry.resolve(format); !ok {
			return fmt.Errorf("wire format %q is unavailable", format)
		}
	}
	return nil
}

func isNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
