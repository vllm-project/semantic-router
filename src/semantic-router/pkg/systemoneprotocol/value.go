package systemoneprotocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
)

// Kind is the JSON type of a Value.
type Kind uint8

const (
	KindNull Kind = iota
	KindBool
	KindNumber
	KindString
	KindArray
	KindObject
)

// maxDepth bounds the nesting Parse accepts, as encoding/json bounds it.
const maxDepth = 10000

// Value is a JSON value that keeps an object's members in order, as a Python
// dict that json.loads built keeps them. A number keeps its literal. The zero
// Value is null.
type Value struct {
	kind    Kind
	boolean bool
	text    string
	items   []Value
	members []Member
}

// Member is one member of an object.
type Member struct {
	Key   string
	Value Value
}

// Null is the JSON null.
func Null() Value { return Value{} }

// Bool is a JSON boolean.
func Bool(value bool) Value { return Value{kind: KindBool, boolean: value} }

// String is a JSON string.
func String(text string) Value { return Value{kind: KindString, text: text} }

// Int is a JSON number holding an integer.
func Int(value int64) Value {
	return Value{kind: KindNumber, text: strconv.FormatInt(value, 10)}
}

// Float is a JSON number holding a finite float, written as its shortest
// round-tripping literal.
func Float(value float64) (Value, error) {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return Value{}, fmt.Errorf("%v is not a JSON number", value)
	}
	return Value{kind: KindNumber, text: strconv.FormatFloat(value, 'g', -1, 64)}, nil
}

// Number is a JSON number with the given literal, which must be valid JSON.
func Number(literal string) (Value, error) {
	if !json.Valid([]byte(literal)) || !isNumberLiteral(literal) {
		return Value{}, fmt.Errorf("%q is not a JSON number", literal)
	}
	return Value{kind: KindNumber, text: literal}, nil
}

// Array is a JSON array of items.
func Array(items ...Value) Value {
	return Value{kind: KindArray, items: append([]Value{}, items...)}
}

// Object is a JSON object of members in order. A repeated key keeps its first
// position and takes its last value, as a Python dict does.
func Object(members ...Member) Value {
	object := Value{kind: KindObject, members: make([]Member, 0, len(members))}
	for _, member := range members {
		object = object.With(member.Key, member.Value)
	}
	return object
}

// Kind is the value's JSON type.
func (v Value) Kind() Kind { return v.kind }

// IsNull reports whether the value is null.
func (v Value) IsNull() bool { return v.kind == KindNull }

// Text is a string's text.
func (v Value) Text() (string, bool) { return v.text, v.kind == KindString }

// Literal is a number's literal.
func (v Value) Literal() (string, bool) { return v.text, v.kind == KindNumber }

// Float64 is a number as a float64.
func (v Value) Float64() (float64, bool) {
	if v.kind != KindNumber {
		return 0, false
	}
	value, err := strconv.ParseFloat(v.text, 64)
	return value, err == nil
}

// Boolean is a boolean's value.
func (v Value) Boolean() (bool, bool) { return v.boolean, v.kind == KindBool }

// Items are an array's elements; callers must not modify them.
func (v Value) Items() []Value {
	if v.kind != KindArray {
		return nil
	}
	return v.items
}

// Members are an object's members in order; callers must not modify them.
func (v Value) Members() []Member {
	if v.kind != KindObject {
		return nil
	}
	return v.members
}

// Get is an object's member value.
func (v Value) Get(key string) (Value, bool) {
	for _, member := range v.Members() {
		if member.Key == key {
			return member.Value, true
		}
	}
	return Value{}, false
}

// Has reports whether an object has the member.
func (v Value) Has(key string) bool {
	_, ok := v.Get(key)
	return ok
}

// Keys are an object's member keys in order.
func (v Value) Keys() []string {
	keys := make([]string, 0, len(v.Members()))
	for _, member := range v.Members() {
		keys = append(keys, member.Key)
	}
	return keys
}

// With is an object with the member set: in its place when the object has
// it, else last. A value that is not an object becomes an empty one first.
func (v Value) With(key string, value Value) Value {
	members := make([]Member, 0, len(v.Members())+1)
	replaced := false
	for _, member := range v.Members() {
		if member.Key == key {
			member.Value, replaced = value, true
		}
		members = append(members, member)
	}
	if !replaced {
		members = append(members, Member{Key: key, Value: value})
	}
	return Value{kind: KindObject, members: members}
}

// Without is an object without the given members.
func (v Value) Without(keys ...string) Value {
	if v.kind != KindObject {
		return v
	}
	members := make([]Member, 0, len(v.members))
	for _, member := range v.members {
		drop := false
		for _, key := range keys {
			drop = drop || member.Key == key
		}
		if !drop {
			members = append(members, member)
		}
	}
	return Value{kind: KindObject, members: members}
}

// Equal reports whether two values are the same JSON value: the same members
// in the same order, and numbers with equal literals.
func (v Value) Equal(other Value) bool {
	if v.kind != other.kind || v.boolean != other.boolean || v.text != other.text ||
		len(v.items) != len(other.items) || len(v.members) != len(other.members) {
		return false
	}
	for index := range v.items {
		if !v.items[index].Equal(other.items[index]) {
			return false
		}
	}
	for index := range v.members {
		if v.members[index].Key != other.members[index].Key || !v.members[index].Value.Equal(other.members[index].Value) {
			return false
		}
	}
	return true
}

// Parse reads one JSON value as Python's json.loads does: an object keeps its
// members in document order, and a repeated key keeps its first position and
// its last value. Invalid UTF-8 and lone surrogates read as U+FFFD.
func Parse(data []byte) (Value, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	value, err := parseValue(decoder, 0)
	if err != nil {
		return Value{}, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return Value{}, errors.New("invalid JSON: data after the value")
	}
	return value, nil
}

func parseValue(decoder *json.Decoder, depth int) (Value, error) {
	if depth > maxDepth {
		return Value{}, fmt.Errorf("invalid JSON: nested more than %d levels", maxDepth)
	}
	token, err := decoder.Token()
	if err != nil {
		return Value{}, fmt.Errorf("invalid JSON: %w", err)
	}
	switch token := token.(type) {
	case json.Delim:
		switch token {
		case '{':
			return parseObject(decoder, depth)
		case '[':
			return parseArray(decoder, depth)
		}
		return Value{}, fmt.Errorf("invalid JSON: unexpected %q", token)
	case string:
		return String(token), nil
	case json.Number:
		return Value{kind: KindNumber, text: token.String()}, nil
	case bool:
		return Bool(token), nil
	case nil:
		return Null(), nil
	}
	return Value{}, fmt.Errorf("invalid JSON: unexpected token %v", token)
}

func parseObject(decoder *json.Decoder, depth int) (Value, error) {
	object := Value{kind: KindObject}
	index := map[string]int{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return Value{}, fmt.Errorf("invalid JSON: %w", err)
		}
		key, ok := token.(string)
		if !ok {
			return Value{}, errors.New("invalid JSON: an object key must be a string")
		}
		value, err := parseValue(decoder, depth+1)
		if err != nil {
			return Value{}, err
		}
		if at, seen := index[key]; seen {
			object.members[at].Value = value
			continue
		}
		index[key] = len(object.members)
		object.members = append(object.members, Member{Key: key, Value: value})
	}
	if _, err := decoder.Token(); err != nil {
		return Value{}, fmt.Errorf("invalid JSON: %w", err)
	}
	return object, nil
}

func parseArray(decoder *json.Decoder, depth int) (Value, error) {
	array := Value{kind: KindArray, items: []Value{}}
	for decoder.More() {
		item, err := parseValue(decoder, depth+1)
		if err != nil {
			return Value{}, err
		}
		array.items = append(array.items, item)
	}
	if _, err := decoder.Token(); err != nil {
		return Value{}, fmt.Errorf("invalid JSON: %w", err)
	}
	return array, nil
}

// FromGo converts a value encoding/json can marshal into a Value, keeping the
// field order encoding/json writes (struct fields in order, map keys sorted).
func FromGo(value any) (Value, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return Value{}, err
	}
	return Parse(data)
}

// MarshalJSON writes the value compactly, its strings escaped as Canonical
// escapes them.
func (v Value) MarshalJSON() ([]byte, error) {
	return appendValue(nil, v, false), nil
}

// UnmarshalJSON reads the value as Parse does.
func (v *Value) UnmarshalJSON(data []byte) error {
	parsed, err := Parse(data)
	if err != nil {
		return err
	}
	*v = parsed
	return nil
}

func isNumberLiteral(literal string) bool {
	if literal == "" {
		return false
	}
	switch literal[0] {
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return true
	}
	return false
}
