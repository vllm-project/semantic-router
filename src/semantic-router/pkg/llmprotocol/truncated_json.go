package llmprotocol

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// TruncatedJSONObject reads function-call arguments that an output limit cut
// short: nothing yet, or a JSON object that stops before it closes, with no
// error in the text before the cut. It reports false for anything else,
// including a complete object.
//
// It also returns the members the model completed before the cut as one JSON
// object, with every open object and array closed. A key without its value,
// and a string, number or literal the cut ended inside, are left out.
func TruncatedJSONObject(arguments string, maximumDepth int) (string, bool) {
	text := strings.TrimLeft(arguments, jsonWhitespace)
	if text == "" {
		return "{}", true
	}
	if text[0] != '{' || !utf8.ValidString(text) {
		return "", false
	}
	scan := truncatedJSONScan{text: text, maximumDepth: maximumDepth}
	if !scan.run() {
		return "", false
	}
	members := scan.members()
	if ValidateJSONObject([]byte(members), maximumDepth) != nil {
		return "", false
	}
	return members, true
}

func truncatedArguments(arguments string, maximumDepth int) bool {
	_, truncated := TruncatedJSONObject(arguments, maximumDepth)
	return truncated
}

const jsonWhitespace = " \t\r\n"

type truncatedJSONState uint8

const (
	expectFirstKey truncatedJSONState = iota
	expectKey
	expectColon
	expectMemberValue
	expectFirstElement
	expectElement
	expectSeparator
)

type truncatedJSONFrame struct {
	array bool
	state truncatedJSONState
	keys  map[string]struct{}
}

type truncatedJSONScan struct {
	text         string
	maximumDepth int
	stack        []truncatedJSONFrame
	// safe ends the text that the open containers' closing brackets turn into
	// valid JSON. Every push and pop moves it, so the stack at the end of the
	// text is the stack at safe.
	safe int
}

// run scans the whole text. It reports whether the text is a JSON object cut
// before it closes.
func (scan *truncatedJSONScan) run() bool {
	position := 0
	for position < len(scan.text) {
		character := scan.text[position]
		var ok, cut bool
		switch {
		case strings.IndexByte(jsonWhitespace, character) >= 0:
			position, ok = position+1, true
		case character == '{' || character == '[':
			position, ok = position+1, scan.open(character == '[')
			scan.safe = position
		case character == '}' || character == ']':
			position, ok = position+1, scan.close(character == ']')
			if len(scan.stack) == 0 {
				// A complete object, or data after its end.
				return false
			}
			scan.safe = position
		case character == ',' || character == ':':
			position, ok = position+1, scan.punctuate(character)
		case character == '"':
			position, ok, cut = scan.readString(position)
		default:
			position, ok, cut = scan.readScalar(position)
		}
		if !ok {
			return false
		}
		if cut {
			return true
		}
	}
	return true
}

func (scan *truncatedJSONScan) top() *truncatedJSONFrame {
	if len(scan.stack) == 0 {
		return nil
	}
	return &scan.stack[len(scan.stack)-1]
}

// startValue reports whether a value may start here: the outer frame expects
// one, and its depth is within the limit ValidateJSONObject applies.
func (scan *truncatedJSONScan) startValue() bool {
	frame := scan.top()
	if frame == nil {
		return false
	}
	if frame.state != expectMemberValue && frame.state != expectFirstElement && frame.state != expectElement {
		return false
	}
	return scan.maximumDepth > 0 && len(scan.stack) <= scan.maximumDepth
}

func (scan *truncatedJSONScan) endValue() {
	scan.top().state = expectSeparator
}

func (scan *truncatedJSONScan) open(array bool) bool {
	if len(scan.stack) > 0 && !scan.startValue() {
		return false
	}
	frame := truncatedJSONFrame{array: array, state: expectFirstKey}
	if array {
		frame.state = expectFirstElement
	} else {
		frame.keys = make(map[string]struct{})
	}
	scan.stack = append(scan.stack, frame)
	return true
}

func (scan *truncatedJSONScan) close(array bool) bool {
	frame := scan.top()
	if frame == nil || frame.array != array {
		return false
	}
	opened := expectFirstKey
	if array {
		opened = expectFirstElement
	}
	if frame.state != opened && frame.state != expectSeparator {
		return false
	}
	scan.stack = scan.stack[:len(scan.stack)-1]
	if len(scan.stack) > 0 {
		scan.endValue()
	}
	return true
}

func (scan *truncatedJSONScan) punctuate(character byte) bool {
	frame := scan.top()
	switch {
	case character == ',' && frame.state == expectSeparator:
		frame.state = expectKey
		if frame.array {
			frame.state = expectElement
		}
		return true
	case character == ':' && frame.state == expectColon:
		frame.state = expectMemberValue
		return true
	default:
		return false
	}
}

// readString reads the string at position, a key or a value. It reports cut
// when the text ends inside it.
func (scan *truncatedJSONScan) readString(position int) (int, bool, bool) {
	frame := scan.top()
	key := frame.state == expectFirstKey || frame.state == expectKey
	if !key && !scan.startValue() {
		return position, false, false
	}
	end, complete, valid := scanJSONString(scan.text, position)
	if !valid {
		return position, false, false
	}
	if !complete {
		return end, true, true
	}
	if !key {
		scan.endValue()
		scan.safe = end
		return end, true, false
	}
	var name string
	if json.Unmarshal([]byte(scan.text[position:end]), &name) != nil {
		return position, false, false
	}
	if _, duplicate := frame.keys[name]; duplicate {
		return position, false, false
	}
	frame.keys[name] = struct{}{}
	frame.state = expectColon
	return end, true, false
}

// readScalar reads the number or literal at position. A number that reaches
// the end of the text is cut: more digits may have followed.
func (scan *truncatedJSONScan) readScalar(position int) (int, bool, bool) {
	if !scan.startValue() {
		return position, false, false
	}
	rest := scan.text[position:]
	for _, literal := range []string{"true", "false", "null"} {
		if strings.HasPrefix(rest, literal) {
			scan.endValue()
			scan.safe = position + len(literal)
			return scan.safe, true, false
		}
		if strings.HasPrefix(literal, rest) {
			return len(scan.text), true, true
		}
	}
	end, complete, valid := scanJSONNumber(scan.text, position)
	if !valid {
		return position, false, false
	}
	if end == len(scan.text) {
		return end, true, true
	}
	if !complete {
		return position, false, false
	}
	scan.endValue()
	scan.safe = end
	return end, true, false
}

// scanJSONNumber scans the number that starts at position. It returns where
// the number ends, whether what it read is a whole number, and whether it is
// the start of one.
func scanJSONNumber(text string, position int) (int, bool, bool) {
	const (
		start = iota
		sign
		zero
		integer
		point
		fraction
		exponent
		exponentSign
		exponentDigits
	)
	state := start
	index := position
	for ; index < len(text); index++ {
		character := text[index]
		digit := character >= '0' && character <= '9'
		next := -1
		switch {
		case state == start && character == '-':
			next = sign
		case (state == start || state == sign) && character == '0':
			next = zero
		case (state == start || state == sign) && digit, state == integer && digit:
			next = integer
		case (state == zero || state == integer) && character == '.':
			next = point
		case (state == point || state == fraction) && digit:
			next = fraction
		case (state == zero || state == integer || state == fraction) && (character == 'e' || character == 'E'):
			next = exponent
		case state == exponent && (character == '+' || character == '-'):
			next = exponentSign
		case (state == exponent || state == exponentSign || state == exponentDigits) && digit:
			next = exponentDigits
		}
		if next < 0 {
			break
		}
		state = next
	}
	if index == position {
		return position, false, false
	}
	complete := state == zero || state == integer || state == fraction || state == exponentDigits
	return index, complete, true
}

func (scan *truncatedJSONScan) members() string {
	var members strings.Builder
	members.Grow(scan.safe + len(scan.stack))
	members.WriteString(scan.text[:scan.safe])
	for index := len(scan.stack) - 1; index >= 0; index-- {
		if scan.stack[index].array {
			members.WriteByte(']')
		} else {
			members.WriteByte('}')
		}
	}
	return members.String()
}

// scanJSONString scans the string that opens at position. It returns where the
// string ends, whether it closed before the end of the text, and whether the
// text read so far is a valid JSON string.
func scanJSONString(text string, position int) (int, bool, bool) {
	for index := position + 1; index < len(text); index++ {
		switch character := text[index]; {
		case character == '"':
			return index + 1, true, true
		case character < 0x20:
			return index, false, false
		case character == '\\':
			end, valid := scanJSONEscape(text, index)
			if !valid {
				return index, false, false
			}
			if end > len(text) {
				return len(text), false, true
			}
			index = end - 1
		}
	}
	return len(text), false, true
}

// scanJSONEscape returns the end of the escape at index; past the end of the
// text when the text ends inside it.
func scanJSONEscape(text string, index int) (int, bool) {
	if index+1 >= len(text) {
		return len(text) + 1, true
	}
	switch text[index+1] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return index + 2, true
	case 'u':
		for offset := index + 2; offset < index+6; offset++ {
			if offset >= len(text) {
				return len(text) + 1, true
			}
			if strings.IndexByte("0123456789abcdefABCDEF", text[offset]) < 0 {
				return index, false
			}
		}
		return index + 6, true
	default:
		return index, false
	}
}
