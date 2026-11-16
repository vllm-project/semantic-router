package testcases

import (
	"net/http"
	"strconv"
	"strings"
)

// The Router's response contract: x-vsr-response-path names how a response
// was produced, x-vsr-warnings lists what to know about it, and x-vsr-trace
// carries the routing details, an RFC 8941 dictionary, for a caller the
// Router allows the debug headers.
// routerSchemaVersion is the response-header contract revision
// x-vsr-schema-version stamps.
const routerSchemaVersion = "3"

const (
	responsePathHeader = "x-vsr-response-path"
	warningsHeader     = "x-vsr-warnings"
	traceHeader        = "x-vsr-trace"
)

// traceMember is one member of x-vsr-trace: its value, with strings and
// tokens unquoted and numbers as written; the items of an inner list; and
// its parameters.
type traceMember struct {
	Value  string
	Items  []string
	Params map[string]string
}

// routerTrace parses the x-vsr-trace header of a response; an absent or
// malformed header parses as no members.
func routerTrace(h http.Header) map[string]traceMember {
	p := &structuredFieldParser{input: h.Get(traceHeader)}
	members := map[string]traceMember{}
	for p.skipSpaces(); !p.done(); {
		key := p.key()
		if key == "" {
			return map[string]traceMember{}
		}
		member := traceMember{Value: "?1"}
		if p.consume('=') {
			if p.peek() == '(' {
				member.Items = p.innerList()
			} else {
				member.Value = p.bareItem()
			}
		}
		member.Params = p.params()
		members[key] = member
		p.skipSpaces()
		if !p.consume(',') {
			break
		}
		p.skipSpaces()
	}
	return members
}

// traceValue returns the value of one trace member, "" when the trace lacks it.
func traceValue(h http.Header, key string) string {
	return routerTrace(h)[key].Value
}

// traceParam returns one parameter of a trace member.
func traceParam(h http.Header, key, param string) string {
	return routerTrace(h)[key].Params[param]
}

// matchedSignals returns, comma-separated, the rules of one signal type the
// trace's signals member lists as type:name.
func matchedSignals(h http.Header, signalType string) string {
	var names []string
	for _, signal := range routerTrace(h)["signals"].Items {
		if name, ok := strings.CutPrefix(signal, signalType+":"); ok {
			names = append(names, name)
		}
	}
	return strings.Join(names, ",")
}

// warningMember is one member of x-vsr-warnings, an RFC 8941 list: a code
// with its parameters.
type warningMember struct {
	Code   string
	Params map[string]string
}

func parseWarnings(value string) []warningMember {
	p := &structuredFieldParser{input: value}
	var members []warningMember
	for p.skipSpaces(); !p.done(); {
		members = append(members, warningMember{Code: p.bareItem(), Params: p.params()})
		p.skipSpaces()
		if !p.consume(',') {
			break
		}
		p.skipSpaces()
	}
	return members
}

// responseWarnings returns the codes x-vsr-warnings lists, in order.
func responseWarnings(h http.Header) []string {
	var codes []string
	for _, member := range parseWarnings(h.Get(warningsHeader)) {
		codes = append(codes, member.Code)
	}
	return codes
}

// hasProtocolFieldDiagnostic reports whether an x-vsr-warnings value lists a
// protocol translation loss of field with action, such as dropped.
func hasProtocolFieldDiagnostic(value, action, field string) bool {
	for _, member := range parseWarnings(value) {
		if member.Code == "protocol" && member.Params["action"] == action && member.Params["field"] == field {
			return true
		}
	}
	return false
}

type structuredFieldParser struct {
	input string
	at    int
}

func (p *structuredFieldParser) done() bool { return p.at >= len(p.input) }

func (p *structuredFieldParser) peek() byte {
	if p.done() {
		return 0
	}
	return p.input[p.at]
}

func (p *structuredFieldParser) consume(b byte) bool {
	if p.peek() != b || p.done() {
		return false
	}
	p.at++
	return true
}

func (p *structuredFieldParser) skipSpaces() {
	for !p.done() && (p.input[p.at] == ' ' || p.input[p.at] == '\t') {
		p.at++
	}
}

func (p *structuredFieldParser) key() string {
	start := p.at
	for !p.done() {
		b := p.input[p.at]
		if b >= 'a' && b <= 'z' || b == '*' || p.at > start && (b >= '0' && b <= '9' || b == '_' || b == '-' || b == '.') {
			p.at++
			continue
		}
		break
	}
	return p.input[start:p.at]
}

func (p *structuredFieldParser) params() map[string]string {
	params := map[string]string{}
	for p.consume(';') {
		p.skipSpaces()
		key := p.key()
		value := "?1"
		if p.consume('=') {
			value = p.bareItem()
		}
		params[key] = value
	}
	return params
}

func (p *structuredFieldParser) innerList() []string {
	p.consume('(')
	var items []string
	for p.skipSpaces(); !p.done() && !p.consume(')'); p.skipSpaces() {
		items = append(items, p.bareItem())
		p.params()
	}
	return items
}

// bareItem reads a string, which it unquotes and percent-decodes, or a
// token, number or boolean, which it returns as written.
func (p *structuredFieldParser) bareItem() string {
	if !p.consume('"') {
		start := p.at
		for !p.done() && !strings.ContainsRune(" ,;()", rune(p.input[p.at])) {
			p.at++
		}
		return p.input[start:p.at]
	}
	var out strings.Builder
	for !p.done() {
		b := p.input[p.at]
		p.at++
		switch {
		case b == '\\' && !p.done():
			out.WriteByte(p.input[p.at])
			p.at++
		case b == '"':
			return out.String()
		case b == '%' && p.at+2 <= len(p.input):
			if decoded, err := strconv.ParseUint(p.input[p.at:p.at+2], 16, 8); err == nil {
				out.WriteByte(byte(decoded))
				p.at += 2
				continue
			}
			out.WriteByte(b)
		default:
			out.WriteByte(b)
		}
	}
	return out.String()
}
