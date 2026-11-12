package llmprotocol

// Router extension fields are the members the Router itself adds to a Chat
// reply as Looper evidence. Their values come only from the Router's own
// emitters, so an upstream reply that carries one of these names is refused
// instead of having the member dropped like other undeclared members: a
// same-format client would otherwise receive the upstream's value as if the
// Router had produced it.
const (
	RouterExtensionFusion = "fusion"
	RouterExtensionFlow   = "flow"
	RouterExtensionReMoM  = "reasoning_mom_responses"
)

// IsRouterExtensionField reports whether a top-level Chat member name is one
// of the Router's own extension fields.
func IsRouterExtensionField(name string) bool {
	switch name {
	case RouterExtensionFusion, RouterExtensionFlow, RouterExtensionReMoM:
		return true
	}
	return false
}
