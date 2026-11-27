package tasks

// TextWindowsRequest requests complete token coverage from a single owned
// tokenizer. Size includes special tokens; Overlap counts content tokens.
type TextWindowsRequest struct {
	Text    string
	Size    int
	Overlap int
}
