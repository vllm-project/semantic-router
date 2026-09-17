package native

import (
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Qualify the supported openvino_tokenizers tokenizer IR before publishing raw
// input counts. Its TruncationStep clamps ragged ends with Minimum, rather than
// Slice. A fixed RaggedToDense length also hides tokens. Unknown operators are
// rejected until their input-length behavior is qualified, rather than assumed.
func validateOpenVINOTokenizerIR(graph string) error {
	path := filepath.Join(filepath.Dir(graph), "openvino_tokenizer.xml")
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var model struct {
		XMLName xml.Name `xml:"net"`
		Layers  []struct {
			Type string `xml:"type,attr"`
			Data struct {
				Attrs []xml.Attr `xml:",any,attr"`
			} `xml:"data"`
		} `xml:"layers>layer"`
	}
	if err = xml.NewDecoder(file).Decode(&model); err != nil {
		return fmt.Errorf("OpenVINO tokenizer IR: %w", err)
	}
	if len(model.Layers) == 0 {
		return fmt.Errorf("%w: empty OpenVINO tokenizer IR", binding.ErrCapability)
	}
	for _, layer := range model.Layers {
		switch layer.Type {
		case "Parameter", "Result", "Const", "Convert", "ShapeOf", "Gather", "Range", "ReduceMax", "Add", "Subtract",
			"StringTensorUnpack", "StringTensorPack", "RegexSplit", "SpecialTokensSplit", "RegexNormalization", "CaseFold", "CharsMapNormalization",
			"BPETokenizer", "WordpieceTokenizer", "UnigramTokenizer", "TrieTokenizer", "VocabEncoder", "BytesToChars", "CharsToBytes", "ByteFallback", "UTF8Validate", "CombineSegments", "FuzeRagged":
		case "RaggedToDense":
			dynamic := false
			for _, attribute := range layer.Data.Attrs {
				if attribute.Name.Local == "m_pad_max_length" || attribute.Name.Local == "pad_max_length" {
					dynamic = strings.EqualFold(attribute.Value, "false")
				}
			}
			if !dynamic {
				return fmt.Errorf("%w: tokenizer IR uses fixed or unknown RaggedToDense length", binding.ErrCapability)
			}
		default:
			return fmt.Errorf("%w: tokenizer IR operator %q has unqualified input-length behavior; export without truncation", binding.ErrCapability, layer.Type)
		}
	}
	return nil
}
