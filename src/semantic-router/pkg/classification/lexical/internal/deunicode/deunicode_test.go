package deunicode

import "testing"

// Cases from the deunicode crate's own documentation and tests.
func TestTransliterate(t *testing.T) {
	for input, want := range map[string]string{
		"":                 "",
		"plain ascii":      "plain ascii",
		"Æneid":            "AEneid",
		"étude":            "etude",
		"北亰":               "Bei Jing",
		"ᔕᓇᓇ":              "shanana",
		"げんまい茶":            "genmaiCha",
		"🦄☣":               "unicorn biohazard",
		"…":                "...",
		"🄏中国":              "NonCommercialZhong Guo",
		"中国x🅶":             "Zhong Guo xG",
		"☃中 国":             "snowman Zhong Guo",
		"gemüse, Gießen":   "gemuse, Giessen",
		"h\u0335\u0321llo": "hllo",
		"\U0010FFFF":       Tofu,
		"a\x7fb":           "ab",
	} {
		if got := Transliterate(input); got != want {
			t.Errorf("Transliterate(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestTransliterateASCIIIsZeroCopy(t *testing.T) {
	input := "no tables needed"
	if got := Transliterate(input); got != input {
		t.Fatalf("Transliterate(%q) = %q", input, got)
	}
}
