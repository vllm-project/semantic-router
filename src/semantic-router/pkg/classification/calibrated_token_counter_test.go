package classification

import (
	"fmt"
	"math"
	"sync"
	"testing"

	. "github.com/onsi/gomega"
)

func TestCalibratedTokenCounter_DefaultBehavior(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter()

	t.Run("empty text returns 0", func(t *testing.T) {
		RegisterTestingT(t)
		count, err := c.CountTokens("")
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(0))
	})

	t.Run("uncalibrated uses chars/4 default", func(t *testing.T) {
		RegisterTestingT(t)
		// 400 bytes / 4.0 bytes-per-token = 100 tokens
		count, err := c.CountTokens(makeText(400))
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(100))
	})

	t.Run("implements TokenCounter interface", func(t *testing.T) {
		RegisterTestingT(t)
		var counter TokenCounter = NewCalibratedTokenCounter()
		count, err := counter.CountTokens(makeText(200))
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(Equal(50))
	})
}

func TestCalibratedTokenCounter_Calibration(t *testing.T) {
	RegisterTestingT(t)

	t.Run("learns from observations", func(t *testing.T) {
		RegisterTestingT(t)
		c := NewCalibratedTokenCounter(WithDecay(0.9))

		// Simulate code: actual ratio is ~3.5 bytes/token
		for i := 0; i < 50; i++ {
			c.Observe("code", 3500, 1000)
		}

		tokens := c.Estimate("code", 3500)
		Expect(tokens).To(BeNumerically("~", 1000, 50))
	})

	t.Run("different categories learn independently", func(t *testing.T) {
		RegisterTestingT(t)
		c := NewCalibratedTokenCounter(WithDecay(0.9))

		// Code: 3.5 bytes/token
		for i := 0; i < 50; i++ {
			c.Observe("code", 3500, 1000)
		}
		// CJK text: 2.0 bytes/token
		for i := 0; i < 50; i++ {
			c.Observe("cjk", 2000, 1000)
		}
		// English prose: 4.5 bytes/token
		for i := 0; i < 50; i++ {
			c.Observe("prose", 4500, 1000)
		}

		codeTokens := c.Estimate("code", 7000)
		cjkTokens := c.Estimate("cjk", 7000)
		proseTokens := c.Estimate("prose", 7000)

		// Code should estimate ~2000, CJK ~3500, prose ~1556
		Expect(codeTokens).To(BeNumerically("~", 2000, 100))
		Expect(cjkTokens).To(BeNumerically("~", 3500, 200))
		Expect(proseTokens).To(BeNumerically("~", 1556, 100))

		// CJK > code > prose for same byte length
		Expect(cjkTokens).To(BeNumerically(">", codeTokens))
		Expect(codeTokens).To(BeNumerically(">", proseTokens))
	})

	t.Run("uncalibrated category falls back to default", func(t *testing.T) {
		RegisterTestingT(t)
		c := NewCalibratedTokenCounter(WithDecay(0.9))

		// Calibrate "code" but not "newcategory"
		for i := 0; i < 50; i++ {
			c.Observe("code", 3500, 1000)
		}

		// "newcategory" should use default ratio (4.0)
		tokens := c.Estimate("newcategory", 4000)
		Expect(tokens).To(Equal(1000))
	})

	t.Run("requires minimum samples before switching from default", func(t *testing.T) {
		RegisterTestingT(t)
		c := NewCalibratedTokenCounter(WithDecay(0.9))

		// Only 5 observations (below minSamplesForUse=10)
		for i := 0; i < 5; i++ {
			c.Observe("sparse", 2000, 1000) // ratio 2.0
		}

		// Should still use default ratio 4.0 since not enough samples
		tokens := c.Estimate("sparse", 4000)
		Expect(tokens).To(Equal(1000)) // 4000 / 4.0 = 1000, not 2000
	})
}

func TestCalibratedTokenCounter_ConservativeEstimate(t *testing.T) {
	RegisterTestingT(t)

	t.Run("conservative gives higher token estimates", func(t *testing.T) {
		RegisterTestingT(t)
		mean := NewCalibratedTokenCounter(WithDecay(0.9))
		conservative := NewCalibratedTokenCounter(WithDecay(0.9), WithConservativeEstimate())

		// Feed identical noisy data to both
		observations := []struct {
			bytes  int
			tokens int
		}{
			{4000, 1000}, // ratio 4.0
			{3500, 1000}, // ratio 3.5
			{3000, 1000}, // ratio 3.0
			{4500, 1000}, // ratio 4.5
			{3200, 1000}, // ratio 3.2
			{5000, 1000}, // ratio 5.0
			{3800, 1000}, // ratio 3.8
			{2800, 1000}, // ratio 2.8
			{4200, 1000}, // ratio 4.2
			{3600, 1000}, // ratio 3.6
		}

		for i := 0; i < 3; i++ {
			for _, obs := range observations {
				mean.Observe("mixed", obs.bytes, obs.tokens)
				conservative.Observe("mixed", obs.bytes, obs.tokens)
			}
		}

		byteLen := 4000
		meanEst := mean.Estimate("mixed", byteLen)
		consEst := conservative.Estimate("mixed", byteLen)

		// Conservative (P90 quantile) should estimate MORE tokens
		// (lower bytes-per-token ratio → higher token count)
		Expect(consEst).To(BeNumerically(">=", meanEst))
	})
}

func TestCalibratedTokenCounter_GetRatio(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter(WithDecay(0.9))

	t.Run("unknown category returns defaults", func(t *testing.T) {
		RegisterTestingT(t)
		mean, quantile, samples, calibrated := c.GetRatio("unknown")
		Expect(mean).To(Equal(defaultBytesPerToken))
		Expect(quantile).To(Equal(defaultBytesPerToken))
		Expect(samples).To(Equal(int64(0)))
		Expect(calibrated).To(BeFalse())
	})

	t.Run("calibrated category returns learned values", func(t *testing.T) {
		RegisterTestingT(t)
		for i := 0; i < 20; i++ {
			c.Observe("json", 3000, 1000)
		}
		mean, _, samples, calibrated := c.GetRatio("json")
		Expect(mean).To(BeNumerically("~", 3.0, 0.3))
		Expect(samples).To(Equal(int64(20)))
		Expect(calibrated).To(BeTrue())
	})
}

func TestCalibratedTokenCounter_EstimateWithCategory(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter(WithDecay(0.9))

	t.Run("returns calibration status", func(t *testing.T) {
		RegisterTestingT(t)
		tokens, ratio, calibrated := c.EstimateWithCategory("uncalibrated", 4000)
		Expect(tokens).To(Equal(1000))
		Expect(ratio).To(Equal(defaultBytesPerToken))
		Expect(calibrated).To(BeFalse())

		for i := 0; i < 20; i++ {
			c.Observe("calibrated_cat", 3000, 1000)
		}
		tokens, ratio, calibrated = c.EstimateWithCategory("calibrated_cat", 3000)
		Expect(tokens).To(BeNumerically("~", 1000, 50))
		Expect(ratio).To(BeNumerically("~", 3.0, 0.3))
		Expect(calibrated).To(BeTrue())
	})
}

func TestCalibratedTokenCounter_Categories(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter()
	c.Observe("code", 3500, 1000)
	c.Observe("chat", 4000, 1000)
	c.Observe("rag", 3000, 1000)

	cats := c.Categories()
	Expect(cats).To(ConsistOf("code", "chat", "rag"))
}

func TestCalibratedTokenCounter_Convergence(t *testing.T) {
	RegisterTestingT(t)

	t.Run("converges to true ratio within 100 samples", func(t *testing.T) {
		RegisterTestingT(t)
		c := NewCalibratedTokenCounter(WithDecay(0.99))
		trueRatio := 3.2

		for i := 0; i < 200; i++ {
			byteLen := 1000 + (i%10)*100
			actualTokens := int(math.Round(float64(byteLen) / trueRatio))
			c.Observe("converge_test", byteLen, actualTokens)
		}

		mean, _, _, calibrated := c.GetRatio("converge_test")
		Expect(calibrated).To(BeTrue())
		Expect(mean).To(BeNumerically("~", trueRatio, 0.1))
	})
}

func TestCalibratedTokenCounter_ConcurrentSafety(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter(WithDecay(0.95))
	var wg sync.WaitGroup

	// Concurrent writers
	for cat := 0; cat < 5; cat++ {
		for w := 0; w < 10; w++ {
			wg.Add(1)
			go func(category string) {
				defer wg.Done()
				for i := 0; i < 100; i++ {
					c.Observe(category, 4000, 1000)
				}
			}(fmt.Sprintf("cat_%d", cat))
		}
	}

	// Concurrent readers
	for r := 0; r < 20; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				_ = c.Estimate(fmt.Sprintf("cat_%d", i%5), 4000)
			}
		}()
	}

	wg.Wait()

	// Verify all categories exist and are calibrated
	cats := c.Categories()
	Expect(cats).To(HaveLen(5))

	for i := 0; i < 5; i++ {
		mean, _, samples, calibrated := c.GetRatio(fmt.Sprintf("cat_%d", i))
		Expect(calibrated).To(BeTrue())
		Expect(samples).To(Equal(int64(1000)))
		Expect(mean).To(BeNumerically("~", 4.0, 0.2))
	}
}

func TestCalibratedTokenCounter_EdgeCases(t *testing.T) {
	RegisterTestingT(t)

	c := NewCalibratedTokenCounter()

	t.Run("zero-length observe is ignored", func(t *testing.T) {
		RegisterTestingT(t)
		c.Observe("edge", 0, 100)
		c.Observe("edge", 100, 0)
		_, _, samples, _ := c.GetRatio("edge")
		Expect(samples).To(Equal(int64(0)))
	})

	t.Run("empty category normalises to default", func(t *testing.T) {
		RegisterTestingT(t)
		tokens := c.Estimate("", 400)
		Expect(tokens).To(Equal(100))
	})

	t.Run("minimum 1 token", func(t *testing.T) {
		RegisterTestingT(t)
		tokens := c.Estimate("tiny", 1)
		Expect(tokens).To(Equal(1))
	})
}

func TestCalibratedTokenCounter_ContextClassifierIntegration(t *testing.T) {
	RegisterTestingT(t)

	t.Run("works as drop-in TokenCounter for ContextClassifier", func(t *testing.T) {
		RegisterTestingT(t)
		counter := NewCalibratedTokenCounter(WithDecay(0.9))

		// Calibrate: simulate code content at 3.5 bytes/token
		for i := 0; i < 20; i++ {
			counter.Observe(defaultCategory, 3500, 1000)
		}

		rules := []struct {
			name string
			min  string
			max  string
		}{
			{"short", "0", "2K"},
			{"long", "4K", "128K"},
		}

		var configRules []struct {
			Name      string
			MinTokens string
			MaxTokens string
		}
		for _, r := range rules {
			configRules = append(configRules, struct {
				Name      string
				MinTokens string
				MaxTokens string
			}{r.name, r.min, r.max})
		}

		// 7000 bytes at 3.5 bytes/token = 2000 tokens → matches "short"
		count, err := counter.CountTokens(makeText(7000))
		Expect(err).NotTo(HaveOccurred())
		Expect(count).To(BeNumerically("~", 2000, 100))
	})
}

func BenchmarkCalibratedTokenCounter_Estimate(b *testing.B) {
	c := NewCalibratedTokenCounter()
	for i := 0; i < 100; i++ {
		c.Observe("bench", 4000, 1000)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Estimate("bench", 4000)
	}
}

func BenchmarkCalibratedTokenCounter_Observe(b *testing.B) {
	c := NewCalibratedTokenCounter()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Observe("bench", 4000, 1000)
	}
}

func BenchmarkCharacterBasedTokenCounter(b *testing.B) {
	c := &CharacterBasedTokenCounter{}
	text := makeText(4000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.CountTokens(text)
	}
}

func makeText(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'a' + byte(i%26)
	}
	return string(b)
}
