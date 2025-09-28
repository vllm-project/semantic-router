import React, { useState, useEffect } from 'react'
import styles from './TypewriterCode.module.css'

const TypewriterCode = () => {
  const codeText = `curl -X POST http://vllm-semantic-router/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{
           "model": "auto",
           "messages": [
             {
               "role": "user",
               "content": "solve the Riemann Hypothesis using advanced Number Theory"
             }
           ]
         }'

# selected: gpt-oss-120b, complexity: high, reasoning_effort: high, domain: math`

  const [displayedText, setDisplayedText] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    if (currentIndex < codeText.length) {
      const timer = setTimeout(() => {
        setDisplayedText(prev => prev + codeText[currentIndex])
        setCurrentIndex(prev => prev + 1)
      }, 50) // 打字速度，可以调整

      return () => clearTimeout(timer)
    }
    else {
      setIsComplete(true)
    }
  }, [currentIndex, codeText])

  // 渲染带颜色的文本
  const renderColoredText = (text) => {
    // 定义特殊单词的样式
    const specialWords = {
      'vllm-semantic-router': styles.vllmSemanticRouterColor,
      'auto': styles.autoColor,
      'Number Theory': styles.claudeColor,
      'Complexity': styles.modernBertColor,
      'Selected': styles.neuralConfidenceColor,
      'extreme': styles.reasoningColor,
      'reasoning_effort': styles.aiSelectionColor,
      'true': styles.highColor,
      'mathematics': styles.scienceColor,
      'domain': styles.confidenceValueColor,
      'Riemann Hypothesis': styles.modernBertColor,
    }

    let result = []
    let currentIndex = 0

    // 遍历文本，查找特殊单词
    while (currentIndex < text.length) {
      let foundSpecialWord = false

      // 检查是否匹配特殊单词
      for (const [word, className] of Object.entries(specialWords)) {
        const wordStart = currentIndex
        const wordEnd = wordStart + word.length

        if (wordEnd <= text.length
          && text.substring(wordStart, wordEnd).toLowerCase() === word.toLowerCase()) {
          // 找到特殊单词，应用特殊样式
          const wordText = text.substring(wordStart, wordEnd)
          result.push(
            <span key={currentIndex} className={className}>
              {wordText}
            </span>,
          )
          currentIndex = wordEnd
          foundSpecialWord = true
          break
        }
      }

      if (!foundSpecialWord) {
        // 普通字符，使用默认白色
        result.push(
          <span key={currentIndex} className={styles.defaultColor}>
            {text[currentIndex]}
          </span>,
        )
        currentIndex++
      }
    }

    return result
  }

  return (
    <div className={styles.typewriterContainer}>
      <div className={styles.codeBlock}>
        <div className={styles.codeHeader}>
          <div className={styles.windowControls}>
            <span className={styles.controlButton}></span>
            <span className={styles.controlButton}></span>
            <span className={styles.controlButton}></span>
          </div>
          <div className={styles.title}>Terminal</div>
        </div>
        <div className={styles.codeContent}>
          <pre className={styles.codeText}>
            {renderColoredText(displayedText)}
            {!isComplete && <span className={styles.cursor}>|</span>}
          </pre>
        </div>
      </div>

    </div>
  )
}

export default TypewriterCode
