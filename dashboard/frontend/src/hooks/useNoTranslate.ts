/**
 * Utility for preventing browser translation on dynamic elements
 * 
 * Browser translation extensions (Google Translate, Edge Translate) can cause
 * React crashes by directly manipulating the DOM during streaming/animation.
 * 
 * This utility provides a consistent way to disable translation on dynamic elements.
 */

/**
 * Get translate attribute value based on streaming state
 * @param isStreaming - Whether content is currently being streamed/animated
 * @returns 'no' during streaming, undefined when complete (allows translation)
 */
export function getTranslateAttr(isStreaming: boolean): 'no' | undefined {
  return isStreaming ? 'no' : undefined
}

/**
 * Get className with notranslate when streaming
 * @param baseClass - Base CSS class name
 * @param isStreaming - Whether content is currently being streamed/animated
 * @returns Combined className string
 */
export function getNoTranslateClass(baseClass: string, isStreaming: boolean): string {
  return isStreaming ? `${baseClass} notranslate` : baseClass
}

/**
 * Props to spread on dynamic elements to prevent translation during streaming
 */
export function getNoTranslateProps(isStreaming: boolean) {
  return {
    translate: getTranslateAttr(isStreaming),
    className: isStreaming ? 'notranslate' : undefined,
  }
}
