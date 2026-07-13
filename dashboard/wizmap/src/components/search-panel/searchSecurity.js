/**
 * @typedef {object} SearchTextSegment
 * @property {string} text
 * @property {boolean} highlighted
 */

/**
 * Split search-result text into plain-text and highlighted segments.
 *
 * The caller renders `text` through normal Svelte interpolation. No source
 * text or query text is ever promoted to HTML.
 *
 * @param {string} text
 * @param {string} query
 * @returns {SearchTextSegment[]}
 */
export function buildSearchTextSegments(text, query) {
  const seen = new Set();
  const terms = query
    .split(/\s+/u)
    .filter(term => term.length > 0)
    .filter(term => {
      const key = term.toLowerCase();
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .sort((left, right) => right.length - left.length);

  if (text.length === 0 || terms.length === 0) {
    return text.length === 0 ? [] : [{ text, highlighted: false }];
  }

  const escapedTerms = terms.map(escapeRegExp);
  const matcher = new RegExp(escapedTerms.join('|'), 'gi');
  /** @type {SearchTextSegment[]} */
  const segments = [];
  let cursor = 0;

  for (const match of text.matchAll(matcher)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      segments.push({
        text: text.slice(cursor, index),
        highlighted: false
      });
    }
    segments.push({ text: match[0], highlighted: true });
    cursor = index + match[0].length;
  }

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), highlighted: false });
  }

  return segments.length > 0 ? segments : [{ text, highlighted: false }];
}

/**
 * @param {string} value
 * @returns {string}
 */
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
