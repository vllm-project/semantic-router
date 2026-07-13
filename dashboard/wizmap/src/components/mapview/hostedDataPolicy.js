/**
 * Resolve an optional hosted-data request while keeping both data sources on
 * the dashboard origin. Invalid or incomplete requests fail closed.
 *
 * @param {URLSearchParams} searchParams
 * @param {string} origin
 * @returns {{
 *   requested: boolean,
 *   dataURLs: {metadata: string, point: string} | null
 * }}
 */
export function resolveHostedDataURLs(searchParams, origin) {
  const requested =
    searchParams.has('metadataURL') || searchParams.has('dataURL');
  if (!requested) {
    return { requested: false, dataURLs: null };
  }

  const metadata = resolveSameOriginHTTPURL(
    searchParams.get('metadataURL'),
    origin
  );
  const point = resolveSameOriginHTTPURL(searchParams.get('dataURL'), origin);
  if (metadata === null || point === null) {
    return { requested: true, dataURLs: null };
  }

  return {
    requested: true,
    dataURLs: { metadata, point }
  };
}

/**
 * @param {string | null} rawValue
 * @param {string} origin
 * @returns {string | null}
 */
function resolveSameOriginHTTPURL(rawValue, origin) {
  if (rawValue === null || rawValue.trim() === '') {
    return null;
  }

  try {
    const base = new URL(origin);
    if (base.protocol !== 'http:' && base.protocol !== 'https:') {
      return null;
    }

    const candidate = new URL(rawValue.trim(), `${base.origin}/`);
    if (
      candidate.origin !== base.origin ||
      (candidate.protocol !== 'http:' && candidate.protocol !== 'https:') ||
      candidate.username !== '' ||
      candidate.password !== '' ||
      candidate.hash !== ''
    ) {
      return null;
    }

    return `${candidate.pathname}${candidate.search}`;
  } catch {
    return null;
  }
}
