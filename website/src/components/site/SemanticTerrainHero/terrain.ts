export type TerrainContour = {
  color: string
  label: string
  max: number
  min: number
  text: string
}

export const TERRAIN_CONTOURS: TerrainContour[] = [
  {
    min: 0.15,
    max: 0.22,
    color: '#3f3f46',
    label: 'SIGNAL FIELD',
    text: 'SIGNAL FIELD. INTENT CONTEXT SAFETY MODALITY HISTORY KNOWLEDGE. ',
  },
  {
    min: 0.35,
    max: 0.42,
    color: '#71717a',
    label: 'PREFERENCE LAYER',
    text: 'PREFERENCE LAYER. USER PRODUCT WORKLOAD POLICY COST QUALITY LATENCY. ',
  },
  {
    min: 0.55,
    max: 0.62,
    color: '#a1a1aa',
    label: 'DECISION SURFACE',
    text: 'DECISION SURFACE. SELECT CASCADE COORDINATE FUSE VERIFY FALLBACK. ',
  },
  {
    min: 0.75,
    max: 0.82,
    color: '#e4e4e7',
    label: 'MODEL FLEET',
    text: 'MODEL FLEET. FRONTIER OPEN SPECIALIZED EDGE HETEROGENEOUS INFERENCE. ',
  },
]

export function getTerrainElevation(x: number, y: number): number {
  let elevation = 0
  let amplitude = 1
  let frequency = 0.003

  for (let octave = 0; octave < 4; octave += 1) {
    elevation += Math.sin(x * frequency + Math.cos(y * frequency * 0.8)) * amplitude
    elevation += Math.cos(y * frequency + Math.sin(x * frequency * 1.2)) * amplitude
    amplitude *= 0.5
    frequency *= 2
  }

  return (elevation + 2.5) / 5
}

export function findTerrainContourIndex(elevation: number): number {
  return TERRAIN_CONTOURS.findIndex(
    contour => elevation >= contour.min && elevation <= contour.max,
  )
}

export function takeWrappedText(
  source: string,
  cursor: number,
  characterCount: number,
): { cursor: number, text: string } {
  if (source.length === 0 || characterCount <= 0) {
    return { cursor, text: '' }
  }

  let text = ''
  let nextCursor = cursor % source.length

  while (text.length < characterCount) {
    const remaining = characterCount - text.length
    const chunk = source.slice(nextCursor, nextCursor + remaining)
    text += chunk
    nextCursor = (nextCursor + chunk.length) % source.length
  }

  return { cursor: nextCursor, text }
}
