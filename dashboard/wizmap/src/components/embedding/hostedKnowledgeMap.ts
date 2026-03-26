import { UMAP } from 'umap-js';
import type {
  GridData,
  KBRawPointRecord,
  KnowledgeMapMetadata,
  PromptPoint,
  TopicDataJSON
} from '../../types/embedding-types';

const KB_GRID_SIZE = 64;
const MIN_TOPIC_LEVEL = 2;
const MAX_TOPIC_LEVEL = 9;
const DEFAULT_GROUP_TOPIC_LEVEL = 4;
const DEFAULT_LABEL_TOPIC_LEVEL = 6;
const GROUP_TILE_SPACING_RATIO = 0.72;
const LABEL_TILE_SPACING_RATIO = 0.56;

export interface HostedKnowledgeMapProjection {
  points: PromptPoint[];
  gridData: GridData;
  topicData: TopicDataJSON;
}

export async function loadHostedKnowledgeMapProjection(
  metadataURL: string,
  pointURL: string
): Promise<HostedKnowledgeMapProjection> {
  const [metadata, rawPoints] = await Promise.all([
    fetchHostedKnowledgeMapMetadata(metadataURL),
    fetchHostedKnowledgeMapPoints(pointURL)
  ]);

  if (rawPoints.length === 0) {
    throw new Error(`Knowledge base ${metadata.name} returned no points`);
  }

  const coordinates = await projectKnowledgeMapPoints(rawPoints, metadata.name);
  const points = coordinates.map(([x, y], index) => {
    const record = rawPoints[index];
    return {
      x,
      y,
      prompt: record.text,
      id: index,
      groupID: record.label_index,
      labelName: record.label_name
    } satisfies PromptPoint;
  });

  const [xRange, yRange] = computePointRanges(points);
  return {
    points,
    gridData: {
      grid: smoothGrid(buildDensityGrid(points, xRange, yRange), 2),
      xRange,
      yRange,
      padded: false,
      sampleSize: points.length,
      totalPointSize: points.length,
      embeddingName: `${metadata.name} (${metadata.model_type})`
    },
    topicData: buildTopicData(points, metadata.groups ?? {}, metadata.label_names, xRange, yRange)
  };
}

async function fetchHostedKnowledgeMapMetadata(
  url: string
): Promise<KnowledgeMapMetadata> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load knowledge map metadata: HTTP ${response.status}`);
  }
  return (await response.json()) as KnowledgeMapMetadata;
}

async function fetchHostedKnowledgeMapPoints(
  url: string
): Promise<KBRawPointRecord[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load knowledge map points: HTTP ${response.status}`);
  }
  const payload = await response.text();
  return payload
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map(line => JSON.parse(line) as KBRawPointRecord);
}

async function projectKnowledgeMapPoints(
  rawPoints: KBRawPointRecord[],
  seedSource: string
): Promise<[number, number][]> {
  if (rawPoints.length === 1) {
    return [[0, 0]];
  }
  if (rawPoints.length === 2) {
    return [
      [-1, 0],
      [1, 0]
    ];
  }

  const vectors = rawPoints.map(point => point.vector);
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.max(2, Math.min(15, rawPoints.length-1)),
    minDist: 0.12,
    spread: 1.4,
    random: seededRandom(seedSource)
  });
  const embedding = await umap.fitAsync(vectors, () => true);
  return embedding.map(([x, y]) => [x, y]);
}

function seededRandom(seedSource: string): () => number {
  let seed = 0;
  for (const char of seedSource) {
    seed = (seed*31 + char.charCodeAt(0)) >>> 0;
  }
  if (seed === 0) {
    seed = 0x9e3779b9;
  }
  return () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) % 1_000_000) / 1_000_000;
  };
}

function computePointRanges(points: PromptPoint[]): [[number, number], [number, number]] {
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;

  for (const point of points.slice(1)) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  return [
    [minX, maxX],
    [minY, maxY]
  ];
}

function buildDensityGrid(
  points: PromptPoint[],
  xRange: [number, number],
  yRange: [number, number]
): number[][] {
  const grid = newGrid(KB_GRID_SIZE);
  const xSpan = xRange[1] - xRange[0] || 1;
  const ySpan = yRange[1] - yRange[0] || 1;

  for (const point of points) {
    const xIndex = clampGridIndex(
      Math.round(((point.x - xRange[0]) / xSpan) * (KB_GRID_SIZE - 1))
    );
    const yIndex = clampGridIndex(
      Math.round(((point.y - yRange[0]) / ySpan) * (KB_GRID_SIZE - 1))
    );
    grid[yIndex][xIndex] += 1;
  }

  return grid;
}

function newGrid(size: number): number[][] {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

function clampGridIndex(index: number): number {
  if (index < 0) {
    return 0;
  }
  if (index >= KB_GRID_SIZE) {
    return KB_GRID_SIZE - 1;
  }
  return index;
}

function smoothGrid(grid: number[][], passes: number): number[][] {
  if (passes <= 0 || grid.length === 0) {
    return grid;
  }
  let current = grid;
  for (let pass = 0; pass < passes; pass += 1) {
    const next = newGrid(current.length);
    for (let yIndex = 0; yIndex < current.length; yIndex += 1) {
      for (let xIndex = 0; xIndex < current.length; xIndex += 1) {
        let sum = 0;
        let weight = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          const neighborY = yIndex + dy;
          if (neighborY < 0 || neighborY >= current.length) {
            continue;
          }
          for (let dx = -1; dx <= 1; dx += 1) {
            const neighborX = xIndex + dx;
            if (neighborX < 0 || neighborX >= current.length) {
              continue;
            }
            let neighborWeight = 1;
            if (dx === 0 && dy === 0) {
              neighborWeight = 4;
            } else if (dx === 0 || dy === 0) {
              neighborWeight = 2;
            }
            sum += current[neighborY][neighborX] * neighborWeight;
            weight += neighborWeight;
          }
        }
        next[yIndex][xIndex] = weight === 0 ? 0 : sum / weight;
      }
    }
    current = next;
  }
  return current;
}

function buildTopicData(
  points: PromptPoint[],
  groups: Record<string, string[]>,
  labelNames: string[],
  xRange: [number, number],
  yRange: [number, number]
): TopicDataJSON {
  const labelBuckets = new Map<string, PromptPoint[]>();
  for (const point of points) {
    const labelName = point.labelName ?? '';
    if (!labelBuckets.has(labelName)) {
      labelBuckets.set(labelName, []);
    }
    labelBuckets.get(labelName)!.push(point);
  }

  const data: Record<string, [number, number, string][]> = {};
  const groupTopics: [number, number, string][] = [];
  for (const [groupName, memberLabels] of Object.entries(groups).sort((left, right) =>
    left[0].localeCompare(right[0])
  )) {
    const memberPoints: PromptPoint[] = [];
    for (const labelName of memberLabels) {
      memberPoints.push(...(labelBuckets.get(labelName) ?? []));
    }
    if (memberPoints.length > 0) {
      groupTopics.push([...centroid(memberPoints), groupName]);
    }
  }

  const labelTopics: [number, number, string][] = [];
  for (const labelName of labelNames) {
    const memberPoints = labelBuckets.get(labelName) ?? [];
    if (memberPoints.length > 0) {
      labelTopics.push([...centroid(memberPoints), labelName]);
    }
  }

  let groupLevel = computeTopicLevel(
    groupTopics,
    DEFAULT_GROUP_TOPIC_LEVEL,
    GROUP_TILE_SPACING_RATIO
  );
  let labelLevel = computeTopicLevel(
    labelTopics,
    DEFAULT_LABEL_TOPIC_LEVEL,
    LABEL_TILE_SPACING_RATIO
  );

  if (groupTopics.length > 0 && labelTopics.length > 0) {
    labelLevel = clampTopicLevel(Math.max(labelLevel, groupLevel + 1));
    if (groupLevel >= labelLevel) {
      groupLevel = clampTopicLevel(labelLevel - 1);
    }
  }

  if (groupTopics.length > 0) {
    data[`${groupLevel}`] = groupTopics;
  }
  if (labelTopics.length > 0) {
    data[`${labelLevel}`] = labelTopics;
  }

  if (Object.keys(data).length === 0) {
    data[`${DEFAULT_LABEL_TOPIC_LEVEL}`] = [[0, 0, 'knowledge']];
  }

  return {
    extent: [
      [xRange[0], yRange[0]],
      [xRange[1], yRange[1]]
    ],
    data
  };
}

function centroid(points: PromptPoint[]): [number, number] {
  const totals = points.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      return acc;
    },
    { x: 0, y: 0 }
  );
  return [totals.x / points.length, totals.y / points.length];
}

function computeTopicLevel(
  topics: [number, number, string][],
  fallbackLevel: number,
  spacingRatio: number
): number {
  if (topics.length <= 1) {
    return fallbackLevel;
  }

  const bounds = computeTopicBounds(topics);
  const extentSpan = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY
  );
  if (!Number.isFinite(extentSpan) || extentSpan <= 0) {
    return fallbackLevel;
  }

  const nearestNeighborDistances = topics
    .map((topic, index) => nearestNeighborDistance(topics, index))
    .filter(distance => Number.isFinite(distance) && distance > 0)
    .sort((left, right) => left - right);

  if (nearestNeighborDistances.length === 0) {
    return fallbackLevel;
  }

  const spacing = quantile(nearestNeighborDistances, 0.35);
  const targetTileWidth = Math.max(
    extentSpan / Math.pow(2, MAX_TOPIC_LEVEL),
    spacing * spacingRatio
  );
  const rawLevel = Math.ceil(Math.log2(extentSpan / targetTileWidth));
  return clampTopicLevel(rawLevel);
}

function computeTopicBounds(topics: [number, number, string][]) {
  let minX = topics[0][0];
  let maxX = topics[0][0];
  let minY = topics[0][1];
  let maxY = topics[0][1];

  for (const topic of topics.slice(1)) {
    minX = Math.min(minX, topic[0]);
    maxX = Math.max(maxX, topic[0]);
    minY = Math.min(minY, topic[1]);
    maxY = Math.max(maxY, topic[1]);
  }

  return { minX, maxX, minY, maxY };
}

function nearestNeighborDistance(
  topics: [number, number, string][],
  index: number
): number {
  let nearest = Infinity;
  const [x, y] = topics[index];

  for (let otherIndex = 0; otherIndex < topics.length; otherIndex += 1) {
    if (otherIndex === index) {
      continue;
    }
    const [otherX, otherY] = topics[otherIndex];
    const distance = Math.hypot(otherX - x, otherY - y);
    if (distance < nearest) {
      nearest = distance;
    }
  }

  return nearest;
}

function quantile(values: number[], q: number): number {
  if (values.length === 1) {
    return values[0];
  }
  const clampedQ = Math.min(Math.max(q, 0), 1);
  const position = (values.length - 1) * clampedQ;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return values[lower];
  }
  const weight = position - lower;
  return values[lower] * (1 - weight) + values[upper] * weight;
}

function clampTopicLevel(level: number): number {
  if (!Number.isFinite(level)) {
    return DEFAULT_LABEL_TOPIC_LEVEL;
  }
  return Math.max(MIN_TOPIC_LEVEL, Math.min(MAX_TOPIC_LEVEL, level));
}
