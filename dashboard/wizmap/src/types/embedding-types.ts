/**
 * Types for the Embedding view
 */
import type { Point, Rect } from './common-types';

export interface SearchResult {
  shortText: string;
  fullText: string;
  isSummary: boolean;
  point: PromptPoint;
}

/**
 * A collection of data files to set up WizMap
 */
export interface DataURLs {
  point: string;
  grid: string;
  topic: string;
}

export interface WebGLMatrices {
  dataScaleMatrix: number[];
  normalizeMatrix: number[];
}

export type EmbeddingInitSetting = {
  showContour: boolean;
  showPoint: boolean;
  showGrid: boolean;
  showLabel: boolean;
};

export type SearchWorkerMessage =
  | {
      command: 'addPoints';
      payload: {
        points: PromptPoint[];
      };
    }
  | {
      command: 'startQuery';
      payload: {
        query: string;
        queryID: number;
      };
    }
  | {
      command: 'finishQuery';
      payload: {
        queryID: number;
        resultIndexes: number[];
      };
    };

export type TreeWorkerMessage =
  | {
      command: 'initQuadtree';
      payload: {
        xRange: [number, number];
        yRange: [number, number];
        times: string[];
        groupIDs: number[];
      };
    }
  | {
      command: 'finishInitQuadtree';
      payload: null;
    }
  | {
      command: 'updateQuadtree';
      payload: {
        points: PromptPoint[];
      };
    }
  | {
      command: 'startQuadtreeSearch';
      payload: {
        x: number;
        y: number;
        time: string;
        groupID: number;
      };
    }
  | {
      command: 'finishQuadtreeSearch';
      payload: {
        point: PromptPoint;
      };
    };

export type LoaderWorkerMessage =
  | {
      command: 'startLoadData';
      payload: {
        /** JSON data url */
        url: string;
      };
    }
  | {
      command: 'transferLoadData';
      payload: {
        isFirstBatch: boolean;
        isLastBatch: boolean;
        points: PromptPoint[];
        loadedPointCount: number;
      };
    };

export interface LabelData {
  tileX: number;
  tileY: number;
  tileCenterX: number;
  tileCenterY: number;
  pointX: number;
  pointY: number;
  name: string;
}

export interface DrawnLabel extends Rect {
  direction: Direction;
  /**
   * Density center point's x coordinate.
   */
  pointX: number;

  /**
   * Density center point's y coordinate.
   */
  pointY: number;

  /**
   * Topic tile's top left point's x coordinate.
   */
  tileX: number;

  /**
   * Topic tile's top left point's y coordinate.
   */
  tileY: number;

  /**
   * Whether to hide this label.
   */
  toHide: boolean;

  /**
   * Label name.
   */
  name: string;

  /**
   * Label lines.
   */
  lines: string[];

  /**
   * Label element's x coordinate.
   */
  labelX: number;

  /**
   * Label element's y coordinate.
   */
  labelY: number;
}

export enum Direction {
  top = 'top',
  bottom = 'bottom',
  left = 'left',
  right = 'right'
}

export interface TopicDataJSON {
  extent: [[number, number], [number, number]];
  data: TopicDataMap;
}

interface TopicDataMap {
  [level: string]: TopicData[];
}

/**
 * A topic center point (x, y, topic label)
 */
export type TopicData = [number, number, string];

/**
 * A UMAP data point (x, y, prompt)
 * [x, y, text]
 * [x, y, text, year]
 * [x, y, text, year, group]
 */
export type UMAPPointStreamData =
  | [number, number, string]
  | [number, number, string, string]
  | [number, number, string, string, number];

export interface LevelTileMap {
  [level: string]: LevelTileDataItem[];
}

export interface LevelTileDataItem {
  w: [string, number][];
  p: [number, number, number, number];
}

export interface TileDataItem {
  /**
   * Node ID
   */
  i: number;

  /**
   * Position [x0, y0, x1, y1]
   */
  p: [number, number, number, number];

  /**
   * Number of points in this tile
   */
  s: number;

  /**
   * Tile level in the quadtree
   */
  l: number;
}

export interface QuadtreeNode {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface GridData {
  grid: number[][];
  xRange: [number, number];
  yRange: [number, number];
  padded: boolean;
  sampleSize: number;
  totalPointSize: number;
  embeddingName: string;
  timeGrids?: { [key: string]: number[][] };
  timeFormat?: string;
  timeCounter?: { [key: string]: number };
  groupGrids?: { [key: string]: number[][] };
  groupTotalPointSizes?: { [key: string]: number };
  groupNames?: string[];
}

export interface PromptPoint extends Point {
  prompt: string;
  id: number;
  time?: string;
  groupID?: number;
}

export interface PromptUMAPData {
  xs: number[];
  ys: number[];
  prompts: string[];
}
