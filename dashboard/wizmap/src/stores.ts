import { writable } from 'svelte/store';
import type { PromptPoint } from './types/embedding-types';
import d3 from './utils/d3-import';

export interface SearchBarStoreValue {
  shown: boolean;
  results: PromptPoint[];
  query: string;
  queryID: number;
  highlightSearchPoint: (point: PromptPoint | undefined) => void;
}

export interface FooterStoreValue {
  numPoints: number;
  curZoomTransform: d3.ZoomTransform;
  xScale: d3.ScaleLinear<number, number, never>;
  embeddingName: string;
  messageID: number;
  messageCommand: 'zoomIn' | 'zoomOut' | 'zoomReset' | '';
}

export interface TooltipStoreValue {
  show: boolean;
  html: string;
  x: number;
  y: number;
  width: number;
  maxWidth: number;
  fontSize: number;
  orientation: string;
  mouseoverTimeout: number | null;
}

export const getSearchBarStoreDefaultValue = (): SearchBarStoreValue => {
  return {
    shown: false,
    results: [],
    query: '',
    queryID: 0,
    highlightSearchPoint: (point: PromptPoint | undefined) => {
      // pass
    }
  };
};

export const getFooterStoreDefaultValue = (): FooterStoreValue => {
  return {
    numPoints: 0,
    curZoomTransform: d3.zoomIdentity,
    xScale: d3.scaleLinear(),
    embeddingName: 'Embedding',
    messageID: 0,
    messageCommand: ''
  };
};

export const getTooltipStoreDefaultValue = (): TooltipStoreValue => {
  return {
    show: false,
    html: 'null',
    x: 0,
    y: 0,
    width: 0,
    maxWidth: 300,
    fontSize: 14,
    orientation: 's',
    mouseoverTimeout: null
  };
};

export const getSearchBarStore = () => {
  return writable(getSearchBarStoreDefaultValue());
};

export const getFooterStore = () => {
  return writable(getFooterStoreDefaultValue());
};

export const getTooltipStore = () => {
  return writable(getTooltipStoreDefaultValue());
};
