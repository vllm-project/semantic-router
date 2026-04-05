import d3 from '../../../utils/d3-import';
import type {
  PromptPoint,
  UMAPPointStreamData,
  LoaderWorkerMessage
} from '../../../types/embedding-types';
import {
  splitStreamTransform,
  parseJSONTransform,
  timeit
} from '../../../utils/utils';
import { config } from '../../../config/config';

const DEBUG = config.debug;
const POINT_THRESHOLD = 5000;

let pendingDataPoints: PromptPoint[] = [];
let loadedPointCount = 0;
let sentPointCount = 0;

let lastDrawnPoints: PromptPoint[] | null = null;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<LoaderWorkerMessage>) => {
  // Stream point data
  switch (e.data.command) {
    case 'startLoadData': {
      console.log('Worker: start streaming data');
      timeit('Stream data', true);

      const url = e.data.payload.url;
      startLoadData(url);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

/**
 * Start loading the large UMAP data
 * @param url URL to the NDJSON file
 */
const startLoadData = (url: string) => {
  fetch(url).then(async response => {
    if (!response.ok) {
      console.error('Failed to load data', response);
      return;
    }

    const reader = response.body
      ?.pipeThrough(new TextDecoderStream())
      ?.pipeThrough(splitStreamTransform('\n'))
      ?.pipeThrough(parseJSONTransform())
      ?.getReader();

    while (true && reader !== undefined) {
      const result = await reader.read();
      const point = result.value as UMAPPointStreamData;
      const done = result.done;

      if (done) {
        timeit('Stream data', DEBUG);
        pointStreamFinished();
        break;
      } else {
        processPointStream(point);

        // // TODO: Remove me in prod
        // if (loadedPointCount >= 305000) {
        //   pointStreamFinished();
        //   timeit('Stream data', DEBUG);
        //   break;
        // }
      }
    }
  });
};

/**
 * Process one data point
 * @param point Loaded data point
 */
const processPointStream = (point: UMAPPointStreamData) => {
  const promptPoint: PromptPoint = {
    x: point[0],
    y: point[1],
    prompt: point[2],
    id: loadedPointCount
  };

  if (point.length > 3) {
    promptPoint.time = point[3]!;
  }

  if (point.length > 4) {
    promptPoint.groupID = point[4]!;
  }

  pendingDataPoints.push(promptPoint);
  loadedPointCount += 1;

  // Notify the main thread if we have load enough data
  if (pendingDataPoints.length >= POINT_THRESHOLD) {
    const result: LoaderWorkerMessage = {
      command: 'transferLoadData',
      payload: {
        isFirstBatch: lastDrawnPoints === null,
        isLastBatch: false,
        points: pendingDataPoints,
        loadedPointCount
      }
    };
    postMessage(result);

    sentPointCount += pendingDataPoints.length;
    lastDrawnPoints = pendingDataPoints.slice();
    pendingDataPoints = [];
  }
};

/**
 * Construct tree and notify the main thread when finish reading all data
 */
const pointStreamFinished = () => {
  // Send any left over points

  const result: LoaderWorkerMessage = {
    command: 'transferLoadData',
    payload: {
      isFirstBatch: lastDrawnPoints === null,
      isLastBatch: true,
      points: pendingDataPoints,
      loadedPointCount
    }
  };
  postMessage(result);

  sentPointCount += pendingDataPoints.length;
  lastDrawnPoints = pendingDataPoints.slice();
  pendingDataPoints = [];
};
