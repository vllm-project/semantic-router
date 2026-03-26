import d3 from '../../../utils/d3-import';
import type {
  PromptPoint,
  SearchWorkerMessage
} from '../../../types/embedding-types';
import { timeit } from '../../../utils/utils';
import { config } from '../../../config/config';
import type * as Flexsearch from 'flexsearch';
import Index from 'flexsearch';

const index: Flexsearch.Index<string> = new Index() as Flexsearch.Index<string>;
const DEBUG = config.debug;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<SearchWorkerMessage>) => {
  // Stream point data
  switch (e.data.command) {
    case 'addPoints': {
      const { points } = e.data.payload;
      addPoints(points);
      break;
    }

    case 'startQuery': {
      const { query, queryID } = e.data.payload;
      searchPoint(query, queryID);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data);
      break;
    }
  }
};

/**
 * Add new points to the search index
 * @param points New points
 */
const addPoints = (points: PromptPoint[]) => {
  for (const point of points) {
    index.add(point.id, point.prompt);
  }
};

/**
 * Start a query
 * @param query Query string
 * @param queryID Query ID
 */
const searchPoint = (query: string, queryID: number) => {
  const resultIndexes = index.search(query, {
    limit: config.layout.searchLimit
  }) as unknown as number[];
  const message: SearchWorkerMessage = {
    command: 'finishQuery',
    payload: {
      queryID,
      resultIndexes
    }
  };
  postMessage(message);
};
