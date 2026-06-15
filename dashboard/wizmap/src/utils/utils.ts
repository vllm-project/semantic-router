// Author: Jay Wang (jay@zijie.wang)
// License: MIT

import d3 from './d3-import';

// import type { SvelteComponent } from 'svelte';

/**
 * Round a number to a given decimal.
 * @param {number} num Number to round
 * @param {number} decimal Decimal place
 * @returns number
 */
export const round = (num: number, decimal: number) => {
  return Math.round((num + Number.EPSILON) * 10 ** decimal) / 10 ** decimal;
};

/**
 * Get a random number between [min, max], inclusive
 * @param {number} min Min value
 * @param {number} max Max value
 * @returns number
 */
export const random = (min: number, max: number) => {
  return Math.floor(Math.random() * (max - min)) + min;
};

/**
 * Download a JSON file
 * @param {any} object
 * @param {HTMLElement | null} [dlAnchorElem]
 * @param {string} [fileName]
 */

export const downloadJSON = (
  object: object,
  dlAnchorElem: HTMLElement | null = null,
  fileName = 'download.json'
) => {
  const dataStr =
    'data:text/json;charset=utf-8,' +
    encodeURIComponent(JSON.stringify(object));

  // Create dlAnchor if it is not given
  let myDlAnchorElem = dlAnchorElem;
  let needToRemoveAnchor = false;

  if (dlAnchorElem === null) {
    myDlAnchorElem = document.createElement('a');
    myDlAnchorElem.classList.add('download-anchor');
    myDlAnchorElem.style.display = 'none';
    needToRemoveAnchor = true;
  }

  myDlAnchorElem?.setAttribute('href', dataStr);
  myDlAnchorElem?.setAttribute('download', `${fileName}`);
  myDlAnchorElem?.click();

  if (needToRemoveAnchor) {
    myDlAnchorElem?.remove();
  }
};

/**
 * Download a text file
 * @param {string} textString
 * @param {HTMLElement | null} [dlAnchorElem]
 * @param {string} [fileName]
 */

export const downloadText = (
  textString: string,
  dlAnchorElem: HTMLElement | null,
  fileName = 'download.json'
) => {
  const dataStr =
    'data:text/plain;charset=utf-8,' + encodeURIComponent(textString);

  // Create dlAnchor if it is not given
  let myDlAnchorElem = dlAnchorElem;
  let needToRemoveAnchor = false;

  if (dlAnchorElem === null) {
    myDlAnchorElem = document.createElement('a');
    myDlAnchorElem.style.display = 'none';
    needToRemoveAnchor = true;
  }

  myDlAnchorElem?.setAttribute('href', dataStr);
  myDlAnchorElem?.setAttribute('download', `${fileName}`);
  myDlAnchorElem?.click();

  if (needToRemoveAnchor) {
    myDlAnchorElem?.remove();
  }
};

/**
 * Compute the luminance of a RGB color
 * https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
 * @param color [R, G, B in 0..255]
 * @returns number
 */

export const getLuminance = (color: number[]) => {
  const r = color[0];
  const g = color[1];
  const b = color[2];

  // Some strange required transformations
  const transformedRGB = [r, g, b].map(v => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });

  return (
    transformedRGB[0] * 0.2126 +
    transformedRGB[1] * 0.7152 +
    transformedRGB[2] * 0.0722
  );
};

/**
 * Compute color contrast ratio
 * @param color1 [r, g, b] in 255 scale
 * @param color2 [r, g, b] in 255 scale
 * @returns Contrast ratio
 */

export const getContrastRatio = (color1: number[], color2: number[]) => {
  const color1L = getLuminance(color1);
  const color2L = getLuminance(color2);
  const ratio =
    color1L > color2L
      ? (color2L + 0.05) / (color1L + 0.05)
      : (color1L + 0.05) / (color2L + 0.05);
  return ratio;
};

/**
 * Check if two colors have enough contrast
 * @param color1 [r, g, b] in 255 scale
 * @param color2 [r, g, b] in 255 scale
 * @param condition 'AA' or 'AAA'
 * @param smallText If it is small text
 * @returns If two colors have enough contrast
 */

export const haveContrast = (
  color1: number[],
  color2: number[],
  condition = 'AAA',
  smallText = true
) => {
  const ratio = getContrastRatio(color1, color2);

  // Compare the ratio with different thresholds
  if (condition === 'AA') {
    if (smallText) {
      return ratio <= 1 / 4.5;
    } else {
      return ratio <= 1 / 3;
    }
  } else {
    if (smallText) {
      return ratio <= 1 / 7;
    } else {
      return ratio <= 1 / 4.5;
    }
  }
};

/**
 * Check if two sets are the same
 * @param set1 Set 1
 * @param set2 Set 2
 */
export const setsAreEqual = <T>(set1: Set<T>, set2: Set<T>): boolean => {
  return set1.size === set2.size && [...set1].every(d => set2.has(d));
};

/**
 * Get the file name and file extension from a File object
 * @param {File} file File object
 * @returns [file name, file extension]
 */

export const splitFileName = (file: File) => {
  const name = file.name;
  const lastDot = name.lastIndexOf('.');
  const value = name.slice(0, lastDot);
  const extension = name.slice(lastDot + 1);
  return [value, extension];
};

/**
 * Split the reader stream text by a string
 * @param sep String used to separate the input string
 * @returns TransformStream
 */
export const splitStreamTransform = (sep: string) => {
  let buffer = '';

  const transform = new TransformStream({
    transform: (chunk, controller) => {
      buffer += chunk;
      const parts = buffer.split(sep);
      parts.slice(0, -1).forEach(part => controller.enqueue(part));
      buffer = parts[parts.length - 1];
    },
    flush: controller => {
      if (buffer) {
        controller.enqueue(buffer);
      }
    }
  });

  return transform;
};

/**
 * Parse the input stream as JSON
 * @returns TransformStream
 */
export const parseJSONTransform = () => {
  const transform = new TransformStream({
    transform: (chunk, controller) => {
      controller.enqueue(JSON.parse(chunk as string));
    }
  });
  return transform;
};

const timeitQueue = new Set();
/**
 * Trace the execution time
 * @param label Label for the time tracer
 * @param show Whether to printout the output in console
 */
export const timeit = (label: string, show: boolean) => {
  if (show) {
    if (timeitQueue.has(label)) {
      console.timeEnd(label);
      timeitQueue.delete(label);
    } else {
      console.time(label);
      timeitQueue.add(label);
    }
  }
};

/**
 * Convert a color from rgb to hex
 * @param r Value in the red channel
 * @param g Value in the green channel
 * @param b Value in the blue channel
 * @returns Hex string
 */
export const rgbToHex = (r: number, g: number, b: number) => {
  const numToHex = (number: number) => {
    const hex = number.toString(16);
    if (hex.length == 1) {
      return `0${hex}`;
    } else {
      return hex;
    }
  };
  return `#${numToHex(r)}${numToHex(g)}${numToHex(b)}`;
};

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Detect if two rectangles overlap.
 * https://stackoverflow.com/a/306332
 *
 * @param rect1 Rectangle 1
 * @param rect2 Rectangle 2
 * @returns True if these two rectangles overlap.
 */
export const rectsIntersect = (rect1: Rect, rect2: Rect) => {
  const right1 = rect1.x + rect1.width;
  const right2 = rect2.x + rect2.width;

  const bottom1 = rect1.y + rect1.height;
  const bottom2 = rect2.y + rect2.height;

  return (
    rect1.x < right2 &&
    right1 > rect2.x &&
    rect1.y < bottom2 &&
    bottom1 > rect2.y
  );
};

/**
 * Get a uniformly random sample from a list.
 * @param items Array of items to sample from
 * @param size Target size of the sample
 * @param seed Random seed (default to 1212)
 * @param replace True if sample with replace
 * @returns Sampled items
 */
export const getRandomSamples = <T>(
  items: Array<T>,
  size: number,
  seed = 1212,
  replace = false
) => {
  const targetSize = Math.min(size, items.length);
  const threshold = targetSize / items.length;
  const randomUniform = d3.randomUniform.source(d3.randomLcg(seed))(0, 1);

  const sampledItems: Array<T> = [];
  const sampledIndexes: Set<number> = new Set();

  // Repeat sampling until we have enough points sampled
  while (sampledItems.length < targetSize) {
    for (const [i, item] of items.entries()) {
      if ((replace || !sampledIndexes.has(i)) && randomUniform() <= threshold) {
        sampledIndexes.add(i);
        sampledItems.push(item);

        // Exit early if we have enough points
        if (sampledItems.length >= targetSize) break;
      }
    }
  }

  return sampledItems;
};

/**
 * A helper function to break up a long function into multiple tasks
 * https://web.dev/optimize-long-tasks/
 * @returns A promise equivalent to sleep(0)
 */
export const yieldToMain = () => {
  return new Promise(resolve => {
    setTimeout(resolve, 0);
  });
};
