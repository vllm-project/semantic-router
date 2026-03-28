import type { Embedding } from './Embedding';
import d3 from '../../utils/d3-import';
import type { PromptPoint } from '../../types/embedding-types';
import { timeit, rgbToHex } from '../../utils/utils';
import { config } from '../../config/config';

const SCATTER_DOT_RADIUS = 0.7;
const SCATTER_BACK_DOT_RADIUS = 1;
const TAU = 2 * Math.PI;
const DEBUG = config.debug;

let pointMouseleaveTimer: number | null = null;
let pointMouseenterTimer: number | null = null;

/**
 * Get a unique color in hex.
 */
export function getNextUniqueColor(this: Embedding) {
  if (this.colorPointMap.size >= 256 * 256 * 256) {
    console.error('Unique color overflow.');
    return '#fffff';
  }

  const rng = d3.randomInt(0, 256);
  let hex = rgbToHex(rng(), rng(), rng());
  while (this.colorPointMap.has(hex) || hex === '#000000') {
    hex = rgbToHex(rng(), rng(), rng());
  }
  return hex;
}

/**
 * Draw a scatter plot for the UMAP on a canvas.
 */
export function drawScatterCanvas(this: Embedding) {
  if (!this.showPoint) return;

  const r =
    (SCATTER_DOT_RADIUS * this.initZoomTransform.k) /
    Math.sqrt(this.curZoomTransform.k);
  this.pointCtx.globalAlpha = 0.5;

  for (const point of this.promptPoints) {
    const x = this.xScale(point.x);
    const y = this.yScale(point.y);

    this.pointCtx.beginPath();
    this.pointCtx.arc(x, y, r, 0, TAU);

    // Fill the data point circle
    const color = config.colors['gray-900'];

    // const modifiedColor = color.replace(
    //   /hsl\((.+),(.+),(.+)\)/,
    //   'hsla($1,$2,$3,0.5)'
    // );

    this.pointCtx.fillStyle = color;
    this.pointCtx.fill();
  }
}

/**
 * Draw a hidden scatter plot for the UMAP on a background canvas. We give
 * each dot a unique color for quicker mouseover detection.
 */
export function drawScatterBackCanvas(this: Embedding) {
  if (!this.showPoint) return;

  this.colorPointMap.clear();

  // Trick: here we draw a slightly larger circle when user zooms out the
  // viewpoint, so that the pixel coverage is higher (smoother/better
  // mouseover picking)
  const defaultR = SCATTER_BACK_DOT_RADIUS * this.initZoomTransform.k;
  const r = Math.max(defaultR, 3.5 * defaultR - this.curZoomTransform.k);

  for (const point of this.promptPoints) {
    this.pointBackCtx.beginPath();
    const x = this.xScale(point.x);
    const y = this.yScale(point.y);
    this.pointBackCtx.arc(x, y, r, 0, TAU);

    // Fill the data point with a unique color
    const color = this.getNextUniqueColor();
    this.colorPointMap.set(color, point);
    this.pointBackCtx.fillStyle = color;
    this.pointBackCtx.fill();
  }
}

/**
 * Highlight the point where the user hovers over
 * @param point The point that user hovers over
 */
export function highlightPoint(
  this: Embedding,
  point: PromptPoint | undefined
) {
  if (this.hoverMode !== 'point') return;
  if (!this.showPoint) return;
  if (point === this.hoverPoint) return;

  // Draw the point on the top svg
  const group = this.topSvg.select('g.top-content g.highlights');
  const oldHighlightPoint = group.select('circle.highlight-point');

  // Hovering empty space
  if (point === undefined) {
    if (!oldHighlightPoint.empty()) {
      if (pointMouseleaveTimer !== null) {
        clearTimeout(pointMouseleaveTimer);
        pointMouseleaveTimer = null;
      }

      // Clear the highlight and tooltip in a short delay
      pointMouseleaveTimer = setTimeout(() => {
        this.hoverPoint = null;
        oldHighlightPoint.remove();
        this.tooltipStoreValue.show = false;
        this.tooltipStore.set(this.tooltipStoreValue);
        pointMouseleaveTimer = null;
      }, 50);
    }
    return;
  }

  // Hovering over a point
  this.hoverPoint = point;
  if (pointMouseleaveTimer !== null) {
    clearTimeout(pointMouseleaveTimer);
    pointMouseleaveTimer = null;
  }

  // There is no point highlighted yet
  const highlightRadius = Math.max(
    SCATTER_DOT_RADIUS * 1.5,
    7 / this.curZoomTransform.k
  );
  const highlightStroke = 1.2 / this.curZoomTransform.k;

  if (oldHighlightPoint.empty()) {
    const highlightPoint = group
      .append('circle')
      .attr('class', 'highlight-point')
      .attr('cx', this.xScale(point.x))
      .attr('cy', this.yScale(point.y))
      .attr('r', highlightRadius)
      .style('stroke-width', highlightStroke);

    // Get the point position
    const position = highlightPoint.node()!.getBoundingClientRect();
    const curWidth = position.width;
    const tooltipCenterX = position.x + curWidth / 2;
    const tooltipCenterY = position.y;
    this.tooltipStoreValue.html = `
          <div class='tooltip-content' style='display: flex; flex-direction:
            column; justify-content: center;'>
            ${point.prompt}
          </div>
        `;

    this.tooltipStoreValue.x = tooltipCenterX;
    this.tooltipStoreValue.y = tooltipCenterY;
    this.tooltipStoreValue.show = true;

    if (pointMouseenterTimer !== null) {
      clearTimeout(pointMouseenterTimer);
      pointMouseenterTimer = null;
    }

    // Show the tooltip after a delay
    pointMouseenterTimer = setTimeout(() => {
      this.tooltipStore.set(this.tooltipStoreValue);
      pointMouseenterTimer = null;
    }, 300);
  } else {
    // There has been a highlighted point already
    oldHighlightPoint
      .attr('cx', this.xScale(point.x))
      .attr('cy', this.yScale(point.y))
      .attr('r', highlightRadius)
      .style('stroke-width', highlightStroke);

    // Get the point position
    const position = (
      oldHighlightPoint.node()! as HTMLElement
    ).getBoundingClientRect();
    const curWidth = position.width;
    const tooltipCenterX = position.x + curWidth / 2;
    const tooltipCenterY = position.y;
    this.tooltipStoreValue.html = `
          <div class='tooltip-content' style='display: flex; flex-direction:
            column; justify-content: center;'>
            ${point.prompt}
          </div>
        `;
    this.tooltipStoreValue.x = tooltipCenterX;
    this.tooltipStoreValue.y = tooltipCenterY;
    this.tooltipStoreValue.show = true;

    if (pointMouseenterTimer !== null) {
      clearTimeout(pointMouseenterTimer);
      pointMouseenterTimer = setTimeout(() => {
        this.tooltipStore.set(this.tooltipStoreValue);
        pointMouseenterTimer = null;
      }, 300);
    } else {
      this.tooltipStore.set(this.tooltipStoreValue);
    }
  }
}

/**
 * Redraw the point front canvas (clear, transform, draw).
 */
export function redrawFrontPoints(this: Embedding) {
  this.pointCtx.save();
  this.pointCtx.setTransform(1, 0, 0, 1, 0, 0);
  this.pointCtx.clearRect(
    0,
    0,
    this.svgFullSize.width,
    this.svgFullSize.height
  );
  this.pointCtx.translate(this.curZoomTransform.x, this.curZoomTransform.y);
  this.pointCtx.scale(this.curZoomTransform.k, this.curZoomTransform.k);
  this.drawScatterCanvas();
  this.pointCtx.restore();
}

/**
 * Redraw the point back canvas (clear, transform, draw).
 */
export function redrawBackPoints(this: Embedding) {
  this.pointBackCtx.save();
  this.pointBackCtx.setTransform(1, 0, 0, 1, 0, 0);
  this.pointBackCtx.clearRect(
    0,
    0,
    this.svgFullSize.width,
    this.svgFullSize.height
  );
  this.pointBackCtx.translate(this.curZoomTransform.x, this.curZoomTransform.y);
  this.pointBackCtx.scale(this.curZoomTransform.k, this.curZoomTransform.k);
  this.drawScatterBackCanvas();
  this.pointBackCtx.restore();
}
