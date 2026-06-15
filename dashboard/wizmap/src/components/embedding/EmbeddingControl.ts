import { config } from '../../config/config';
import { timeit, rgbToHex } from '../../utils/utils';
import d3 from '../../utils/d3-import';
import { computePosition, flip, shift, offset, arrow } from '@floating-ui/dom';
import type { Embedding } from './Embedding';
import type { PromptPoint } from '../../types/embedding-types';
import fragmentShader from './shaders/point.frag?raw';
import vertexShader from './shaders/point.vert?raw';

const DEBUG = config.debug;
let sliderAnimationStartTime: number | null = null;
let curSliderProgress = 0;

/**
 * Event handler for the thumb mouse down on time slider
 * @param this Embedding object
 */
export function timeSliderMouseDownHandler(this: Embedding, e: MouseEvent) {
  e.preventDefault();
  e.stopPropagation();

  const thumb = e.target! as HTMLElement;
  if (!thumb.id.includes('thumb')) {
    console.error('Thumb event target is not thumb itself.');
  }

  const eventBlocker = this.component.querySelector('.grab-blocker')!;
  const track = thumb.parentElement!;
  const rangeTrack = track.querySelector('.range-track') as HTMLElement;

  const thumbBBox = thumb.getBoundingClientRect();
  const trackBBox = track.getBoundingClientRect();

  const trackWidth = trackBBox.width;
  // Need to use focus instead of active because FireFox doesn't treat dragging
  // as active
  thumb.focus();

  const mouseMoveHandler = (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (this.timeScale === null || this.timeFormatter === null) return;

    // Block the mouse event outside the slider
    eventBlocker.classList.add('activated');

    const deltaX = e.pageX - trackBBox.x;
    const progress = Math.min(1, Math.max(0, deltaX / trackWidth));

    // Move the thumb
    thumb.setAttribute('data-curValue', String(progress));

    // Compute the position to move the thumb to
    const xPos = progress * trackBBox.width - thumbBBox.width / 2;
    thumb.style.left = `${xPos}px`;
    rangeTrack.style.width = `${Math.max(0, xPos)}px`;

    // Update the label
    this.moveTimeSliderThumb(progress);
  };

  const mouseUpHandler = () => {
    document.removeEventListener('mousemove', mouseMoveHandler);
    document.removeEventListener('mouseup', mouseUpHandler);
    eventBlocker.classList.remove('activated');
    thumb.blur();
  };

  // Listen to mouse move on the whole page (users can drag outside of the
  // thumb, track, or even WizMap!)
  document.addEventListener('mousemove', mouseMoveHandler);
  document.addEventListener('mouseup', mouseUpHandler);
}

/**
 * Move the thumb on the time slider
 * @param this Embedding
 * @param progress Slider progress
 */
export function moveTimeSliderThumb(this: Embedding, progress: number) {
  if (this.timeScale === null || this.timeFormatter === null) return;
  if (!this.timeInspectMode) return;

  curSliderProgress = progress;

  const thumb = this.component.querySelector(
    '.time-menu .middle-thumb'
  ) as HTMLElement;
  const thumbLabel = thumb.querySelector('.thumb-label-span');
  const track = thumb.parentElement!;
  const rangeTrack = track.querySelector('.range-track') as HTMLElement;

  const thumbBBox = thumb.getBoundingClientRect();
  const trackBBox = track.getBoundingClientRect();

  // Compute the position to move the thumb to
  const xPos = progress * trackBBox.width - thumbBBox.width / 2;
  thumb.style.left = `${xPos}px`;
  rangeTrack.style.width = `${Math.max(0, xPos)}px`;

  // Update the label
  const curTime = this.timeScale.invert(
    progress * config.layout.timeSliderWidth
  );
  thumbLabel!.textContent = this.timeFormatter(curTime);
  this.curTime = this.timeFormatter(curTime);

  // Redraw the scatter plot
  if (anyTrue(this.showPoints)) {
    if (this.frontPositionBuffer && this.frontTextureCoordinateBuffer) {
      this.drawScatterPlot();
    }
  }

  // Redraw the contour plot
  if (anyTrue(this.showContours)) {
    this.drawContourTimeSlice();
  }
}

/**
 * Initialize the top control bar
 * @param this Embedding
 */
export function initTopControlBar(this: Embedding) {
  if (this.timeScale) {
    // Initialize the time slider
    const timeMenu = d3.select(this.component).select('.time-menu');

    // Bind event handlers
    timeMenu.select('.middle-thumb').on('mousedown', e => {
      this.timeSliderMouseDownHandler(e as MouseEvent);
    });

    timeMenu.select('.play-pause-button').on('click', () => {
      if (this.playingTimeSlider) {
        this.playPauseClickHandler(false);
      } else {
        this.playPauseClickHandler(true);
      }
      this.updateEmbedding();
    });

    // Initialize the slider label svg
    const labelSVG = d3.select(this.component).select('.time-menu .slider-svg');
    const axisGroup = labelSVG.append('g').attr('class', 'axis-group');
    axisGroup.call(d3.axisBottom(this.timeScale).ticks(5).tickSize(9));

    this.moveTimeSliderThumb(0);
  }
}

export function playPauseClickHandler(this: Embedding, play: boolean) {
  if (!play) {
    this.playingTimeSlider = false;
    // Hide the thumb label
    d3.select(this.component)
      .select('.time-menu .middle-thumb')
      .classed('animating', false);
  } else {
    this.playingTimeSlider = true;
    sliderAnimationStartTime = null;
    this.startTimeSliderAnimation();
  }
  this.updateEmbedding();
}

/**
 * Start to play the time slider animation
 * @param this Embedding
 */
export function startTimeSliderAnimation(this: Embedding) {
  const oneLoopMS = 8000;

  const loop = (timestamp: number) => {
    if (sliderAnimationStartTime === null) {
      // Fake the start time so that animation starts at the current progress
      const progressMS = curSliderProgress * oneLoopMS;
      sliderAnimationStartTime = timestamp - progressMS;
    }

    const elapsed = timestamp - sliderAnimationStartTime;
    let progress = elapsed / oneLoopMS;

    // Restart if we overshoot the slider
    if (progress > 1) {
      sliderAnimationStartTime = timestamp;
      progress = 0;
    }

    this.moveTimeSliderThumb(progress);

    if (this.playingTimeSlider) {
      window.requestAnimationFrame(loop);
    }
  };

  // Show the thumb label
  d3.select(this.component)
    .select('.time-menu .middle-thumb')
    .classed('animating', true);

  window.requestAnimationFrame(loop);
}

/**
 * Draw the KDE contour in the background.
 */
export function drawContourTimeSlice(this: Embedding) {
  if (this.gridData == null) {
    throw Error('Grid data not initialized');
  }

  if (this.gridData.timeGrids === undefined) {
    throw Error('Calling drawContourTimeSlice() when there is no time info');
  }

  if (!this.timeInspectMode || this.curTime === null) {
    throw Error('Not in timeInspectMode.');
  }

  const contourGroup = this.svg.select<SVGGElement>('.contour-group-time');

  const grid = this.gridData.timeGrids[this.curTime];
  if (grid === undefined) return;

  const gridData1D: number[] = [];
  for (const row of grid) {
    for (const item of row) {
      gridData1D.push(item);
    }
  }

  // Linear interpolate the levels to determine the thresholds
  const levels = config.layout.contourLevels;
  const thresholds: number[] = [];
  const minValue = Math.min(...gridData1D);
  const maxValue = Math.max(...gridData1D);
  const step = (maxValue - minValue) / levels;
  for (let i = 0; i < levels; i++) {
    thresholds.push(minValue + step * i);
  }

  let contours = d3
    .contours()
    .thresholds(thresholds)
    .size([this.gridData.grid.length, this.gridData.grid[0].length])(
    gridData1D
  );

  // Convert the scale of the generated paths
  const contourXScale = d3
    .scaleLinear()
    .domain([0, this.gridData.grid.length])
    .range(this.gridData.xRange);

  const contourYScale = d3
    .scaleLinear()
    .domain([0, this.gridData.grid[0].length])
    .range(this.gridData.yRange);

  contours = contours.map(item => {
    item.coordinates = item.coordinates.map(coordinates => {
      return coordinates.map(positions => {
        return positions.map(point => {
          return [
            this.xScale(contourXScale(point[0])),
            this.yScale(contourYScale(point[1]))
          ];
        });
      });
    });
    return item;
  });

  // Create a new blue interpolator based on d3.interpolateBlues
  // (starting from white here)
  const colorInterpolate = d3.interpolateLab(
    '#ffffff',
    config.colors['purple-800']
  );
  const colorScale = d3.scaleSequential(d3.extent(thresholds) as number[], d =>
    colorInterpolate(d / 1)
  );

  // Draw the contours
  contourGroup
    .selectAll<SVGPathElement, d3.ContourMultiPolygon>('path')
    .data(contours.slice(1))
    .join('path')
    .attr('fill', d => colorScale(d.value))
    .attr('d', d3.geoPath());

  // Animate the contours with Flubber (not working)
  // contourGroup
  //   .selectAll<SVGPathElement, d3.ContourMultiPolygon>('path')
  //   .data(contours)
  //   .join(
  //     enter => {
  //       const path = enter
  //         .append('path')
  //         .style('opacity', 0)
  //         .attr('fill', d => colorScale(d.value))
  //         .attr('d', d3.geoPath());

  //       path.transition('update').duration(100).style('opacity', 1);

  //       return path;
  //     },
  //     update => {
  //       update.each((d, i, g) => {
  //         const element = d3.select(g[i]);
  //         const oldPath = element.attr('d');
  //         const newPath = d3.geoPath()(d)!;
  //         if (oldPath !== newPath) {
  //           const interpolator = interpolate(oldPath, newPath);
  //           element
  //             .transition('update')
  //             .duration(100)
  //             .attrTween('d', () => interpolator);
  //         }
  //       });
  //       return update;
  //     },
  //     exit => {
  //       exit
  //         .transition('update')
  //         .duration(100)
  //         .style('opacity', 0)
  //         .on('end', () => {
  //           exit.remove();
  //         });
  //       return exit;
  //     }
  //   );
}

const anyTrue = (items: boolean[]) => items.reduce((a, b) => a || b);
const allTrue = (items: boolean[]) => items.reduce((a, b) => a && b);
