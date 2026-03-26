import type { Embedding } from './Embedding';
import d3 from '../../utils/d3-import';
import { computePosition, flip, shift, offset, arrow } from '@floating-ui/dom';
import type {
  TopicData,
  DrawnLabel,
  LabelData
} from '../../types/embedding-types';
import { Direction } from '../../types/embedding-types';
import type { Size, Padding, Rect, Point } from '../../types/common-types';
import { timeit, rectsIntersect } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';
import { config } from '../../config/config';

const IDEAL_TILE_WIDTH = 35;
const LABEL_SPLIT = '-';
let labelMouseenterTimer: string | null = null;
let labelMouseleaveTimer: number | null = null;

/**
 * Draw the labels using computed layouts
 * @param group Container group of current zoom level
 * @param drawnLabels Array of labels to draw
 * @param tileScreenWidth Tile width in the screen coordinate
 * @param idealTreeLevel Ideal tree level
 * @param fontSize Font size
 * @returns Drawn label selections
 */
export function drawLabels(
  this: Embedding,
  group: d3.Selection<d3.BaseType | SVGGElement, number, d3.BaseType, unknown>,
  drawnLabels: DrawnLabel[],
  tileScreenWidth: number,
  idealTreeLevel: number,
  fontSize: number
) {
  const transAddition = d3
    .transition('label-addition')
    .duration(300)
    .ease(d3.easeCubicInOut);

  const transRemoval = d3
    .transition('label-removal')
    .duration(100)
    .ease(d3.easeLinear);

  const enterFunc = (
    enter: d3.Selection<
      d3.EnterElement,
      DrawnLabel,
      d3.BaseType | SVGGElement,
      number
    >
  ) => {
    // Add the group element
    const labelGroup = enter
      .append('g')
      .attr('class', `label-group zoom-${idealTreeLevel}`)
      .classed('hidden', d => d.toHide);

    // Animation for individual group addition
    if (this.lastLabelNames.size > 0) {
      labelGroup
        .style('opacity', 0)
        .transition(transAddition)
        .style('opacity', 1);
    }

    // Draw the text label
    const text = labelGroup
      .append('text')
      .attr('class', d => `topic-label ${d.direction}`)
      .attr('transform', d => `translate(${d.labelX}, ${d.labelY})`)
      .style('font-size', `${fontSize}px`)
      .text(d => (d.lines.length > 1 ? null : d.lines[0]))
      .attr('paint-order', 'stroke')
      .style('stroke', '#fff')
      .style('stroke-width', 3.2 / this.curZoomTransform.k);

    text
      .append('tspan')
      .attr('class', 'line-1')
      .attr('x', 0)
      .attr('y', 0)
      .text(d => (d.lines.length > 1 ? d.lines[0] : ''));
    text
      .append('tspan')
      .attr('class', 'line-2')
      .attr('x', 0)
      .attr('y', 0)
      .attr('dy', 0.96 * fontSize)
      .text(d => (d.lines.length > 1 ? d.lines[1] : ''));

    // Draw the topic region
    const tileRect = labelGroup
      .append('rect')
      .attr('class', 'topic-tile')
      .attr('x', d => this.xScale(d.tileX))
      .attr('y', d => this.yScale(d.tileY))
      .attr('rx', 4 / this.curZoomTransform.k)
      .attr('ry', 4 / this.curZoomTransform.k)
      .attr('width', tileScreenWidth)
      .attr('height', tileScreenWidth)
      .style('stroke-width', 1.6 / this.curZoomTransform.k);

    // tileRect
    //   .clone(true)
    //   .lower()
    //   .attr('class', 'topic-tile-back')
    //   .style('stroke-width', 2.6 / this.curZoomTransform.k);

    // Add a dot to indicate the label direction
    labelGroup
      .append('path')
      .attr('class', 'direction-indicator')
      .attr('transform-origin', 'center')
      .each((d, i, g) => this.addTileIndicatorPath(d, i, g, tileScreenWidth));

    // // For debugging: show the label bounding box
    // labelGroup
    //   .append('rect')
    //   .attr('x', d => d.x)
    //   .attr('y', d => d.y)
    //   .attr('width', d => d.width)
    //   .attr('height', d => d.height)
    //   .style('fill', 'none')
    //   .style('stroke', 'orange');

    // // For debugging: draw the high density center
    // labelGroup
    //   .append('circle')
    //   .attr('cx', d => this.xScale(d.pointX))
    //   .attr('cy', d => this.yScale(d.pointY))
    //   .attr('r', 2);

    return labelGroup;
  };

  const updateFunc = (
    update: d3.Selection<
      d3.BaseType,
      DrawnLabel,
      d3.BaseType | SVGGElement,
      number
    >
  ) => {
    // Animate to hide labels
    const labelGroup = update.each((d, i, g) => {
      const selection = d3.select(g[i]);
      const lastToHide = selection.classed('hidden');
      if (!lastToHide && d.toHide && this.contoursInitialized) {
        selection
          .style('opacity', 1)
          .transition(transRemoval)
          .style('opacity', 0)
          .on('end', () => {
            selection.classed('hidden', d.toHide);
          });
      } else if (lastToHide && !d.toHide && this.contoursInitialized) {
        selection
          .style('opacity', 0)
          .classed('hidden', d.toHide)
          .transition(transRemoval)
          .style('opacity', 1);
      } else {
        selection.classed('hidden', d.toHide);
      }
    });

    // Update text location
    labelGroup
      .select('text')
      .style('stroke-width', 3.2 / this.curZoomTransform.k)
      .style('font-size', `${fontSize}px`)
      .each((d, i, g) => {
        const selection = d3.select(g[i]);
        const oldClass = selection.attr('class');
        const newClass = `topic-label ${d.direction}`;

        selection
          .attr('class', newClass)
          .select('.line-2')
          .attr('dy', 0.96 * fontSize);

        // If direction is changed, apply animation
        if (newClass !== oldClass) {
          selection
            .transition('update')
            .duration(300)
            .ease(d3.easeCubicInOut)
            .attr('transform', `translate(${d.labelX}, ${d.labelY})`);
        } else {
          selection.attr('transform', `translate(${d.labelX}, ${d.labelY})`);
        }
      });

    // Update the tile region
    labelGroup
      .select('rect.topic-tile')
      .attr('x', d => this.xScale(d.tileX))
      .attr('y', d => this.yScale(d.tileY))
      .attr('rx', 4 / this.curZoomTransform.k)
      .attr('ry', 4 / this.curZoomTransform.k)
      .attr('width', tileScreenWidth)
      .attr('height', tileScreenWidth)
      .style('stroke-width', 1.6 / this.curZoomTransform.k);

    // Update the dot orientation
    labelGroup
      .select<SVGPathElement>('path.direction-indicator')
      .each((d, i, g) => this.addTileIndicatorPath(d, i, g, tileScreenWidth));
    return update;
  };

  const exitFunc = (
    exit: d3.Selection<
      d3.BaseType,
      DrawnLabel,
      d3.BaseType | SVGGElement,
      number
    >
  ) => {
    // Animation for individual group removal
    if (this.lastLabelNames.size > 0 && this.contoursInitialized) {
      exit
        .transition(transRemoval)
        .style('opacity', 0)
        .on('end', () => {
          exit.remove();
        });
      return exit;
    } else {
      return exit.remove();
    }
  };

  const labelGroups = group
    .selectAll('g.label-group')
    .data(drawnLabels, d => (d as DrawnLabel).name)
    .join(
      enter => enterFunc(enter),
      update => updateFunc(update),
      exit => exitFunc(exit)
    );

  return labelGroups;
}

/**
 * Draw topic tile grid based on the zoom level
 * @param this Embedding
 * @param strokeColor Use this argument to override grid stroke color
 */
export function drawTopicGrid(
  this: Embedding,
  strokeColor: string | null = null
) {
  if (!this.showGrid) return;

  // Choose the topic tree level based on the current zoom level
  const idealTreeLevel = this.getIdealTopicTreeLevel();
  if (idealTreeLevel === null) return;

  // Crossfade two canvas elements if tree level changes
  if (this.lastGridTreeLevels.length > 0) {
    const lastLevel =
      this.lastGridTreeLevels[this.lastGridTreeLevels.length - 1];

    // Tree level changes
    if (idealTreeLevel !== lastLevel) {
      // Fade the last canvas
      this.topicCanvases[lastLevel % 2]
        .classed('faded', true)
        .on('transitionend', () => {
          if (this.lastGridTreeLevels[0] === lastLevel) {
            // No need to draw this faded canvas
            this.lastGridTreeLevels.shift();
          }
        });

      // Show the current canvas
      this.topicCanvases[idealTreeLevel % 2].classed('faded', false);

      // Track this new level
      this.lastGridTreeLevels.push(idealTreeLevel);

      // The stack only tracks two levels
      if (this.lastGridTreeLevels.length > 2) {
        this.lastGridTreeLevels.shift();
      }
    }
  }

  if (this.lastGridTreeLevels.length === 0) {
    this.lastGridTreeLevels.push(idealTreeLevel);
  }

  for (const treeLevel of this.lastGridTreeLevels) {
    const canvas = this.topicCanvases[
      treeLevel % 2
    ].node() as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const topicTree = this.topicLevelTrees.get(treeLevel)!;
    const treeExtent = topicTree.extent()!;
    const tileWidth =
      (treeExtent[1][0] - treeExtent[0][0]) / Math.pow(2, treeLevel);
    const tileScreenWidth = Math.abs(this.xScale(tileWidth) - this.xScale(0));

    //  Only draw the tiles that are visible
    const zoomBox = this.getCurZoomBox();
    interface NamedRect extends Rect {
      name: string;
      label: string;
    }

    const tiles = topicTree
      .data()
      .map(d => {
        const tileRect: NamedRect = {
          x: this.xScale(d[0] - tileWidth / 2),
          y: this.yScale(d[1] - tileWidth / 2),
          width: tileScreenWidth,
          height: tileScreenWidth,
          name: `${(d[0], d[1])}`,
          label: d[2]
        };
        return tileRect;
      })
      .filter(d => rectsIntersect(d, zoomBox));

    // Drw the tiles on a canvas
    ctx.save();

    // Use white stroke on contour background
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
    } else {
      if (anyTrue(this.showContours)) {
        ctx.strokeStyle = config.gridColorLight;
      } else {
        // Dark stroke if there is no contour background
        ctx.strokeStyle = config.gridColorDark;
      }
    }
    ctx.lineWidth = 1 / (4 * this.curZoomTransform.k);

    for (const tile of tiles) {
      ctx.moveTo(tile.x, tile.y);
      roundRect(
        ctx,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        4 / this.curZoomTransform.k
      );
      ctx.stroke();
    }
    ctx.restore();
  }
}

/**
 * Draw topic grid in one animation frame
 * @param this Embedding
 * @param time Current time
 * @param startTime Animation start time
 * @param duration Animation duration (ms)
 * @param colorScale Function map (0, 1) to a color
 * @returns Void
 */
export function drawTopicGridFrame(
  this: Embedding,
  time: number,
  startTime: number | null,
  duration: number,
  colorScale: (t: number) => string
) {
  const delta = time - (startTime ? startTime : time);
  const progress = delta / duration;
  if (progress > 1) return;

  // Draw a frame
  const curColor = colorScale(progress);
  this.redrawTopicGrid(curColor);

  // Next frame
  requestAnimationFrame(newTime => {
    this.drawTopicGridFrame(
      newTime,
      startTime ? startTime : time,
      duration,
      colorScale
    );
  });
}

/**
 * Draws a rounded rectangle using the current state of the canvas.
 * Derived from: https://stackoverflow.com/a/3368118/5379444
 * @param CanvasRenderingContext2D ctx
 * @param x x The top left x coordinate
 * @param y y The top left y coordinate
 * @param width width The width of the rectangle
 * @param height height The height of the rectangle
 * @param r radius  The corner radius; It can also be an object
 *  to specify different radii for corners
 */
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  r: number | [number, number, number, number]
) {
  const radius = {
    tl: 0,
    tr: 0,
    br: 0,
    bl: 0
  };

  if (typeof r === 'number') {
    radius.tl = r;
    radius.tr = r;
    radius.br = r;
    radius.bl = r;
  } else {
    radius.tl = r[0];
    radius.tr = r[1];
    radius.br = r[2];
    radius.bl = r[3];
  }
  ctx.beginPath();
  ctx.moveTo(x + radius.tl, y);
  ctx.lineTo(x + width - radius.tr, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
  ctx.lineTo(x + width, y + height - radius.br);
  ctx.quadraticCurveTo(
    x + width,
    y + height,
    x + width - radius.br,
    y + height
  );
  ctx.lineTo(x + radius.bl, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
  ctx.lineTo(x, y + radius.tl);
  ctx.quadraticCurveTo(x, y, x + radius.tl, y);
  ctx.closePath();
}

/**
 * Show the topic labels at different zoom scales.
 */
export function layoutTopicLabels(
  this: Embedding,
  maxLabels: number | null = null,
  canSkipLayout = true
) {
  if (this.topicLevelTrees.size <= 1) return;
  if (this.contours === null) return;
  if (!this.showLabel) return;

  // Fin near labels for high density regions
  const topicGroup = this.topSvg.select('g.top-content g.topics');

  const polygonCenters = [];
  for (let i = this.contours.length - 1; i >= 0; i--) {
    const contour = this.contours[i];

    // Compute the geometric center of each polygon
    for (const polygon of contour.coordinates) {
      const xs = [];
      const ys = [];
      for (const point of polygon[0]) {
        xs.push(point[0]);
        ys.push(point[1]);
      }
      const centerX = xs.reduce((a, b) => a + b) / xs.length;
      const centerY = ys.reduce((a, b) => a + b) / ys.length;
      polygonCenters.push([centerX, centerY, contour.value]);
    }
  }

  // Compute the current view extent based on the zoom
  const curZoomBox = this.getCurZoomBox();

  // Choose the topic tree level based on the current zoom level
  const idealTreeLevel = this.getIdealTopicTreeLevel()!;
  const topicTree = this.topicLevelTrees.get(idealTreeLevel)!;
  const treeExtent = topicTree.extent()!;
  const tileWidth =
    (treeExtent[1][0] - treeExtent[0][0]) / Math.pow(2, idealTreeLevel);
  const tileScreenWidth = Math.abs(this.xScale(tileWidth) - this.xScale(0));

  // Show animation when we shift zoom level
  const trans = d3.transition('removal').duration(400).ease(d3.easeCubicInOut);

  const group = topicGroup
    .selectAll('g.topics-content')
    .data([idealTreeLevel], d => d as number)
    .join(
      enter => {
        const newGroup = enter
          .append('g')
          .attr('class', d => `topics-content zoom-${d}`)
          .style('opacity', 0);

        newGroup.transition(trans).style('opacity', 1);

        if (!enter.empty()) {
          this.lastLabelNames = new Map();
        }

        return newGroup;
      },
      update => {
        return update;
      },
      exit => {
        if (this.lastLabelTreeLevel !== idealTreeLevel) {
          return exit
            .transition(trans)
            .style('opacity', 0)
            .on('end', () => {
              exit.remove();
            });
        } else {
          return exit;
        }
      }
    );

  // Find closest topic labels for each high density point
  const labelDataMap = new Map<string, LabelData>();
  const labelDataCounter = new Map<string, number>();
  const fontSize = 14 / this.curZoomTransform.k;

  // Skip layout if we have just done it
  if (canSkipLayout && Date.now() - this.lastLabelLayoutTime < 300) {
    // Draw the labels
    this.drawLabels(
      group,
      this.lastDrawnLabels,
      tileScreenWidth,
      idealTreeLevel,
      fontSize
    );
    return;
  }

  this.lastLabelTreeLevel = idealTreeLevel;

  for (const point of polygonCenters) {
    const viewX = this.xScale.invert(point[0]);
    const viewY = this.yScale.invert(point[1]);

    // Use 2D search to potentially detect multiple tiles for one density
    // center. Radius is a hyper parameter.
    const radius = tileWidth * 0.5;
    const closestTopics = search2DQuadTree(
      topicTree,
      viewX - radius,
      viewY - radius,
      viewX + radius,
      viewY + radius
    );

    for (const closestTopic of closestTopics) {
      const curLabelData: LabelData = {
        tileX: closestTopic[0] - tileWidth / 2,
        tileY: closestTopic[1] + tileWidth / 2,
        tileCenterX: closestTopic[0],
        tileCenterY: closestTopic[1],
        pointX: viewX,
        pointY: viewY,
        name: closestTopic[2]
      };

      if (labelDataCounter.has(curLabelData.name)) {
        labelDataCounter.set(
          curLabelData.name,
          labelDataCounter.get(curLabelData.name)! + point[2]
        );
      } else {
        labelDataCounter.set(curLabelData.name, point[2]);
        labelDataMap.set(curLabelData.name, curLabelData);
      }
    }
  }

  // Sort the label data by their accumulated density scores
  const sortedLabelData = [...labelDataCounter]
    .sort((a, b) => b[1] - a[1])
    .map(pair => labelDataMap.get(pair[0])!);

  const drawnLabels: DrawnLabel[] = [];
  const drawnTiles: Rect[] = [];

  const textHeight = fontSize * 1.1;
  const vPadding = 6.4 / this.curZoomTransform.k;
  const hPadding = 6.4 / this.curZoomTransform.k;

  // Count the number of labels that are shown
  let shownLabelNum = 0;
  let inViewLabelNum = 0;

  for (const label of sortedLabelData) {
    const twoLine = label.name.length > 12;
    let line1 = label.name.slice(0, Math.floor(label.name.length / 2));
    let line2 = label.name.slice(Math.floor(label.name.length / 2));

    if (twoLine && label.name.split(LABEL_SPLIT).length >= 4) {
      const words = label.name.split(LABEL_SPLIT);
      line1 = words.slice(0, 2).join('-') + '-';
      line2 = words.slice(2).join('-');
    }

    const textWidth = twoLine
      ? Math.max(
          getLatoTextWidth(line1, fontSize),
          getLatoTextWidth(line2, fontSize)
        )
      : getLatoTextWidth(label.name, fontSize);
    const curTextHeight = twoLine ? textHeight * 1.8 : textHeight;

    // Try 4 different layout
    let fit = true;
    let fitRect: Rect | null = null;
    let fitDirection: Direction | null = null;

    // Simple greedy heuristic:
    // https://en.wikipedia.org/wiki/Automatic_label_placement
    // (1) Prioritize left and right over top and bottom;
    const directions = [
      Direction.left,
      Direction.right,
      Direction.bottom,
      Direction.top
    ];

    // (2) Prioritize right if the tile is on the right half
    if (label.tileCenterX >= (treeExtent[0][0] + treeExtent[1][0]) / 2) {
      directions[0] = Direction.right;
      directions[1] = Direction.left;
    }

    // (3) pick the opposite direction as the connected drawn neighbor's tile.
    // We also use this iteration to check if this topic tile would overlaps
    // with previous labels.
    let neighborDirection: Direction | null = null;
    let tileIntersects = false;
    const curTileRect: Rect = {
      x: this.xScale(label.tileX),
      y: this.yScale(label.tileY),
      width: tileScreenWidth,
      height: tileScreenWidth
    };

    for (const drawnLabel of drawnLabels) {
      if (rectsIntersect(curTileRect, drawnLabel)) {
        tileIntersects = true;
        break;
      }

      const xDiff = Math.abs(drawnLabel.tileX - label.tileX);
      const yDiff = Math.abs(drawnLabel.tileY - label.tileY);
      if (xDiff + yDiff <= tileWidth) {
        neighborDirection = drawnLabel.direction;
      }
    }

    // Do now show this label if it overlaps
    if (tileIntersects) {
      continue;
    }

    switch (neighborDirection) {
      case Direction.left: {
        directions.splice(directions.indexOf(Direction.right), 1);
        directions.unshift(Direction.right);
        break;
      }
      case Direction.right: {
        directions.splice(directions.indexOf(Direction.left), 1);
        directions.unshift(Direction.left);
        break;
      }
      case Direction.top: {
        directions.splice(directions.indexOf(Direction.bottom), 1);
        directions.unshift(Direction.bottom);
        break;
      }
      case Direction.bottom: {
        directions.splice(directions.indexOf(Direction.top), 1);
        directions.unshift(Direction.top);
        break;
      }
      default: {
        break;
      }
    }

    // (4) Prioritize a safer direction first if there is another (future)
    // tile connecting to the current tile's left or right
    for (const futureLabel of sortedLabelData) {
      if (futureLabel.tileY === label.tileY) {
        const xDiff = futureLabel.tileX - label.tileX;
        if (xDiff === tileWidth) {
          // There is a future tile on the right, prioritize left
          directions.splice(directions.indexOf(Direction.left), 1);
          directions.unshift(Direction.left);
        } else if (xDiff === -tileWidth) {
          // There is a future tile on the left, prioritize right
          directions.splice(directions.indexOf(Direction.right), 1);
          directions.unshift(Direction.right);
        }
      }
    }

    // (5) Highest priority: prioritize previous shown direction to avoid
    // labels moving around
    if (this.lastLabelNames.has(label.name)) {
      const lastDirection = this.lastLabelNames.get(label.name)!;
      directions.splice(directions.indexOf(lastDirection), 1);
      directions.unshift(lastDirection);
    }

    for (const direction of directions) {
      fit = true;
      const curRect: Rect = {
        x: 0,
        y: 0,
        width: textWidth,
        height: curTextHeight
      };

      // Compute the bounding box for this current layout
      switch (direction) {
        case Direction.top:
          curRect.x = this.xScale(label.tileCenterX) - textWidth / 2;
          curRect.y =
            this.yScale(label.tileCenterY) -
            tileScreenWidth / 2 -
            curTextHeight -
            vPadding;
          break;

        case Direction.bottom:
          curRect.x = this.xScale(label.tileCenterX) - textWidth / 2;
          curRect.y =
            this.yScale(label.tileCenterY) + tileScreenWidth / 2 + vPadding;
          break;

        case Direction.left:
          curRect.x =
            this.xScale(label.tileCenterX) -
            tileScreenWidth / 2 -
            textWidth -
            hPadding;
          curRect.y = this.yScale(label.tileCenterY) - curTextHeight / 2;
          break;

        case Direction.right:
          curRect.x =
            this.xScale(label.tileCenterX) + tileScreenWidth / 2 + hPadding;
          curRect.y = this.yScale(label.tileCenterY) - curTextHeight / 2;
          break;

        default:
          console.error('Unknown direction value.');
          break;
      }

      // Compare the current direction with existing labels to see if there is
      // any overlapping
      for (const drawnLabel of drawnLabels) {
        if (rectsIntersect(curRect, drawnLabel)) {
          fit = false;
          break;
        }
      }

      // Compare the current direction with existing tile squares to see if
      // there is any overlapping
      for (const drawnTile of drawnTiles) {
        if (rectsIntersect(curRect, drawnTile)) {
          fit = false;
          break;
        }
      }

      // The current direction does not overlap with any existing rects
      if (fit) {
        fitRect = curRect;
        fitDirection = direction;
        break;
      }
    }

    // Draw this label if we find a location for it
    if (fit && fitRect && fitDirection) {
      const drawnLabel: DrawnLabel = {
        x: fitRect.x,
        y: fitRect.y,
        width: fitRect.width,
        height: fitRect.height,
        direction: fitDirection,
        pointX: label.pointX,
        pointY: label.pointY,
        tileX: label.tileX,
        tileY: label.tileY,
        toHide: false,
        name: label.name,
        lines: twoLine ? [line1, line2] : [label.name],
        labelX: this.xScale(label.tileCenterX),
        labelY: this.yScale(label.tileCenterY)
      };
      const drawnTile: Rect = {
        x: this.xScale(label.tileX),
        y: this.yScale(label.tileY),
        width: tileScreenWidth,
        height: tileScreenWidth
      };

      // Check if this label and tile rect intersects with the view extent
      drawnLabel.toHide =
        !rectsIntersect(fitRect, curZoomBox) &&
        !rectsIntersect(drawnTile, curZoomBox);

      if (!drawnLabel.toHide) {
        inViewLabelNum += 1;
        if (maxLabels !== null) {
          // We have shown enough labels, stop showing this label if we haven't
          // drawn it last time. Edge case: we stop showing extra labels during
          // the initial zoom triggered by drawContours()
          if (
            shownLabelNum >= maxLabels &&
            (!this.contoursInitialized || !this.lastLabelNames.has(label.name))
          ) {
            drawnLabel.toHide = true;
          } else {
            shownLabelNum += 1;
          }
        }
      }

      switch (fitDirection) {
        case Direction.top: {
          drawnLabel.labelY -=
            vPadding + tileScreenWidth / 2 + (twoLine ? textHeight : 0);
          break;
        }
        case Direction.bottom: {
          drawnLabel.labelY += vPadding + tileScreenWidth / 2;
          break;
        }
        case Direction.left: {
          drawnLabel.labelX -= hPadding + tileScreenWidth / 2;
          drawnLabel.labelY -= twoLine ? textHeight : 0;
          break;
        }
        case Direction.right: {
          drawnLabel.labelX += hPadding + tileScreenWidth / 2;
          drawnLabel.labelY -= twoLine ? textHeight : 0;
          break;
        }
        default: {
          console.error('Unknown layout value.');
          break;
        }
      }

      drawnLabels.push(drawnLabel);
      drawnTiles.push(drawnTile);
    }
  }

  this.lastDrawnLabels = drawnLabels;
  this.lastLabelLayoutTime = Date.now();

  // Draw the labels
  this.drawLabels(
    group,
    drawnLabels,
    tileScreenWidth,
    idealTreeLevel,
    fontSize
  );

  // Track the labels we have shown
  this.lastLabelNames = new Map();
  drawnLabels
    .filter(d => !d.toHide)
    .forEach(d => this.lastLabelNames.set(d.name, d.direction));

  this.maxLabelNum = inViewLabelNum;
  this.curLabelNum = shownLabelNum;

  const sliderElem = this.component.querySelector(
    '.label-menu input#slider-label-num'
  ) as HTMLInputElement;
  sliderElem.max = `${this.maxLabelNum}`;
  sliderElem.value = `${this.curLabelNum}`;

  const sliderCountElem = this.component.querySelector(
    '.label-menu span.slider-count'
  ) as HTMLSpanElement;
  sliderCountElem.innerText = `${this.curLabelNum}`;
}

/**
 * Get the ideal quadtree level based on the ideal tile width and the current
 * zoom level
 */
export function getIdealTopicTreeLevel(this: Embedding) {
  if (this.topicLevelTrees.size < 1) return null;

  let bestLevel = -1;
  let bestDistance = Infinity;

  for (const level of this.topicLevelTrees.keys()) {
    const extent = this.topicLevelTrees.get(level)!.extent()!;
    const treeViewWidth = extent[1][0] - extent[0][0];
    const tileNum = Math.pow(2, level);
    const tileSize = treeViewWidth / tileNum;
    const scaledTileWidth =
      Math.max(
        this.xScale(tileSize) - this.xScale(0),
        this.yScale(tileSize) - this.yScale(0)
      ) * this.curZoomTransform.k;

    if (Math.abs(scaledTileWidth - IDEAL_TILE_WIDTH) < bestDistance) {
      bestLevel = level;
      bestDistance = Math.abs(scaledTileWidth - IDEAL_TILE_WIDTH);
    }
  }

  return bestLevel;
}

/**
 * Add direction indicator path
 * @param d Datum
 * @param i Datum index
 * @param g Nodes
 * @param tileScreenWidth Tile width in the screen coordinate
 * @returns This path element
 */
export function addTileIndicatorPath(
  this: Embedding,
  d: DrawnLabel,
  i: number,
  g: SVGPathElement[] | ArrayLike<SVGPathElement>,
  tileScreenWidth: number
) {
  const pathGenerator = d3.arc();
  const pathArgs: d3.DefaultArcObject = {
    innerRadius: 0,
    outerRadius: 3 / this.curZoomTransform.k,
    startAngle: -Math.PI / 2,
    endAngle: Math.PI / 2
  };
  const selection = d3.select(g[i]);

  let tx = this.xScale(d.tileX) + tileScreenWidth / 2;
  let ty = this.yScale(d.tileY);

  switch (d.direction) {
    case Direction.left: {
      pathArgs.startAngle = -Math.PI;
      pathArgs.endAngle = 0;
      tx = this.xScale(d.tileX);
      ty = this.yScale(d.tileY) + tileScreenWidth / 2;
      break;
    }

    case Direction.right: {
      pathArgs.startAngle = 0;
      pathArgs.endAngle = Math.PI;
      tx = this.xScale(d.tileX) + tileScreenWidth;
      ty = this.yScale(d.tileY) + tileScreenWidth / 2;
      break;
    }

    case Direction.bottom: {
      pathArgs.startAngle = Math.PI / 2;
      pathArgs.endAngle = (Math.PI * 3) / 2;
      tx = this.xScale(d.tileX) + tileScreenWidth / 2;
      ty = this.yScale(d.tileY) + tileScreenWidth;
      break;
    }

    default: {
      break;
    }
  }

  return selection
    .attr('d', pathGenerator(pathArgs))
    .attr('transform', `translate(${tx}, ${ty})`);
}

export function labelNumSliderChanged(this: Embedding, e: InputEvent) {
  const newValue = parseInt((e.currentTarget as HTMLInputElement).value);
  this.userMaxLabelNum = newValue;
  this.lastLabelNames = new Map();
  this.layoutTopicLabels(newValue);
}

/**
 * Show a label when mouseover a region
 * @param this Embedding
 * @param x Mouse x coordinate
 * @param y Mouse y coordinate
 */
export function mouseoverLabel(
  this: Embedding,
  x: number | null,
  y: number | null
) {
  if (!this.showGrid) return;
  if (this.hideHighlights) return;

  const bottomGroup = this.topSvg.select('g.top-content g.topics-bottom');
  const labelGroup = this.topSvg.select('g.top-content g.topics');
  const topGroup = this.topSvg.select('g.top-content g.topics-top');

  const oldBottomRect = bottomGroup.select('rect.highlight-tile');
  const oldTopRect = topGroup.select('rect.highlight-tile');

  const hoverDelay = this.showLabel ? 700 : 300;

  const removeHighlight = () => {
    if (labelMouseleaveTimer !== null) {
      clearTimeout(labelMouseleaveTimer);
      labelMouseleaveTimer = null;
    }

    // Clear the highlight and tooltip in a short delay
    labelMouseleaveTimer = setTimeout(() => {
      labelGroup.classed('faded', false);
      oldTopRect.interrupt('top-fade').remove();
      oldBottomRect.remove();
      this.tooltipBottom.classList.add('hidden');
      labelMouseleaveTimer = null;
    }, 50);
  };

  // Remove the tile if x and y are null
  if (x === null || y === null) {
    removeHighlight();
    return;
  }

  // Get the coordinate in the embedding coordinate
  const x0 = this.xScale.invert(this.curZoomTransform.invertX(x));
  const y0 = this.yScale.invert(this.curZoomTransform.invertY(y));

  // Get the corresponding tree
  const idealTreeLevel = this.getIdealTopicTreeLevel()!;
  const tree = this.topicLevelTrees.get(idealTreeLevel)!;
  const treeExtent = tree.extent()!;
  const tileWidth =
    (treeExtent[1][0] - treeExtent[0][0]) / Math.pow(2, idealTreeLevel);
  const tileScreenWidth = Math.abs(this.xScale(tileWidth) - this.xScale(0));

  const radius = Math.sqrt(2) * tileWidth;
  const tile = tree.find(x0, y0, radius);

  // No tile near the mouse location
  if (tile === undefined) {
    if (!oldBottomRect.empty()) {
      removeHighlight();
    }
    return;
  }

  if (oldBottomRect.empty()) {
    // Add a new highlight rect at the bottom layer
    const rect = bottomGroup
      .append('rect')
      .attr('class', 'highlight-tile')
      .attr('x', this.xScale(tile[0]) - tileScreenWidth / 2)
      .attr('y', this.yScale(tile[1]) - tileScreenWidth / 2)
      .attr('width', tileScreenWidth)
      .attr('height', tileScreenWidth)
      .attr('rx', 4 / this.curZoomTransform.k)
      .attr('ry', 4 / this.curZoomTransform.k)
      .style('stroke-width', 2.6 / this.curZoomTransform.k)
      .style(
        'stroke',
        anyTrue(this.showContours)
          ? config.gridColorLight
          : config.gridColorDark
      );

    // Show the tooltip
    updatePopperTooltip(
      this.tooltipBottom,
      rect.node()! as unknown as HTMLElement,
      tile[2],
      'bottom'
    );

    // Insert a clone to the top layer
    const clone = rect.clone(true).style('stroke', null).remove().node()!;
    const topRect = d3.select(
      (topGroup.node() as HTMLElement).appendChild(clone)
    );

    labelMouseenterTimer = tile[2];
    topRect
      .style('opacity', 0)
      .transition('top-fade')
      .duration(hoverDelay)
      .ease(d3.easeCubicInOut)
      .on('end', () => {
        topRect.style('opacity', 1);
        labelGroup.classed('faded', true);
        this.tooltipBottom.classList.remove('hidden');
        labelMouseenterTimer = null;
      });
  } else {
    // Update the old highlight rect
    oldBottomRect
      .attr('x', this.xScale(tile[0]) - tileScreenWidth / 2)
      .attr('y', this.yScale(tile[1]) - tileScreenWidth / 2)
      .attr('width', tileScreenWidth)
      .attr('height', tileScreenWidth)
      .attr('rx', 4 / this.curZoomTransform.k)
      .attr('ry', 4 / this.curZoomTransform.k)
      .style('stroke-width', 2.6 / this.curZoomTransform.k);

    oldTopRect
      .attr('x', this.xScale(tile[0]) - tileScreenWidth / 2)
      .attr('y', this.yScale(tile[1]) - tileScreenWidth / 2)
      .attr('width', tileScreenWidth)
      .attr('height', tileScreenWidth)
      .attr('rx', 4 / this.curZoomTransform.k)
      .attr('ry', 4 / this.curZoomTransform.k)
      .style('stroke-width', 2.6 / this.curZoomTransform.k);

    // Show the tooltip
    updatePopperTooltip(
      this.tooltipBottom,
      oldBottomRect.node()! as unknown as HTMLElement,
      tile[2],
      'bottom'
    );

    if (labelMouseenterTimer === null) {
      this.tooltipBottom.classList.remove('hidden');
    } else {
      labelMouseenterTimer = tile[2];
      oldTopRect
        .interrupt('top-fade')
        .style('opacity', 0)
        .transition('top-fade')
        .duration(hoverDelay)
        .ease(d3.easeCubicInOut)
        .on('end', () => {
          oldTopRect.style('opacity', 1);
          labelGroup.classed('faded', true);
          this.tooltipBottom.classList.remove('hidden');
          labelMouseenterTimer = null;
        });
    }
  }
}

/**
 * Update the popper tooltip for the highlighted prompt point
 * @param tooltip Tooltip element
 * @param anchor Anchor point for the tooltip
 * @param point The prompt point
 */
export const updatePopperTooltip = (
  tooltip: HTMLElement,
  anchor: HTMLElement,
  text: string,
  placement: 'bottom' | 'left' | 'top' | 'right'
) => {
  // Truncate the text if it is too long
  if (text.length > 300) {
    text = text.slice(0, 300);
    text = text.slice(0, text.lastIndexOf(' '));
    text = text.concat('...');
  }

  const arrowElement = tooltip.querySelector('.popper-arrow')! as HTMLElement;
  const contentElement = tooltip.querySelector(
    '.popper-content'
  )! as HTMLElement;
  contentElement.innerText = text;

  computePosition(anchor, tooltip, {
    placement: placement,
    middleware: [offset(6), flip(), shift(), arrow({ element: arrowElement })]
  }).then(({ x, y, placement, middlewareData }) => {
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;

    const { x: arrowX, y: arrowY } = middlewareData.arrow!;
    let staticSide: 'bottom' | 'left' | 'top' | 'right' = 'bottom';
    if (placement.includes('top')) staticSide = 'bottom';
    if (placement.includes('right')) staticSide = 'left';
    if (placement.includes('bottom')) staticSide = 'top';
    if (placement.includes('left')) staticSide = 'right';

    arrowElement.style.left = arrowX ? `${arrowX}px` : '';
    arrowElement.style.top = arrowY ? `${arrowY}px` : '';
    arrowElement.style.right = '';
    arrowElement.style.bottom = '';
    arrowElement.style[staticSide] = '-4px';
  });
};

/**
 * Redraw the topic canvases (clear, transform, draw).
 */
export function redrawTopicGrid(
  this: Embedding,
  strokeColor: string | null = null
) {
  const topicCtxs = this.topicCanvases.map(
    c => (c.node() as HTMLCanvasElement).getContext('2d')!
  );

  for (const topicCtx of topicCtxs) {
    topicCtx.setTransform(1, 0, 0, 1, 0, 0);
    topicCtx.clearRect(0, 0, this.svgFullSize.width, this.svgFullSize.height);
    topicCtx.translate(this.curZoomTransform.x, this.curZoomTransform.y);
    topicCtx.scale(this.curZoomTransform.k, this.curZoomTransform.k);
  }

  this.drawTopicGrid(strokeColor);
}

/**
 * Search all data points in the given bounding box
 * @param quadtree Quadtree
 * @param xMin Min x
 * @param yMin Min y
 * @param xMax Max x
 * @param yMax Max y
 * @returns Array of points in this regin
 */
const search2DQuadTree = (
  quadtree: d3.Quadtree<TopicData>,
  xMin: number,
  yMin: number,
  xMax: number,
  yMax: number
) => {
  const results: TopicData[] = [];
  quadtree.visit((node, x1, y1, x2, y2) => {
    if (!node.length) {
      let leaf: d3.QuadtreeLeaf<TopicData> | undefined =
        node as d3.QuadtreeLeaf<TopicData>;
      do {
        const d = leaf.data;
        if (d[0] >= xMin && d[0] < xMax && d[1] >= yMin && d[1] < yMax) {
          results.push(d);
        }
      } while ((leaf = leaf.next));
    }
    return x1 >= xMax || y1 >= yMax || x2 < xMin || y2 < yMin;
  });
  return results;
};

const anyTrue = (items: boolean[]) => items.reduce((a, b) => a || b);
const allTrue = (items: boolean[]) => items.reduce((a, b) => a && b);
