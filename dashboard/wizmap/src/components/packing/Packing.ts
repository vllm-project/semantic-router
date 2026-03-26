import d3 from '../../utils/d3-import';
import type { Size, Padding, Point, Rect } from '../../types/common-types';
import type {
  PhraseTreeData,
  PhraseTextInfo,
  PhraseTextLineInfo
} from '../../types/packing-types';
import { timeit, round, yieldToMain, rectsIntersect } from '../../utils/utils';

import { getLatoTextWidth } from '../../utils/text-width';
import type { Writable } from 'svelte/store';
import type { TooltipStoreValue } from '../../stores';
import { getTooltipStoreDefaultValue } from '../../stores';
import { config } from '../../config/config';

const DEBUG = config.debug;
const FONT_SIZE = 12;
const HALO_WIDTH = 4;
const GRACE_PADDING = 3;

let circleMouseenterTimer: number | null = null;
let circleMouseleaveTimer: number | null = null;

interface NodeRect extends Rect {
  node: d3.HierarchyCircularNode<PhraseTreeData>;
}

/**
 * Class for the circle packing view
 */

export class Packer {
  svg: d3.Selection<HTMLElement, unknown, null, undefined>;
  /** The size of the BBox of the SVG element */
  svgFullSize: Size;
  /** The size of the drawing space of the SVG element */
  svgSize: Size;
  svgPadding: Padding;
  svgBBox: DOMRect;

  xScale: d3.ScaleLinear<number, number, never>;
  yScale: d3.ScaleLinear<number, number, never>;
  component: HTMLElement;
  updatePacker: () => void;

  // Circle packing
  pack: d3.HierarchyCircularNode<PhraseTreeData> | null = null;

  // Zooming
  focusNode: d3.HierarchyCircularNode<PhraseTreeData> | null = null;
  view: d3.ZoomView | null = null;
  baseView: d3.ZoomView | null = null;
  circleGroups: d3.Selection<
    SVGGElement,
    d3.HierarchyCircularNode<PhraseTreeData>,
    SVGGElement,
    unknown
  > | null = null;
  topTextGroups: d3.Selection<
    SVGGElement,
    d3.HierarchyCircularNode<PhraseTreeData>,
    SVGGElement,
    unknown
  > | null = null;

  // Mouse over
  hoverNode: d3.HierarchyCircularNode<PhraseTreeData> | null = null;

  // Stores
  tooltipStore: Writable<TooltipStoreValue>;
  tooltipStoreValue: TooltipStoreValue = getTooltipStoreDefaultValue();

  /**
   *
   * @param args Named parameters
   * @param args.component The component
   */
  constructor({
    component,
    tooltipStore,
    updatePacker
  }: {
    component: HTMLElement;
    tooltipStore: Writable<TooltipStoreValue>;
    updatePacker: () => void;
  }) {
    this.component = component;
    this.updatePacker = updatePacker;

    // Initialize the SVG
    this.svg = d3.select(this.component).select('svg.packing-svg');
    this.svgFullSize = { width: 0, height: 0 };
    this.svgBBox = this.svg.node()!.getBoundingClientRect();
    this.svgFullSize.width = this.svgBBox.width;
    this.svgFullSize.height = this.svgBBox.height;

    this.svgPadding = {
      top: 5,
      bottom: 5,
      left: 5,
      right: 5
    };
    this.svgSize = {
      width:
        this.svgFullSize.width - this.svgPadding.left - this.svgPadding.right,
      height:
        this.svgFullSize.width - this.svgPadding.top - this.svgPadding.bottom
    };

    // Initialize SVG layers
    const content = this.svg
      .append('g')
      .attr('class', 'content')
      .attr(
        'transform',
        `translate(${this.svgPadding.left}, ${this.svgPadding.top})`
      );
    content
      .append('rect')
      .attr('class', 'back-rect')
      .attr('width', this.svgSize.width)
      .attr('height', this.svgSize.height)
      .on('click', (e: MouseEvent) => {
        if (this.pack) {
          this.circleClickHandler(e, this.pack);
        }
      });

    // Subscribe the store
    this.tooltipStore = tooltipStore;
    this.tooltipStore.subscribe(value => {
      this.tooltipStoreValue = value;
    });

    // d3.pack() uses [0, 1] ranges by default
    this.xScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, this.svgSize.width]);

    this.yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([this.svgSize.height, 0]);

    this.initData().then(() => {
      // Draw the circle packing after loading the data
      timeit('Draw circle packing', DEBUG);
      this.drawCirclePacking();
      timeit('Draw circle packing', DEBUG);
    });
  }

  /**
   * Load the data
   */
  initData = async () => {
    const jsonURL = `${import.meta.env.BASE_URL}data/phrases-tree.json`;
    const phraseData = (await d3.json(jsonURL)) as PhraseTreeData;

    const root = d3
      .hierarchy(phraseData, d => d.c)
      .sum(d => d.v)
      .sort((a, b) => b.data.v - a.data.v);

    this.pack = d3
      .pack<PhraseTreeData>()
      .padding(3)
      .size([this.svgSize.width, this.svgSize.height])(root);
  };

  /**
   * Draw the circle packing
   */
  drawCirclePacking = () => {
    if (this.pack === null) return;
    const content = this.svg.select('g.content');
    const circleContent = content.append('g').attr('class', 'content-circle');

    for (let h = 1; h < this.pack.height + 2; h++) {
      content.append('g').attr('class', `content-text-${h}`);
    }

    // Initialize the zoom
    this.focusNode = this.pack;
    this.view = [this.pack.x, this.pack.y, this.pack.r * 2];
    this.baseView = [this.pack.x, this.pack.y, this.pack.r * 2];

    // Visible all children and compute the text size information
    const nodes = this.pack.descendants().slice(1);
    let nodeID = 1;
    for (const node of nodes) {
      node.data.id = nodeID++;
      node.data.textInfo = processName(node.data.n);
    }
    this.circleGroups = circleContent
      .selectAll<SVGGElement, d3.HierarchyCircularNode<PhraseTreeData>>(
        'g.circle-group'
      )
      .data(nodes)
      .join('g')
      .attr('class', d => `circle-group circle-group-${d.depth}`)
      .attr('id', d => `circle-group-${d.data.id!}`)
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .classed('no-pointer', d => d.r < 10 && d.children === undefined)
      .style('font-size', `${FONT_SIZE}px`);

    this.circleGroups
      .append('circle')
      .attr('class', 'phrase-circle')
      .attr('cx', 0)
      .attr('cy', 0)
      .attr('r', d => d.r)
      .on('mouseenter', (e, d) =>
        this.circleMouseenterHandler(e as MouseEvent, d)
      )
      .on('mouseleave', e => this.circleMouseleaveHandler(e as MouseEvent))
      .on('click', (e, d) => this.circleClickHandler(e as MouseEvent, d));

    // Draw the text
    this.circleGroups
      .append('text')
      .attr('class', d => `phrase-label phrase-label-${d.depth}`)
      .each((d, i, g) =>
        drawLabelInCircle({
          d,
          i,
          g,
          hideParent: true,
          checkHidden: true,
          showHalo: false,
          scale: 1,
          markVisible: true
        })
      );

    // Draw the text of first level circles on top of all circles
    // Find all first level nodes without any text drawn and have enough space
    // to show a text label
    const firstLevelNodes = nodes.filter(d => d.depth === 1);
    const topLabelNodes = [];
    for (const node of firstLevelNodes) {
      // Check if we have drawn label for this node or its descendants
      let hasDrawnLabel = false;
      for (const child of node.descendants()) {
        if (child.data.textInfo!.visible) {
          hasDrawnLabel = true;
          break;
        }
      }
      if (hasDrawnLabel) continue;

      // Check if the circle is large enough to hold the first-level label
      const minDiagonal = Math.min(
        node.data.textInfo!.infos[0].diagonal,
        node.data.textInfo!.infos[1].diagonal
      );
      if (minDiagonal < 2 * node.r) {
        topLabelNodes.push(node);
      }
    }

    const textContent1 = this.svg.select<SVGGElement>('g.content-text-1');
    this.topTextGroups = textContent1
      .selectAll<SVGGElement, d3.HierarchyCircularNode<PhraseTreeData>>(
        'g.top-text-group'
      )
      .data(topLabelNodes)
      .join('g')
      .attr('class', 'top-text-group')
      .attr('transform', d => `translate(${d.x - d.r}, ${d.y - d.r})`)
      .style('font-size', `${FONT_SIZE}px`);

    // Draw the text
    this.topTextGroups
      .append('text')
      .attr('class', 'phrase-label')
      .attr('transform', d => `translate(${d.r}, ${d.r})`)
      .each((d, i, g) =>
        drawLabelInCircle({
          d,
          i,
          g,
          hideParent: false,
          checkHidden: true,
          showHalo: true,
          scale: 1,
          markVisible: false
        })
      );

    this.zoomToView(this.view);
  };

  /**
   * Mouse enter handler
   * @param e Mouse event
   * @param d Datum
   */
  circleMouseenterHandler = (
    e: MouseEvent,
    d: d3.HierarchyCircularNode<PhraseTreeData>
  ) => {
    if (this.circleGroups === null) return;

    // Ignore hovering that is on the focus node's ancestors
    if (this.focusNode && this.focusNode.ancestors().includes(d)) return;

    const element = d3.select(e.target as HTMLElement);
    element.classed('hovered', true);

    if (circleMouseleaveTimer !== null) {
      clearTimeout(circleMouseleaveTimer);
      circleMouseleaveTimer = null;
    }

    // Get the tooltip position
    const circleGroup = this.svg.select(`#circle-group-${d.data.id!}`);

    const position = (
      circleGroup.node()! as HTMLElement
    ).getBoundingClientRect();
    const curWidth = position.width;
    const tooltipCenterX = position.x + curWidth / 2;
    const tooltipCenterY = Math.max(this.svgBBox.y, position.y);

    this.tooltipStoreValue.html = this.getTooltipMessage(d);
    this.tooltipStoreValue.x = tooltipCenterX;
    this.tooltipStoreValue.y = tooltipCenterY;
    this.tooltipStoreValue.show = true;

    if (this.hoverNode === null) {
      if (circleMouseenterTimer) clearTimeout(circleMouseenterTimer);
      circleMouseenterTimer = setTimeout(() => {
        this.hoverNode = d;
        this.tooltipStore.set(this.tooltipStoreValue);
        circleMouseenterTimer = null;
      }, 500);
    } else {
      if (circleMouseenterTimer) clearTimeout(circleMouseenterTimer);
      circleMouseenterTimer = setTimeout(() => {
        this.hoverNode = d;
        this.tooltipStore.set(this.tooltipStoreValue);
        circleMouseenterTimer = null;
      }, 200);
    }
  };

  /**
   * Mouse leave handler
   * @param e Mouse event
   * @param d Datum
   */
  circleMouseleaveHandler = (e: MouseEvent) => {
    const element = d3.select(e.target as HTMLElement);
    element.classed('hovered', false);

    if (circleMouseenterTimer !== null) {
      clearTimeout(circleMouseenterTimer);
      circleMouseenterTimer = null;
    }

    if (this.hoverNode !== null) {
      if (circleMouseleaveTimer !== null) {
        clearTimeout(circleMouseleaveTimer);
        circleMouseleaveTimer = null;
      }

      circleMouseleaveTimer = setTimeout(() => {
        this.hoverNode = null;
        this.tooltipStoreValue.show = false;
        this.tooltipStore.set(this.tooltipStoreValue);
      }, 50);
    }
  };

  /**
   * Compose a tooltip message for the hovered node
   * @param d Hovered node
   */
  getTooltipMessage = (d: d3.HierarchyCircularNode<PhraseTreeData>) => {
    let parentName = '';
    let label = `<span class="label">"${d.data.n}"</span>`;
    if (d.depth > 1) {
      parentName = d.parent!.data.n;
      const occurrenceIndex = d.data.n.match(
        new RegExp(parentName, 'i')
      )?.index;

      if (occurrenceIndex !== undefined) {
        if (occurrenceIndex == 0) {
          label = `<span class="parent-label">${parentName}</span>`;
          label += `<span class="label">${d.data.n.slice(
            parentName.length
          )}</span>`;
        } else {
          label = `<span class="label">${d.data.n.slice(
            0,
            occurrenceIndex
          )}</span>`;
          label += `<span class="parent-label">${parentName}</span>`;
          label += `<span class="label">${d.data.n.slice(
            occurrenceIndex + parentName.length
          )}</span>`;
        }
      }
    }

    const message = `
    <div class='tooltip-content packing-tooltip'>
      <div class="top">
        ${label}
      </div>

      <div class="splitter"></div>

      <div class="bottom">
        Count: ${d.data.v}
      </div>

    </div>
    `;
    return message;
  };

  /**
   * Apply zoom at one frame
   * @param view [center x, center y, view width]
   */
  zoomToView = (view: d3.ZoomView) => {
    this.view = view;

    const scale = this.svgSize.width / view[2];
    const x0 = view[0] - view[2] / 2;
    const y0 = view[1] - view[2] / 2;

    this.circleGroups
      ?.attr(
        'transform',
        d => `translate(${(d.x - x0) * scale},${(d.y - y0) * scale})`
      )
      .style('font-size', `${FONT_SIZE * scale}px`)
      .select('.phrase-circle')
      .attr('r', d => d.r * scale);
  };

  /**
   * Mouse leave handler
   * @param e Mouse event
   * @param d Datum
   */
  circleClickHandler = (
    e: MouseEvent,
    d: d3.HierarchyCircularNode<PhraseTreeData>
  ) => {
    if (this.focusNode === null || this.view === null) return;
    if (d === this.focusNode) return;
    e.stopPropagation();

    this.resetZoomInteractions();

    const textContent1 = this.svg.select<SVGGElement>('g.content-text-1');

    // Start zooming
    const previousFocusNode = this.focusNode;
    this.focusNode = d;
    const x0 = this.focusNode!.x - this.focusNode!.r;
    const y0 = this.focusNode!.y - this.focusNode!.r;
    const scale = this.svgSize.width / (this.focusNode!.r * 2);

    const trans = this.svg
      .transition('zoom')
      .duration(800)
      .tween('zoom', () => {
        const interpolate = d3.interpolateZoom(this.view!, [
          this.focusNode!.x,
          this.focusNode!.y,
          this.focusNode!.r * 2
        ]);

        return (t: number) => this.zoomToView(interpolate(t));
      }) as unknown as d3.Transition<d3.BaseType, unknown, null, undefined>;

    if (this.focusNode !== this.pack) {
      if (!this.circleGroups) return;

      textContent1
        .transition('label-removal')
        .duration(150)
        .style('opacity', 0);

      const lastTextContent = this.svg.select<SVGGElement>(
        `g.content-text-${previousFocusNode.depth + 1}`
      );
      const curTextContent = this.svg.select<SVGGElement>(
        `g.content-text-${this.focusNode.depth + 1}`
      );

      lastTextContent
        .transition('label-removal')
        .duration(150)
        .style('opacity', 0);

      curTextContent.style('opacity', 0).selectAll('*').remove();

      // Allow users to interact with all descendants
      this.circleGroups
        .filter(d => {
          return d.ancestors().slice(0, 3).includes(this.focusNode!);
        })
        .style('--base-stroke', `${scale}px`)
        .classed('no-pointer', false);

      // Disallow users to interact with the current node
      this.circleGroups
        .filter(d => d == this.focusNode)
        .classed('no-pointer', true);

      // If a node is focused, show all descendants' texts
      const topLabelNodes: d3.HierarchyCircularNode<PhraseTreeData>[] = [];
      const drawnRects: NodeRect[] = [];

      // Check if we have drawn label for this node or its descendants
      // Prioritize drawing children in the deeper levels
      let curDepth = -1;
      const allDescendants: d3.HierarchyCircularNode<PhraseTreeData>[][] = [];
      for (const descendant of this.focusNode.descendants().slice(1)) {
        if (descendant.depth !== curDepth) {
          allDescendants.unshift([]);
          curDepth = descendant.depth;
        }
        allDescendants[0].push(descendant);
      }
      allDescendants.forEach(ds => d3.shuffle(ds));
      let descendants =
        allDescendants.length === 0
          ? []
          : allDescendants.reduce((a, b) => a.concat(b));

      if (descendants.length === 0) {
        descendants = this.focusNode.descendants();
      }

      for (const child of descendants) {
        if (!child.data.textInfo!.visible) {
          const curRect: NodeRect = {
            x: (child.x - x0) * scale - child.data.textInfo!.infos[1].width / 2,
            y:
              (child.y - y0) * scale - child.data.textInfo!.infos[1].height / 2,
            width: child.data.textInfo!.infos[1].width,
            height: child.data.textInfo!.infos[1].height,
            node: child
          };

          // Check if this label is much taller or wider than the back circle
          if (curRect.height > child.r * scale * 2 - 5) {
            continue;
          }

          if (curRect.width > child.r * scale * 3) {
            continue;
          }

          // Check if this label would interact with other labels
          // Also check if we are drawing a label where its children's label has
          // already been drawn (ignore the parent's label in this case)
          let intersect = false;
          for (const drawnRect of drawnRects) {
            if (rectsIntersect(drawnRect, curRect)) {
              intersect = true;
              break;
            }

            if (child === drawnRect.node.parent) {
              intersect = true;
              break;
            }
          }

          if (!intersect) {
            topLabelNodes.push(child);
            drawnRects.push(curRect);
          }
        }
      }

      // If we are focusing on one node without children, we can increase the
      // label font size
      let curFontSize = FONT_SIZE;
      if (descendants.length === 1) {
        let size = 5;
        while (size < 150) {
          if (descendants[0].data.textInfo?.infos[1].lines.length == 1) {
            // This word has only one line
            const width = getLatoTextWidth(
              descendants[0].data.textInfo!.infos[0].lines[0],
              size
            );
            const height = size;
            const diagonal = Math.sqrt(width ** 2 + height ** 2);
            if (diagonal > descendants[0].r * 2 * scale) {
              size -= 10;
              break;
            }

            size += 5;
          } else {
            // Use two line
            const width = Math.max(
              getLatoTextWidth(
                descendants[0].data.textInfo!.infos[1].lines[0],
                size
              ),
              getLatoTextWidth(
                descendants[0].data.textInfo!.infos[1].lines[1],
                size
              )
            );
            const height = 2 * size;
            const diagonal = Math.sqrt(width ** 2 + height ** 2);
            if (diagonal > descendants[0].r * 2 * scale) {
              size -= 10;
              break;
            }

            size += 5;
          }
        }
        curFontSize = size;
      }

      const localLabels = curTextContent
        .selectAll<SVGGElement, d3.HierarchyCircularNode<PhraseTreeData>>(
          'g.top-text-group'
        )
        .data(topLabelNodes)
        .join('g')
        .attr('class', 'top-text-group')
        .attr(
          'transform',
          d => `translate(${(d.x - x0) * scale}, ${(d.y - y0) * scale})`
        )
        .style(
          'font-size',
          `${descendants.length === 1 ? curFontSize : FONT_SIZE}px`
        );

      // Draw the text
      localLabels
        .append('text')
        .attr('class', 'phrase-label')
        .classed('phrase-label-2', d => descendants.length === 1 && d.depth > 1)
        .each((d, i, g) =>
          drawLabelInCircle({
            d,
            i,
            g,
            hideParent: false,
            checkHidden: false,
            showHalo: descendants.length > 1,
            scale,
            markVisible: false
          })
        );

      trans.on('end', () => {
        curTextContent
          .transition('show-top-label')
          .duration(200)
          .style('opacity', 1);
      });
    } else {
      const lastTextContent = this.svg.select<SVGGElement>(
        `g.content-text-${previousFocusNode.depth + 1}`
      );
      lastTextContent
        .transition('label-removal')
        .duration(150)
        .style('opacity', 0)
        .on('end', () => {
          lastTextContent.selectAll('*').remove();
        });

      trans.on('end', () => {
        textContent1.transition('label-show').duration(150).style('opacity', 1);
      });
    }
  };

  /**
   * Reset all zoom-related configurations for drawn elements
   */
  resetZoomInteractions = () => {
    if (this.circleGroups === null) return;
    // Reset stroke base values
    this.circleGroups
      .style('--base-stroke', '1px')
      .classed('no-pointer', d => d.r < 10 && d.children === undefined)
      .style('font-size', `${FONT_SIZE}px`);
  };
}

/**
 * Get the size info about a phrase text
 * @param name The phrase text
 * @returns Size info about this text
 */
const processName = (name: string) => {
  const words = name.split(' ');
  const lineInfo1: PhraseTextLineInfo = {
    width: getLatoTextWidth(name, FONT_SIZE),
    height: FONT_SIZE,
    diagonal: Math.sqrt(
      getLatoTextWidth(name, FONT_SIZE) ** 2 + FONT_SIZE ** 2
    ),
    lines: [name]
  };

  const lineInfo2: PhraseTextLineInfo = {
    width: 0,
    height: 0,
    diagonal: 0,
    lines: [name]
  };

  if (words.length == 1) {
    lineInfo2.width = getLatoTextWidth(name, FONT_SIZE);
    lineInfo2.height = FONT_SIZE;
  } else {
    // Split the name into two lines with the same number of words
    const line1 = words.slice(0, Math.floor(words.length / 2)).join(' ');
    const line2 = words.slice(Math.floor(words.length / 2)).join(' ');

    lineInfo2.lines = [line1, line2];
    lineInfo2.width = Math.max(
      getLatoTextWidth(line1, FONT_SIZE),
      getLatoTextWidth(line2, FONT_SIZE)
    );
    lineInfo2.height = FONT_SIZE * 2;
  }

  lineInfo2.diagonal = Math.sqrt(lineInfo2.width ** 2 + lineInfo2.height ** 2);

  const result: PhraseTextInfo = {
    visible: false,
    infos: [lineInfo1, lineInfo2]
  };

  return result;
};

/**
 * Return true if the circle should not display its text
 * @param d Node data
 * @param hideParent True if hide text of nodes with children nodes
 * @param scale Current zoom scale for the circle's radius
 * @returns True if the circle should not display its text
 */
const shouldHideText = (
  d: d3.HierarchyCircularNode<PhraseTreeData>,
  hideParent: boolean,
  scale: number
) => {
  if (hideParent && d.children !== undefined) return true;

  return d.data.textInfo
    ? Math.min(
        d.data.textInfo.infos[0].diagonal,
        d.data.textInfo.infos[1].diagonal
      ) >
        2 * d.r * scale + GRACE_PADDING
    : true;
};

const drawLabelInCircle = ({
  d,
  i,
  g,
  hideParent,
  checkHidden,
  showHalo,
  scale,
  markVisible
}: {
  d: d3.HierarchyCircularNode<PhraseTreeData>;
  i: number;
  g: SVGTextElement[] | ArrayLike<SVGTextElement>;
  hideParent: boolean;
  checkHidden: boolean;
  showHalo: boolean;
  scale: number;
  markVisible: boolean;
}) => {
  if (d.data.textInfo === undefined) return;
  if (checkHidden && shouldHideText(d, hideParent, scale)) return;

  if (markVisible) d.data.textInfo.visible = true;
  const element = d3.select(g[i]);

  // Prioritize fitting the text in one line
  if (
    (checkHidden &&
      d.data.textInfo.infos[0].diagonal < 2 * d.r * scale + GRACE_PADDING) ||
    (!checkHidden && d.data.textInfo.infos[1].lines.length == 1)
  ) {
    // One line
    const line = element
      .append('tspan')
      .attr('class', 'line-1')
      .attr('x', 0)
      .attr('y', 0)
      .text(d.data.textInfo.infos[0].lines[0]);

    if (showHalo) {
      line
        .attr('paint-order', 'stroke')
        .attr('stroke', 'white')
        .attr('stroke-width', HALO_WIDTH);
    }
  } else {
    // Two lines
    const line1 = element
      .append('tspan')
      .attr('class', 'line-1')
      .attr('x', 0)
      .attr('y', 0)
      .attr('dy', '-0.5em')
      .text(d.data.textInfo.infos[1].lines[0]);

    const line2 = element
      .append('tspan')
      .attr('class', 'line-2')
      .attr('x', 0)
      .attr('y', 0)
      .attr('dy', '0.5em')
      .text(d.data.textInfo.infos[1].lines[1]!);

    if (showHalo) {
      for (const line of [line1, line2]) {
        line
          .attr('paint-order', 'stroke')
          .attr('stroke', 'white')
          .attr('stroke-width', HALO_WIDTH);
      }
    }
  }
};
