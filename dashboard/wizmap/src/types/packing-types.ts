/**
 * Types for the packing view.
 */

export interface PhraseTreeData {
  /**
   * Name of this phrase
   */
  n: string;

  /**
   * Count of this phrase across all documents
   */
  v: number;

  /**
   * Type of this phrase
   */
  l: string;

  /**
   * Other phrases including this phrase
   */
  c?: PhraseTreeData[];

  /**
   * Unique identifier for this data record
   */
  id?: number;

  /**
   * Size information about the rendered phrase text
   */
  textInfo?: PhraseTextInfo;
}

export interface PhraseTextInfo {
  /**
   * True if the text is visible for this node.
   */
  visible: boolean;
  infos: [PhraseTextLineInfo, PhraseTextLineInfo];
}

export interface PhraseTextLineInfo {
  /**
   * The width of the rendered phrase text
   */
  width: number;

  /**
   * The height of the rendered phrase text
   */
  height: number;

  /**
   * Length of the diagonal of the text box
   */
  diagonal: number;

  /**
   * One or two lines of this phrase
   */
  lines: [string] | [string, string];
}
