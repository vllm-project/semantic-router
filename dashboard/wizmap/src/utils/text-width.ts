// Compute the text width without adding it to DOM!
// https://chrishewett.com/blog/calculating-text-width-programmatically/

import latoTextWidth from './font-width-data/lato.json';

const getTextWidth = (fontJson: object, text: string, fontSize = 16) => {
  // split the string up using the spread operator (to handle UTF-8)
  const letterMapSingle = new Map<string, number>(fontJson.textWidth);
  const letterMapKern = new Map<string, number>(fontJson.kerningWidth);

  const letterSplit = [...text];
  let letterWidth = 0;

  // go through each letter
  for (const [key, letter] of letterSplit.entries()) {
    // add on the width of this letter to the sum
    letterWidth +=
      letterMapSingle.get(letter) || letterMapSingle.get('_median');

    if (key !== letterSplit.length - 1) {
      // add/remove the kerning modifier of this letter and the next one
      letterWidth += letterMapKern.get(`${letter}${letterSplit[key + 1]}`) || 0;
    }
  }

  // now you have your string width for font-size: 100px letters
  // divide by the ratio between 100 and your font-size
  return letterWidth / (100 / fontSize);
};

/**
 * Get the text width in px
 * @param text Text string
 * @param fontSize Font size
 * @returns Text width in px
 */
export const getLatoTextWidth = (text: string, fontSize = 16) => {
  return getTextWidth(latoTextWidth, text, fontSize);
};
