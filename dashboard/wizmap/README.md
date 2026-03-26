# WizMap <a href="https://poloclub.github.io/wizmap/"><img align="right" src="./src/imgs/icon-wizmap.svg" height="38"></img></a>

[![build](https://github.com/poloclub/wizmap/actions/workflows/build.yml/badge.svg)](https://github.com/poloclub/wizmap/actions/workflows/build.yml)
[![npm](https://img.shields.io/npm/v/wizmap?color=red)](https://www.npmjs.com/package/wizmap)
[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/poloclub/wizmap/blob/main/LICENSE)

<!-- [![arxiv badge](https://img.shields.io/badge/arXiv-2209.09227-red)](https://arxiv.org/abs/2209.09227) -->
<!-- [![DOI:10.1145/3491101.3519653](https://img.shields.io/badge/DOI-10.1145/3491101.3519653-blue)](https://doi.org/10.1145/3491101.3519653) -->

Exploring and interpreting large embeddings in your browser!

<table>
  <tr>
    <td colspan="2"><a href="https://poloclub.github.io/timbertrek"><img src='https://i.imgur.com/U5LqUi4.png'></a></td>
  </tr>
  <tr></tr>
  <tr>
    <td><a href="https://poloclub.github.io/wizmap">🚀 Live Demo</a></td>
    <td><a href="https://youtu.be/8fJG87QVceQ">📺 Demo Video</a></td>
  </tr>
</table>


## What is WizMap?

WizMap is a scalable interactive visualization tool to help you easily explore large machine learning embeddings. With a novel multi-resolution embedding summarization method and a familiar map-like interaction design, WizMap allows you to navigate and interpret embedding spaces with ease.

## Web Demo

For a live web demo, visit: <https://poloclub.github.io/wizmap>.

## Get Started

Clone or download this repository:

```bash
git clone git@github.com:poloclub/wizmap.git
```

Install the dependencies:

```bash
npm install
```

Then run WizMap:

```
npm run dev
```

Navigate to localhost:3000. You should see WizMap running in your browser :)

### Use My Dataset

To use WizMap with your dataset, you can check out this [notebook](./example/acl-abstracts.ipynb) to see how to create three JSON files to set up WizMap. These three JSON files contain pre-computed embedding summaries, embedding distributions, and original embedding data in a streamable format.

## Credits

WizMap is created by <a href='https://zijie.wang/' target='_blank'>Jay Wang</a>, <a href='http://fredhohman.com/' target='_blank'>Fred Hohman</a>, and <a href='https://poloclub.github.io/polochau/' target='_blank'>Polo Chau</a>.


## License

The software is available under the [MIT License](https://github.com/poloclub/wizmap/blob/master/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/poloclub/wizmap/issues/new) or contact [Jay Wang](https://zijie.wang).
