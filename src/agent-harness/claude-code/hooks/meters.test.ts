import { expect, test } from 'claude-code/testing'

import { difficultyMeter, holdMeter } from './lib/meters'
import { plain } from './kit'

const riskMeter = (value: number, cells?: number) => holdMeter(value, 0.9, cells)
const TIERS = { easyBelow: 0.5, hardFrom: 1.5 }

test('the risk meter fills 24 cells with the hold line at 0.9, before the last two', () => {
  expect(plain(riskMeter(0))).toBe('░░░░░░░░░░░░░░░░░░░░░░┃░░')
  expect(plain(riskMeter(0.5))).toBe('████████████░░░░░░░░░░┃░░')
  expect(plain(riskMeter(0.96))).toBe('██████████████████████┃█░')
  expect(plain(riskMeter(1))).toBe('██████████████████████┃██')
  expect(riskMeter(0.96)).toEqual([
    { text: '██████████████████████', tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: '█', tone: 'fill' },
    { text: '░', tone: 'rest' },
  ])
})

test('the risk meter draws the partial cell in eighths and keeps to its width', () => {
  expect(plain(riskMeter(0.02))).toBe('▌░░░░░░░░░░░░░░░░░░░░░┃░░')
  expect(plain(riskMeter(0.76, 12))).toBe('█████████▏░┃░')
  expect(plain(riskMeter(-3))).toBe(plain(riskMeter(0)))
  expect(plain(riskMeter(9))).toBe(plain(riskMeter(1)))
  expect(plain(riskMeter(Number.NaN))).toBe(plain(riskMeter(0)))
})

test('the difficulty meter fills 24 cells across the rules at 0.5 and 1.5', () => {
  expect(plain(difficultyMeter(0.09, TIERS))).toBe('█▏░░░░│░░░░░░░░░░░░│░░░░░░')
  expect(plain(difficultyMeter(0.74, TIERS))).toBe('██████│██▉░░░░░░░░░│░░░░░░')
  expect(plain(difficultyMeter(1.61, TIERS))).toBe('██████│████████████│█▍░░░░')
  expect(difficultyMeter(1.61, TIERS)).toEqual([
    { text: '██████', tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: '████████████', tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: '█▍', tone: 'fill' },
    { text: '░░░░', tone: 'rest' },
  ])
})

test('the hold line and the tier rules move with the thresholds', () => {
  expect(plain(holdMeter(0.85, 0.8))).toBe('███████████████████┃█▍░░░')
  expect(plain(difficultyMeter(0.09, { easyBelow: 1, hardFrom: 1.75 }))).toBe('█▏░░░░░░░░░░│░░░░░░░░░│░░░')
})
