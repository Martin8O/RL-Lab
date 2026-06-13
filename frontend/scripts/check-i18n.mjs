#!/usr/bin/env node
/**
 * i18n completeness checker (F1).
 *
 * Fails (exit 1) when:
 *   1. the two locale files (en, cz) don't have an identical set of keys, or
 *   2. a static `t('some.key')` call in src/ references a key missing from either locale.
 *
 * Dynamic keys — `t(`a.${x}`)` or `t(variable)` — are skipped (can't be resolved
 * statically); they're exercised by the live UI instead.
 *
 * Run: `npm run i18n:check`  (also wired into `.\tasks.ps1 i18n`).
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')
const SRC = join(root, 'src')
const LOCALES = { en: join(SRC, 'i18n', 'en.json'), cz: join(SRC, 'i18n', 'cz.json') }

// ── Flatten a nested locale object into dotted leaf keys ──────────────────────
function flatten(obj, prefix = '', out = new Set()) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, out)
    else out.add(key)
  }
  return out
}

const locales = Object.fromEntries(
  Object.entries(LOCALES).map(([lng, path]) => [lng, flatten(JSON.parse(readFileSync(path, 'utf8')))]),
)

const problems = []

// ── 1. Key parity between en and cz ───────────────────────────────────────────
const [a, b] = Object.keys(locales)
for (const key of locales[a]) if (!locales[b].has(key)) problems.push(`key "${key}" present in ${a} but missing in ${b}`)
for (const key of locales[b]) if (!locales[a].has(key)) problems.push(`key "${key}" present in ${b} but missing in ${a}`)

// ── 2. Every static t('...') key exists in both locales ───────────────────────
function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, files)
    else if (/\.(ts|tsx)$/.test(name)) files.push(p)
  }
  return files
}

// Match t('key'), t("key"), t(`key`) — capturing the first string-literal argument.
const T_CALL = /\bt\(\s*(['"`])([^'"`]+)\1/g
const usedKeys = new Map() // key -> first "file:line" seen

for (const file of walk(SRC)) {
  const text = readFileSync(file, 'utf8')
  let m
  while ((m = T_CALL.exec(text)) !== null) {
    const key = m[2]
    if (key.includes('${')) continue // dynamic (template) key — skip
    if (!usedKeys.has(key)) {
      const line = text.slice(0, m.index).split('\n').length
      usedKeys.set(key, `${relative(root, file)}:${line}`)
    }
  }
}

for (const [key, where] of usedKeys) {
  for (const lng of Object.keys(locales)) {
    if (!locales[lng].has(key)) problems.push(`t("${key}") at ${where} missing from ${lng}`)
  }
}

// ── Report ────────────────────────────────────────────────────────────────────
const keyCount = locales[a].size
if (problems.length > 0) {
  console.error(`✗ i18n check failed (${problems.length} problem(s)):`)
  for (const p of problems) console.error(`  - ${p}`)
  process.exit(1)
}
console.log(`✓ i18n check passed — ${keyCount} keys, ${usedKeys.size} static t() keys, en/cz in parity.`)
