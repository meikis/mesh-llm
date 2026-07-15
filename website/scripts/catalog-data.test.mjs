import assert from 'node:assert/strict'
import test from 'node:test'
import {
  flattenCatalogEntry,
  loadCanonicalCatalogRows,
  loadPublishedCatalogRows,
} from '../src/assets/catalog-data.js'

const qwenEntry = {
  schema_version: 1,
  source_repo: 'unsloth/Qwen3.6-27B-MTP-GGUF',
  variants: {
    'Qwen3.6-27B-UD-Q4_K_XL': {
      source: {
        repo: 'unsloth/Qwen3.6-27B-MTP-GGUF',
        file: 'Qwen3.6-27B-UD-Q4_K_XL.gguf',
        revision: 'main',
      },
      curated: {
        name: 'Qwen3.6-27B-UD-Q4_K_XL',
        size: '65 layers',
        description: 'Layer package for Qwen3.6',
      },
      packages: [
        {
          type: 'layer-package',
          repo: 'meshllm/Qwen3.6-27B-UD-Q4_K_XL-layers',
          layer_count: 65,
        },
      ],
    },
  },
}

test('flattens the canonical Qwen source-to-package mapping', () => {
  const [row] = flattenCatalogEntry(qwenEntry, 'entries/unsloth/Qwen3.6-27B-MTP-GGUF.json')
  assert.equal(row.source_repo, 'unsloth/Qwen3.6-27B-MTP-GGUF')
  assert.equal(row.source_file, 'Qwen3.6-27B-UD-Q4_K_XL.gguf')
  assert.equal(row.layer_package_available, true)
  assert.equal(row.primary_package_repo, 'meshllm/Qwen3.6-27B-UD-Q4_K_XL-layers')
  assert.equal(row.primary_package_layer_count, 65)
})

test('loads entry files from the canonical tree instead of flattened catalog rows', async () => {
  const responses = new Map([
    ['tree', [
      { type: 'file', path: 'entries/unsloth/Qwen3.6-27B-MTP-GGUF.json' },
      { type: 'file', path: 'catalog_rows.jsonl' },
    ]],
    ['raw/entries/unsloth/Qwen3.6-27B-MTP-GGUF.json', qwenEntry],
  ])
  const fetchImpl = async (url) => {
    const value = responses.get(url)
    return { ok: value !== undefined, status: value === undefined ? 404 : 200, json: async () => value }
  }
  const rows = await loadCanonicalCatalogRows({
    fetchImpl,
    treeEndpoint: 'tree',
    rawRoot: 'raw/',
    concurrency: 2,
  })
  assert.equal(rows.length, 1)
  assert.equal(rows[0].name, 'Qwen3.6-27B-UD-Q4_K_XL')
})

test('loads the generated same-origin catalog in the browser', async () => {
  const fetchImpl = async (url) => ({
    ok: url === '/assets/catalog.generated.json',
    status: 200,
    json: async () => flattenCatalogEntry(qwenEntry),
  })
  const rows = await loadPublishedCatalogRows({ fetchImpl })
  assert.equal(rows[0].primary_package_repo, 'meshllm/Qwen3.6-27B-UD-Q4_K_XL-layers')
})
