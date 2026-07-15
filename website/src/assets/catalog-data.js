const DEFAULT_TREE_ENDPOINT =
  'https://huggingface.co/api/datasets/meshllm/catalog/tree/main/entries?recursive=true&expand=false&limit=1000'
const DEFAULT_RAW_ROOT = 'https://huggingface.co/datasets/meshllm/catalog/raw/main/'
const MAX_CATALOG_ENTRIES = 1000
const DEFAULT_CONCURRENCY = 12

function variantEntries(variants) {
  if (Array.isArray(variants)) {
    return variants.map((variant, index) => [variant.curated?.name || `variant-${index + 1}`, variant])
  }
  return Object.entries(variants || {})
}

function layerPackages(variant) {
  return (variant.packages || [])
    .filter((item) => item.type === 'layer-package')
    .sort((left, right) => String(left.repo || '').localeCompare(String(right.repo || '')))
}

export function flattenCatalogEntry(entry, entryPath = '') {
  return variantEntries(entry.variants).map(([variantId, variant]) => {
    const source = variant.source || {}
    const curated = variant.curated || {}
    const packages = layerPackages(variant)
    const primaryPackage = packages[0]
    return {
      schema_version: entry.schema_version || 1,
      entry_path: entryPath,
      variant_id: variantId,
      name: curated.name || variantId,
      size: curated.size || '',
      description: curated.description || '',
      capabilities: Array.isArray(curated.capabilities) ? curated.capabilities.join(' ') : '',
      source_model_repo: entry.source_repo || source.repo || '',
      source_repo: source.repo || entry.source_repo || '',
      source_revision: source.revision || 'main',
      source_file: source.file || '',
      package_count: packages.length,
      package_repos: packages.map((item) => item.repo).filter(Boolean).join(','),
      layer_package_available: packages.length > 0,
      primary_package_repo: primaryPackage?.repo || '',
      primary_package_layer_count: primaryPackage?.layer_count || null,
      primary_package_total_bytes: primaryPackage?.total_bytes || null,
      runtime: packages.length > 0 ? 'multi-machine' : 'single-machine',
      draft_model: curated.draft_model || variant.draft_model || '',
    }
  })
}

function rawEntryUrl(rawRoot, path) {
  return rawRoot + path.split('/').map(encodeURIComponent).join('/')
}

async function fetchJson(fetchImpl, url, label) {
  const response = await fetchImpl(url)
  if (!response.ok) throw new Error(`${label}: HTTP ${response.status}`)
  return response.json()
}

async function mapConcurrent(values, concurrency, callback) {
  const results = new Array(values.length)
  let nextIndex = 0
  async function worker() {
    while (nextIndex < values.length) {
      const index = nextIndex++
      results[index] = await callback(values[index], index)
    }
  }
  const workerCount = Math.min(Math.max(1, concurrency), values.length)
  await Promise.all(Array.from({ length: workerCount }, worker))
  return results
}

export async function loadCanonicalCatalogRows({
  fetchImpl = globalThis.fetch,
  treeEndpoint = DEFAULT_TREE_ENDPOINT,
  rawRoot = DEFAULT_RAW_ROOT,
  concurrency = DEFAULT_CONCURRENCY,
} = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('Catalog fetch is unavailable')
  const tree = await fetchJson(fetchImpl, treeEndpoint, 'Catalog entry listing failed')
  if (!Array.isArray(tree)) throw new Error('Catalog entry listing returned an invalid payload')
  const entryPaths = tree
    .filter((item) => item.type === 'file' && /^entries\/.+\.json$/i.test(item.path || ''))
    .map((item) => item.path)
    .sort()
  if (entryPaths.length === 0) throw new Error('Catalog entry listing contained no entries')
  if (entryPaths.length >= MAX_CATALOG_ENTRIES) {
    throw new Error(`Catalog entry listing reached its ${MAX_CATALOG_ENTRIES}-entry safety limit`)
  }

  const entries = await mapConcurrent(entryPaths, concurrency, async (path) => ({
    path,
    entry: await fetchJson(fetchImpl, rawEntryUrl(rawRoot, path), `Catalog entry ${path} failed`),
  }))
  return entries
    .flatMap(({ path, entry }) => flattenCatalogEntry(entry, path))
    .sort((left, right) => left.name.localeCompare(right.name))
}

export async function loadPublishedCatalogRows({
  fetchImpl = globalThis.fetch,
  endpoint = '/assets/catalog.generated.json',
} = {}) {
  const rows = await fetchJson(fetchImpl, endpoint, 'Published catalog failed')
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error('Published catalog contained no entries')
  }
  return rows
}
