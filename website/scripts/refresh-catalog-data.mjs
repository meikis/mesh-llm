import { writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { loadCanonicalCatalogRows } from '../src/assets/catalog-data.js'

const output = fileURLToPath(new URL('../src/assets/catalog.generated.json', import.meta.url))
const requireFresh = process.env.MESH_CATALOG_REQUIRE_FRESH === '1'

try {
  const rows = await loadCanonicalCatalogRows()
  await writeFile(output, `${JSON.stringify(rows, null, 2)}\n`)
  console.log(`Refreshed canonical model catalog: ${rows.length} variants`)
} catch (error) {
  if (requireFresh) throw error
  console.warn(`Catalog refresh unavailable; using checked-in snapshot: ${error.message}`)
}
