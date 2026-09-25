// Validates the bundled model catalog. Run: node src/model_catalog.test.mjs
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)))
const catalog = load('./model_catalog.json')
const form = load('./forms/sd_options_adv.json')

// families from docs/backend_protocol.md
const FAMILIES = ['sd15', 'sdxl', 'sd3', 'flux', 'flux2', 'zimage', 'qwenimage']
const TYPES = ['sd_model', 'inpaint_model', 'controlnet']
const FLAGS = ['supports_negative_prompt', 'supports_img2img', 'supports_inpaint', 'supports_controlnet']

assert.deepEqual(Object.keys(catalog.families).sort(), [...FAMILIES].sort())
for (const [family, d] of Object.entries(catalog.families)) {
    for (const k of ['default_steps', 'default_guidance', 'default_size']) assert.equal(typeof d[k], 'number', `${family}.${k}`)
    for (const k of FLAGS) assert.equal(typeof d[k], 'boolean', `${family}.${k}`)
}

const ids = new Set()
for (const m of catalog.models) {
    for (const k of ['id', 'title', 'hf_repo']) assert.ok(typeof m[k] == 'string' && m[k], `${m.id}: ${k}`)
    assert.match(m.hf_repo, /^[\w.-]+\/[\w.-]+$/, m.id)
    assert.ok(!ids.has(m.id), `duplicate id ${m.id}`)
    ids.add(m.id)
    const meta = m.model_meta_data || {}
    assert.ok(FAMILIES.includes(meta.family), `${m.id}: family ${meta.family}`)
    assert.ok(TYPES.includes(meta.type), `${m.id}: type ${meta.type}`)
    assert.ok(m.size_gb > 0 && m.min_ram_gb > 0, `${m.id}: size_gb/min_ram_gb`)
    assert.ok(m.variant === undefined || m.variant == 'fp16', `${m.id}: variant`)
    if (meta.type == 'controlnet') assert.ok(meta.controlnet_mode, `${m.id}: controlnet_mode`)
}

// walk the form for a field
const find = (els, id) => els.reduce((r, el) => r || (el.id == id ? el : find(el.children || [], id)), undefined)

// the default model is a catalog model
const model_el = find(form, 'selected_sd_model')
const default_model = catalog.models.find(m => m.id == model_el.default_value)
assert.ok(default_model && default_model.model_meta_data.type == 'sd_model', `default model ${model_el.default_value}`)

// every ControlNet mode in the UI (plus "Inpaint", used by Generative Fill) has a catalog model
for (const mode of find(form, 'controlnet_model').options.filter(x => x != 'None').concat(['Inpaint']))
    assert.ok(catalog.models.some(m => m.model_meta_data.controlnet_mode == mode), `no ControlNet for ${mode}`)

console.log(`model catalog ok: ${catalog.models.length} models`)
