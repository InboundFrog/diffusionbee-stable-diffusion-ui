<template>
    <div>
    </div>
</template>
<script>

import Vue from 'vue'
import model_catalog from './model_catalog.json'

const FAMILIES = model_catalog.families

// catalog entries with the family defaults merged into model_meta_data
const CATALOG = model_catalog.models.map(m => ({
    ...m,
    model_meta_data: { ...FAMILIES[m.model_meta_data.family], ...m.model_meta_data },
}))

// runs a download over the `to_download` ipc channel. send_args = [channel, ...args], the download id is appended
function ipc_download(send_args, onProgress, onSuccess, onError) {
    const downloadId = Date.now().toString() + Math.random().toString().substr(2);
    window.bind_ipc_download_on(downloadId, onProgress, function(m){
        window.unbind_ipc_download_on(downloadId)
        onSuccess(m)
    }, function(m){
        window.unbind_ipc_download_on(downloadId)
        onError(m)
    } )
    window.ipcRenderer.send(...send_args, downloadId);
}

// v2 (TensorFlow) assets point at .tdict files the diffusers backend can't load
function drop_tdict_assets(assets){
    let n = 0
    for(let k of Object.keys(assets)){
        if(String((assets[k] || {}).asset_path || "").endsWith(".tdict")){
            delete assets[k]
            n += 1
        }
    }
    return n
}


export default {
    name: 'AssetsManager.vue',
    props: {},
    components: {},
    mounted() {
        // catalog models already in the HF cache (fetched by `hf download` or another app) count as downloaded,
        // except ones the user removed here
        let hidden = window.ipcRenderer.sendSync('load_data', 'hidden_hf_models.json')
        let todo = CATALOG.filter(m => m.hf_repo && !this.downloaded_assets[m.id] && !hidden[m.id])
        if(todo.length)
            window.ipcRenderer.invoke('find_cached_hf_models', todo.map(m => m.hf_repo + ":" + (m.variant || ""))).then(found => {
                for(let m of todo)
                    if(found[m.hf_repo] && !this.downloaded_assets[m.id])
                        Vue.set(this.downloaded_assets, m.id, {...JSON.parse(JSON.stringify(m)), asset_path: found[m.hf_repo], status: 'done'})
            })

        if(this.n_legacy_dropped > 0){
            window.ipcRenderer.sendSync('save_data', this.downloaded_assets , 'downloaded_assets.json');
            window.ipcRenderer.sendSync('save_data', this.local_assets , 'locally_loaded_assets.json');
            // show once the splash screen is gone. The entries are gone from disk now, so this only ever shows once.
            let unwatch = this.$watch(() => this.$parent.app_state.is_start_screen, (is_start) => {
                if(is_start) return
                Vue.$toast.default("Models from the previous DiffusionBee version (.tdict) are no longer supported and were removed. Re-download them from the Models page, or re-import the original .safetensors file.", {duration: 20000})
                unwatch()
            })
        }
    },
    data() {
        let downloaded_assets_storage = window.ipcRenderer.sendSync('load_data' , 'downloaded_assets.json'); // get from local storage
        let local_assets_storage = window.ipcRenderer.sendSync('load_data' , 'locally_loaded_assets.json'); // get from local storage

        return {
            downloaded_assets: downloaded_assets_storage ,
            local_assets: local_assets_storage ,
            n_legacy_dropped: drop_tdict_assets(downloaded_assets_storage) + drop_tdict_assets(local_assets_storage),
            downloading: {} , // id , status : done/downloading/error/not_downloaded , progress , hash,
            catalog: CATALOG,
            families: FAMILIES,
        };
    },

    watch:{
         'downloaded_assets': {
            handler: function(new_value) {
                window.ipcRenderer.sendSync('save_data', new_value , 'downloaded_assets.json');
            },
            deep: true
        } ,

         'local_assets': {
            handler: function(new_value) {
                window.ipcRenderer.sendSync('save_data', new_value , 'locally_loaded_assets.json');
            },
            deep: true
        } ,

    },

    computed: {
        all_avail_assets(){
            return { // update the dict
              ...this.downloaded_assets ,
              ...this.local_assets
            }
        }
    },

    methods: {

        catalog_entry(asset_id){
            return CATALOG.find(x => x.id == asset_id)
        },

        // the ControlNet for a UI mode ("Depth", "Inpaint", ...) that works with a model family
        controlnet_entry(mode, family){
            return CATALOG.find(x => x.model_meta_data.type == 'controlnet' && x.model_meta_data.controlnet_mode == mode && x.model_meta_data.family == family)
        },

        // model_meta_data of a catalog or downloaded/imported asset. MLX transformer files, and families the backend
        // runs on its MLX engine on this Mac, can't inpaint
        model_meta(asset_id){
            let asset = this.catalog_entry(asset_id) || this.all_avail_assets[asset_id]
            let meta = asset && asset.model_meta_data
            if(meta && (meta.mlx || ((this.$parent.$refs.stable_diffusion || {}).mlx_families || []).includes(meta.family)))
                return {...meta, supports_inpaint: false}
            return meta
        },

        // info = inspect_model output {family, is_inpaint, type}
        add_local_asset(asset_path, asset_id, info){
            let type = (info.type == 'sd_model' && info.is_inpaint) ? 'inpaint_model' : info.type
            if(type != 'lora' && !FAMILIES[info.family])
                return "Unsupported model type (" + (info.family || "unknown") + ")"

            Vue.set(this.local_assets, asset_id, {
                id: asset_id,
                title: asset_id,
                asset_path: asset_path,
                is_locally_imported: true,
                status: 'done',
                model_meta_data: { ...FAMILIES[info.family], ...info.defaults, type: type, family: info.family },
            })
        },

        // any diffusers repo on huggingface.co, or one .safetensors file of a repo (single-file model, LoRA, MLX
        // transformer): fetched into the HF cache (nothing to fetch if it's already there) and registered like an
        // imported model, so Remove only forgets it. base_id: the asset of the base model the repo's card names,
        // which gives a LoRA its family and an MLX transformer the rest of its pipeline. callback(error or undefined)
        import_hf_model(repo, asset_id, file, base_id, on_progress, callback){
            ipc_download(['download_hf_model', repo, file ? '' : 'fp16', file || ''], on_progress, (asset_path) => {
                window.ipcRenderer.invoke('inspect_model', asset_path).then(result => {
                    let info = result.info || {}
                    let base = this.catalog_entry(base_id) || this.all_avail_assets[base_id]
                    info.family = info.family || (base && base.model_meta_data.family)
                    let err = result.success ? this.add_local_asset(asset_path, asset_id, info) : result.error
                    if(!err)
                        Vue.set(this.local_assets[asset_id], 'hf_repo', repo)
                    if(!err && info.mlx_bits){
                        Vue.set(this.local_assets[asset_id], 'base_model_id', base_id)
                        Vue.set(this.local_assets[asset_id].model_meta_data, 'mlx', true)
                    }
                    callback(err)
                })
            }, callback)
        },

        get_downloaded_asset_path(asset_id){
            return (this.downloaded_assets[asset_id] || this.local_assets[asset_id] || {}).asset_path
        },

        get_downloaded_asset(asset_id){
            return (this.downloaded_assets[asset_id] || this.local_assets[asset_id] )
        },

        delete_asset(asset_id){

            let asset_details = this.downloaded_assets[asset_id] || this.local_assets[asset_id] || this.downloading[asset_id]

            Vue.delete(this.downloaded_assets, asset_id );
            Vue.delete(this.downloading, asset_id );
            Vue.delete(this.local_assets, asset_id );

            // imported models are the user's own files: only forget them
            if(!asset_details || asset_details.is_locally_imported || !asset_details.asset_path)
                return
            // the HF cache is shared with other tools and only ever written by huggingface_hub downloads: only forget
            if(asset_details.hf_repo){
                this.set_hf_hidden(asset_id, true)
                Vue.$toast.default(`Removed ${asset_details.title || asset_id}. Its files stay in the Hugging Face cache; "hf cache rm model/${asset_details.hf_repo}" deletes them.`, {duration: 15000})
            }
            else
                window.ipcRenderer.sendSync('delete_file',  asset_details.asset_path );
        },

        // catalog models the user removed: the startup HF cache scan skips them until they're downloaded again
        set_hf_hidden(asset_id, hidden){
            let h = window.ipcRenderer.sendSync('load_data', 'hidden_hf_models.json')
            if(hidden)
                h[asset_id] = true
            else
                delete h[asset_id]
            window.ipcRenderer.sendSync('save_data', h, 'hidden_hf_models.json')
        },

        download_asset(asset_details){
            asset_details = JSON.parse(JSON.stringify(asset_details))
            let that = this;
            let asset_id = asset_details.id;

            if(this.downloaded_assets[asset_id]){
                return;
            }
            if(this.downloading[asset_id] && ['done', 'downloading'].includes(this.downloading[asset_id].status) ){
                return;
            }

            Vue.set( that.downloading  , asset_id , asset_details)
            Vue.set( that.downloading[asset_id] , 'status' , 'downloading')

            function on_progress(progress){
                Vue.set( that.downloading[asset_id] , 'progress' , progress)
            }

            function on_error(error ){
                Vue.set( that.downloading[asset_id] , 'status' , 'error')
                Vue.set( that.downloading[asset_id] , 'error' , error )
            }

            function on_done(asset_path){
                Vue.set( that.downloading[asset_id] , 'asset_path' , asset_path)
                Vue.set( that.downloading[asset_id] , 'status' , 'done')
                Vue.set( that.downloaded_assets , asset_id , that.downloading[asset_id])
            }

            if(asset_details.hf_repo){
                this.set_hf_hidden(asset_id, false)
                // the backend fetches the repo into the HF cache and returns the snapshot folder. Gated repos use the
                // Settings token or the `hf auth login` one; without either the backend reports the gate
                ipc_download(['download_hf_model', asset_details.hf_repo, asset_details.variant || "", ""], on_progress, on_done, on_error)
                return
            }

            // plain file (the ControlNet preprocessor .onnx files), md5 checked
            let dest_path = window.ipcRenderer.sendSync('get_assets_dir') + "/" + asset_id + "_" + asset_details.filename
            ipc_download(['download-file', asset_details.url, dest_path], on_progress, function(file_hash){
                if(file_hash == asset_details.md5)
                    on_done(dest_path)
                else
                    on_error("failed to match checksum")
            }, on_error)
        }

    },
}
</script>
<style>
</style>
<style scoped>
</style>
