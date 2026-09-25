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

        // model_meta_data of a catalog or downloaded/imported asset
        model_meta(asset_id){
            let asset = this.catalog_entry(asset_id) || this.all_avail_assets[asset_id]
            return asset && asset.model_meta_data
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
                model_meta_data: { ...FAMILIES[info.family], type: type, family: info.family },
            })
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
            if(asset_details.hf_repo)
                window.ipcRenderer.sendSync('delete_hf_model',  asset_details.asset_path );
            else
                window.ipcRenderer.sendSync('delete_file',  asset_details.asset_path );
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
                let settings = ((this.$parent.app_state || {}).app_data || {}).settings || {}
                if(asset_details.requires_hf_token && !settings.hf_token){
                    on_error("This model is gated. Accept its license on huggingface.co and add a Hugging Face token in Settings.")
                    return
                }
                // the backend fetches the repo into the HF cache and returns the snapshot folder
                ipc_download(['download_hf_model', asset_details.hf_repo, asset_details.variant || ""], on_progress, on_done, on_error)
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
