<template>
    <div class="main_container">

        <div class="l_button button_colored button_medium" style="float:right;" @click="import_model_locally"> Import From Computer </div>
        <div class="l_button button_colored button_medium" style="float:right;" @click="show_hf_import"> Import From Hugging Face </div>
        <p style="opacity:0.6"> Import a .safetensors model or LoRA, or a diffusers model folder. </p>
        <div v-if="hf_cached_repos">
            <input v-model.trim="hf_repo" list="hf_cached_repos" placeholder="org/model" @keyup.enter="import_hf_model" style="width: 360px">
            <datalist id="hf_cached_repos"><option v-for="r in hf_cached_repos" :key="r" :value="r"></option></datalist>
            <div class="l_button button_colored button_small" style="display:inline-block" @click="import_hf_model"> Import </div>
            <div v-if="hf_files">
                <select v-model="hf_file" style="width: 360px">
                    <option v-for="f in hf_files.files" :key="f[0]" :value="f[0]"> {{f[0]}} ({{(f[1] / 1e9).toFixed(2)}} GB) </option>
                </select>
                <div class="l_button button_colored button_small" style="display:inline-block" @click="import_hf_file(hf_file)"> Import File </div>
            </div>
            <p style="opacity:0.6"> A repo id or huggingface.co link: a diffusers model, or a repo of .safetensors models, LoRAs or MLX transformers (GGUF isn't supported). The list has the ones already in the Hugging Face cache ({{hf_cached_repos.length}}); others are downloaded into it. </p>
        </div>
        <hr>

        <h2 v-if="downloaded_models_list.length > 0 || is_local_model_importing"> My Models </h2>
        <div v-if="downloaded_models_list.length > 0 || is_local_model_importing" class="icon_container">

            <div v-if="is_local_model_importing"  class="model_card" style="padding:20px">
                <h2>Importing model</h2>
                <p v-if="import_progress"> Downloading {{import_progress}}% </p>
                <p v-else-if="hf_checking"> Checking the repo </p>
                <br>
                 <MoonLoader class="moonloader" color="#000000" size="50px" style="zoom:0.4"></MoonLoader>
            </div>

            <div v-for="model in downloaded_models_list" :key="model.id" class="model_card" v-bind:style="{ 'background-image': 'url(' + (model.img_url || default_img_url )+ ')' }">
                 <div class="card_desc"> 
                    <h2> {{model.title || model.id}} </h2> 
                    <p> {{model.description}} </p> 
                    <p style="zoom:0.7"> {{ model_metadata_to_str(model) }}</p>
                    <DownloadButton :app=app  :asset_details="model"> </DownloadButton>
                </div> 
            </div>

        </div>

        <br>


        <h2> Available Models </h2>
        <div class="icon_container">

            <div v-for="model in not_downloaded_models_list" :key="model.id" class="model_card" v-bind:style="{ 'background-image': 'url(' + (model.img_url || default_img_url) + ')' }">
                 <div class="card_desc"> 
                    <h2> {{model.title || model.id}} </h2> 
                    <p> {{model.description}} </p> 
                    <p style="zoom:0.7"> {{ model_metadata_to_str(model) }}</p>
                    <p v-if="model.min_ram_gb > total_ram_gb" style="color:red; zoom:0.8"> Needs {{model.min_ram_gb}} GB RAM, this Mac has {{total_ram_gb}} GB. It may be very slow. </p>
                    <p v-if="model.requires_hf_token" style="zoom:0.8"> Gated: accept the license on <a href="#" @click.prevent="open_repo_page(model)">huggingface.co</a>, then log in with <code>hf auth login</code> or add a token in Settings. </p>
                    <DownloadButton :app=app  :asset_details="model"> </DownloadButton>
                </div> 
            </div>

        </div>

    </div>
</template>
<script>

import DownloadButton from "../components/DownloadButton.vue"
import MoonLoader from 'vue-spinner/src/MoonLoader.vue'

const ModelStore ={
    name: 'ModelStore',
    props: {app:Object, },
    components: {DownloadButton, MoonLoader},
    mounted() {
    },
    data() {
        return {
            is_local_model_importing : false, 
            hf_cached_repos : null, // HF cache repos not added yet, once the Hugging Face import is open
            hf_repo : "",
            hf_files : null, // {repo, base_model, files: [[name, bytes, info]]} of a repo of single files, to pick one
            hf_file : "",
            hf_checking : false,
            import_progress : 0,
            default_img_url : require("../assets/imgs/page_icon_imgs/default.png"),
            total_ram_gb : window.ipcRenderer.sendSync('get_total_ram_gb'),
        };
    },
    computed: {
        downloaded_models_list(){
            if(!this.app.is_mounted)
                return []

            let ret = []
            for(let k in this.app.assets_manager.all_avail_assets){
                ret.unshift(this.app.assets_manager.all_avail_assets[k])
            }
            return ret;
        } , 
        not_downloaded_models_list(){
            if(!this.app.is_mounted)
                return []
            let am = this.app.assets_manager
            return am.catalog.filter(model  => !am.downloaded_assets[model.id])
        }
    },
    methods: {
        open_repo_page(model){
            window.ipcRenderer.sendSync('open_url', "https://huggingface.co/" + model.hf_repo)
        },

        model_metadata_to_str(asset_details){
            let meta = asset_details.model_meta_data || {}
            let r = [meta.family, meta.type && meta.type.replaceAll("_", " ")]
            if(asset_details.size_gb)
                r.push(asset_details.size_gb + " GB")
            return r.filter(x => x).join(" · ")
        },

        show_hf_import(){
            window.ipcRenderer.invoke('list_cached_hf_repos').then(repos => {
                let am = this.app.assets_manager
                let added = Object.values(am.all_avail_assets).map(a => a.hf_repo)
                this.hf_cached_repos = repos.filter(r => !added.includes(r)).sort()
            })
        },

        import_hf_model(){
            let repo = this.hf_repo.replace(/^https?:\/\/huggingface\.co\//, "").split("/").slice(0, 2).join("/")
            if(!/^[\w.-]+\/[\w.-]+$/.test(repo)){
                this.app.show_toast("Enter a Hugging Face repo id like black-forest-labs/FLUX.1-schnell")
                return
            }
            let am = this.app.assets_manager
            let entry = am.catalog.find(m => m.hf_repo == repo)
            if(entry){ // catalog models download on their own card, with the catalog's settings
                if(!am.downloaded_assets[entry.id])
                    am.download_asset(entry)
                this.app.show_toast(entry.title + " is in the model list below")
                return
            }
            if(this.is_local_model_importing){
                this.app.show_toast("Model is already importing. Please wait")
                return;
            }
            this.hf_files = null
            this.is_local_model_importing = this.hf_checking = true
            window.ipcRenderer.invoke('repo_info', repo).then(result => {
                this.is_local_model_importing = this.hf_checking = false
                if(!result.success)
                    return this.app.show_toast("Error while importing " + result.error)
                if(result.info.diffusers)
                    return this.start_hf_import(repo, repo.split("/")[1])
                this.hf_files = {repo: repo, ...result.info}
                this.hf_file = result.info.files[0][0]
                if(result.info.files.length == 1)
                    this.import_hf_file(this.hf_file)
            })
        },

        // one file of hf_files. A LoRA gets its family, and an MLX transformer its pipeline, from the base model
        import_hf_file(file){
            let am = this.app.assets_manager
            let {repo, base_model, files} = this.hf_files
            let info = files.find(f => f[0] == file)[2]
            let base = am.catalog.find(m => m.hf_repo == base_model) || Object.values(am.all_avail_assets).find(a => a.hf_repo == base_model)
            if(info.mlx_bits && !base){
                this.app.show_toast(`${file} is an MLX transformer for ${base_model || "an unnamed base model"}. Import that model first.`)
                return
            }
            let name = files.length == 1 ? repo.split("/")[1] : file.replace(/\.safetensors$/, "")
            this.start_hf_import(repo, name, file, base && base.id)
        },

        start_hf_import(repo, model_name, file, base_id){
            let am = this.app.assets_manager
            if(am.all_avail_assets[model_name] || am.catalog_entry(model_name)){
                this.app.show_toast("A model with this name already exists");
                return;
            }
            if(this.is_local_model_importing){
                this.app.show_toast("Model is already importing. Please wait")
                return;
            }
            this.is_local_model_importing = true
            this.import_progress = 0
            am.import_hf_model(repo, model_name, file, base_id, p => { this.import_progress = p }, err => {
                this.is_local_model_importing = false
                this.import_progress = 0
                if(err)
                    this.app.show_toast("Error while importing " + err)
                else {
                    this.hf_repo = ""
                    this.hf_files = null
                    this.hf_cached_repos = this.hf_cached_repos.filter(r => r != repo)
                }
            })
        },

        import_model_locally(){

            if(this.is_local_model_importing)
            {
                this.app.show_toast("Model is already importing. Please wait")
                return;
            }

            let that = this;
            let model_path = window.ipcRenderer.sendSync('file_dialog',  "weights_file" );
            if(!model_path || model_path == "NULL")
                return;

            let model_name = model_path.replaceAll("\\" , "/").replace(/\/+$/, "").split("/").pop().replace(/\.safetensors$/i, "")

            if(model_name.trim() == ""){
                this.app.show_toast("Put non empty model name");
                return;
            }

            if(this.app.assets_manager.all_avail_assets[model_name] || this.app.assets_manager.catalog_entry(model_name)){
                this.app.show_toast("A model with this name already exists");
                return;
            }

            this.is_local_model_importing = true;

            // no conversion: the backend just reads the header to find the model family
            window.ipcRenderer.invoke('inspect_model', model_path).then((result) => {
                that.is_local_model_importing = false;
                let err = result.success ? that.app.assets_manager.add_local_asset(model_path, model_name, result.info) : result.error
                if(err)
                    that.app.show_toast("Error while importing " + err )
            })
        }
    },
}

export default ModelStore;
ModelStore.title = "Models"
ModelStore.icon = "cubes"
ModelStore.description = "Download, imoport and manage models"
ModelStore.img_icon = require("../assets/imgs/page_icon_imgs/models.png")
ModelStore.home_category = "pages"
ModelStore.sidebar_show = "always"
// add this to the always_on_pages to the PagesRouter

</script>
<style>
</style>
<style scoped>

.main_container{
    padding: 20px;
    width: 100%;
    height: 100%;
    overflow: auto;
}

.icon_container{
    display: flex;
   flex-wrap: wrap;
}

.model_card{
    width:230px;
    height: 170px;
    margin: 5px;
    background-size: contain;
    background-color: var(--sidebar-color);
    position: relative;
}


@media only screen and (max-width: 1730px) {
  .model_card {
    width : calc(20% - 10px)
  }
}



@media only screen and (max-width: 1430px) {
  .model_card {
    width : calc(25% - 10px)
  }
}


@media only screen and (max-width: 1200px) {
  .model_card {
    width : calc(33% - 10px)
  }
}



.card_desc{
    background-color: var(--sidebar-color); ;
    padding: 5px;
    position: absolute;
    bottom: 0;
    width: 100%;
}

.card_desc > p{
    margin-bottom: 3px;
}



</style>