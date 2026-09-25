<template>
    <div class="main_container">

        <div class="l_button button_colored button_medium" style="float:right;" @click="import_model_locally"> Import From Computer </div>
        <p style="opacity:0.6"> Import a .safetensors model or LoRA, or a diffusers model folder. </p>
        <hr>

        <h2 v-if="downloaded_models_list.length > 0 || is_local_model_importing"> My Models </h2>
        <div v-if="downloaded_models_list.length > 0 || is_local_model_importing" class="icon_container">

            <div v-if="is_local_model_importing"  class="model_card" style="padding:20px">
                <h2>Importing model</h2>
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
                    <p v-if="model.requires_hf_token" style="zoom:0.8"> Gated: accept the license on <a href="#" @click.prevent="open_repo_page(model)">huggingface.co</a> and add a Hugging Face token in Settings. </p>
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