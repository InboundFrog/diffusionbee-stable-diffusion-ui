<template>
    <TwoColAppletLayout>
        
        <template v-slot:input_workspace>
              <slot name="input_workspace_pre_form"></slot>
              <Form ref="form"  :form_data="input_form_elements_processed"  :form_values="sd_options" :form_save_key="'1fdaedjdfddeef1'+name" :tags="all_tags" ></Form>
              <div @click="$refs.form.reset_to_default()" class="l_button" style="margin-left:-10px"> Reset to default </div>
              <slot name="input_workspace_post_form"></slot>
        </template>

        <template v-slot:input_buttons>
            <slot name="input_buttons"></slot>
        </template>

        <template v-slot:output_workpace>
            <div class="model_dialog_container" v-if="to_download_left.length > 0 ">
                
            </div>
            <div class="model_dialog" v-if="to_download_left.length > 0 ">
                <h2> You need to download the following models to generate: </h2>
                <br>
                <p>{{to_download_left[0].title}} <span v-if="to_download_left[0].size_gb">({{to_download_left[0].size_gb}} GB)</span></p> <DownloadButton :app=app  :asset_details="to_download_left[0]"> </DownloadButton>
            </div>

            <slot name="output_workpace"></slot>
        </template>

    </TwoColAppletLayout>
</template>
<script>

import Form from "../components_bare/Form.vue"
import TwoColAppletLayout from "../components_bare/TwoColAppletLayout.vue"
import DownloadButton from "./DownloadButton.vue"
import Vue from 'vue'
import {find_in_form_recursive} from "../utils.js"

// drop form elements by id. Containers left without children are dropped too.
function drop_form_fields(form, ids){
    return form.filter(el => {
        if(ids.includes(el.id))
            return false
        if(!el.children || el.children.length == 0)
            return true
        el.children = drop_form_fields(el.children, ids)
        return el.children.length > 0
    })
}

function for_each_form_field(form, id, fn){
    for(let el of form){
        if(el.id == id)
            fn(el)
        if(el.children)
            for_each_form_field(el.children, id, fn)
    }
}

// multiples of 64 around the family's native resolution (512 -> 256..896)
function size_options(default_size){
    let sizes = []
    for(let s = 64*Math.ceil(default_size/128); s <= default_size*1.75; s += 64)
        sizes.push(s)
    if(!sizes.includes(default_size))
        sizes.push(default_size)
    return sizes.sort((a, b) => a - b)
}

// which capability a page needs from the model
const PAGE_CAPABILITY = {img2img: 'supports_img2img', inpainting: 'supports_inpaint'}

function prep_sd_options(options){
    options = JSON.parse(JSON.stringify(options))


    if(options.seed)
        options.seed = Number(options.seed);
    else if(options.seed != undefined)
        options.seed = Math.floor(Math.random() * 100000);

    if(options.seed < 0){
        options.seed = Math.floor(Math.random() * 100000);
    }

    return options
}



export default {
    name: 'BasicSDApplet',
    props: {
        app:Object, 
        input_form : Array,
        sd_options: Object,
        name:String,
        form_tags: Array,
        required_assets : Array, 
        model_options_types : Array, // what kind of models should it show in teh dropdown ( sd, sd_inpaint etc )
    },
    components: {Form  , TwoColAppletLayout, DownloadButton },
    mounted() {

    },
    data() {
        return {
            input_form_options: {}
        };
    },
    methods: {
        
        // given an options dict, load those values inthe form
        load_options(options){
            console.log("load")
            console.log(options)
            let raw_options = options; // the unprocessed form ouptuts
            if(options.raw_form_options){
                raw_options = options.raw_form_options
            }

            for(let k of Object.keys(options)){
                if(raw_options[k] != undefined){
                    console.log("setting " + k )
                    Vue.set(this.sd_options, k ,  raw_options[k])
                }
                    
            }
        } , 

        request_ojects_from_img_element(object_name , object ){
            console.log(object_name)
            if(object_name == "get_mask_img_b64"){
                this.get_mask_img_b64 = object
            }

            if(object_name == "get_mask_b64"){
                this.get_mask_b64 = object
            }
            
       },

        get_sd_form_outputs(){
            let options = this.$refs.form.post_processed_form_outputs()
            options = prep_sd_options(options)

            options.applet_name = this.name

            let am = this.app.assets_manager
            if(options.selected_sd_model){
                let meta = am.model_meta(options.selected_sd_model) || {}
                options.model_path = am.get_downloaded_asset_path(options.selected_sd_model)
                options.model_family = meta.family
                if(meta.type == 'inpaint_model')
                    options.inpaint_model_path = options.model_path
                // hidden (basic mode, or CFG-free family): use the family default
                if(options.guidance_scale === undefined)
                    options.guidance_scale = meta.default_guidance
            }

            if(options.selected_lora && options.selected_lora != "None")
                options.lora_paths = [am.get_downloaded_asset_path(options.selected_lora)]

            // if possible get the input image masks and stuff
            if(options.input_img){
                if(this.get_mask_b64){
                    let mask_b64 = this.get_mask_b64()
                    if(mask_b64){
                        options.mask_image = (window.ipcRenderer.sendSync('save_b64_image',  mask_b64 , true ))
                    }
                }
                if(this.get_mask_img_b64){
                    let mask_img_b64 = this.get_mask_img_b64()
                    if(mask_img_b64){
                        options.input_image_with_mask =  (window.ipcRenderer.sendSync('save_b64_image',  mask_img_b64  ))
                    }
                }
            }

            return options
        } , 

        check_input_form_n_show_error(){
            if(this.to_download_left.length > 0){
                this.app.show_toast("First you need to download the model to generate")
                return false;
            } 

            if( this.get_sd_form_outputs().prompt != undefined && this.get_sd_form_outputs().prompt.trim() == ""){
                console.log(this.get_sd_form_outputs())
                this.app.show_toast('You need to enter a prompt')
                return false;
            }

            if(this.sd_options.selected_sd_model){
                if(!(this.app.assets_manager.get_downloaded_asset_path(this.sd_options.selected_sd_model))){
                    this.app.show_toast("Model not found. Please select a valid model.")
                    return false                    
                }
            }

            return true;
        },

    },
    computed : {
        all_tags(){
            console.log("sddd options ")
            console.log(this.sd_options)
            let l = (this.form_tags || []).concat( this.sd_options.is_adv_mode ? ['advanced']:[]  )
            return l
        } , 

        selected_model_meta(){
            return this.app.assets_manager.model_meta(this.sd_options.selected_sd_model) || {}
        },

        input_form_elements_processed(){
            let form = JSON.parse(JSON.stringify(this.input_form))
            let am = this.app.assets_manager
            let meta = this.selected_model_meta

            // add the avail models that can do what this page needs
            let el = find_in_form_recursive( "selected_sd_model" , form)
            if(el){
                let types = this.model_options_types || ["sd_model"]
                let capability = PAGE_CAPABILITY[this.name]
                for(let id in am.all_avail_assets){
                    let m = am.model_meta(id)
                    if(m && types.includes(m.type) && (!capability || m[capability]) && !el.options.includes(id))
                        el.options.push(id)
                }
            }

            // LoRAs made for the selected model family
            let lora_el = find_in_form_recursive( "selected_lora" , form)
            if(lora_el){
                for(let id in am.all_avail_assets){
                    let m = am.model_meta(id) || {}
                    if(m.type == 'lora' && (!m.family || m.family == meta.family))
                        lora_el.options.push(id)
                }
            }

            // get the functions to get the mask and stuff
            el =  find_in_form_recursive( "input_img" , form)
            if(el && el.draw_mask && el.store_masked_image){
                el.request_objects_from_element = "yes"
                el.request_objects_from_element_fn = this.request_ojects_from_img_element
            }

            let drop = []
            if(!lora_el || lora_el.options.length < 2)
                drop.push("selected_lora")

            // model family defaults and capabilities
            if(meta.family){
                for_each_form_field(form, "num_steps", x => x.default_value = meta.default_steps)
                for_each_form_field(form, "guidance_scale", x => x.default_value = meta.default_guidance)
                for(let id of ["img_width", "img_height"]){
                    for_each_form_field(form, id, x => {
                        x.default_value = meta.default_size
                        x.options = size_options(meta.default_size)
                    })
                }

                if(!meta.supports_negative_prompt)
                    drop.push("negative_prompt")
                if(!meta.default_guidance) // CFG-distilled (Z-Image Turbo, FLUX.1 schnell)
                    drop.push("guidance_scale")
                if(meta.family != "sd15")
                    drop.push("is_clip_skip_2")
                if(!(meta.supports_controlnet && am.catalog.some(c => c.model_meta_data.type == 'controlnet' && c.model_meta_data.family == meta.family)))
                    drop.push("controlnet_acc")
            }

            return drop_form_fields(form, drop);
        },

        required_assets_modified(){
            let ret = JSON.parse(JSON.stringify(this.required_assets || []))
            // a catalog model is selected but not downloaded yet
            let entry = this.app.assets_manager.catalog_entry(this.sd_options.selected_sd_model)
            if(entry)
                ret.unshift(entry)
            return ret
        },

        to_download_left(){
            let to_download = []
           
            for(let asset of this.required_assets_modified){
                let asset_id = asset.id
                if(!(this.app.is_mounted && this.app.assets_manager.downloaded_assets[asset_id] && this.app.assets_manager.downloaded_assets[asset_id].status == 'done')){
                    to_download.push(asset)
                }
            }
            return to_download
        }
    },

    watch: {
        // switching to a model of another family: load that family's defaults
        'sd_options.selected_sd_model'(new_id, old_id){
            if(old_id === undefined) // the saved form is being restored
                return
            let meta = this.app.assets_manager.model_meta(new_id)
            let old_meta = this.app.assets_manager.model_meta(old_id)
            if(!meta || (old_meta && old_meta.family == meta.family))
                return
            Vue.set(this.sd_options, 'num_steps', meta.default_steps)
            Vue.set(this.sd_options, 'guidance_scale', meta.default_guidance)
            Vue.set(this.sd_options, 'img_width', meta.default_size)
            Vue.set(this.sd_options, 'img_height', meta.default_size)
        }
    }
}
</script>
<style>
</style>
<style scoped>


.model_dialog_container{
    position: fixed;
    left: calc(    var(--sidebar-width )  + 360px );
    right:0 ;
    bottom:0;
    background-color: rgba(0,0,0,0.3);
/*    height: 100%;*/
    z-index: 900;
    top: var(--titlebar-height);
}

.model_dialog{
    position: fixed;
    width: 250px ;
    height:170px;
    background: var(--options-input-bg );
    left: calc( 50% +  var(--sidebar-width ) /2 + 175px );
    top:50%;
     transform: translateY(-50%) translateX(-50%);
    z-index: 1000;
    padding:20px;
}


</style>