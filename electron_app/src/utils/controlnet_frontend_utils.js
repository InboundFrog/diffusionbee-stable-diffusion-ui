
// ONNX preprocessors that turn an input image into a control image. Framework independent, plain md5-checked downloads.
const PREPROCESSORS = {
    "Depth" : {
        "id": "midas_monodepth",
        "filename": "midas_monodepth.onnx",
        "md5": "3ad9d8a9d820214d2bfe65c0d37683a7",
        is_stock_model : true,
        "url": "https://huggingface.co/divamgupta/controlnet_tensorflow/resolve/main/midas_monodepth.onnx",
        "title": "MiDaS Depth Extractor",
        "description": "MiDaS Depth Extractor, for processing ControlNet input.",
        model_meta_data : {"type" : "controlnet_preprocess_model"   }
    },
    "BodyPose" : {
        "id": "body_pose_model",
        "filename": "body_pose_model.onnx",
        "md5": "51499a50aff296971a99605f32c7e8e7",
        is_stock_model : true,
        "url": "https://huggingface.co/divamgupta/controlnet_tensorflow/resolve/main/body_pose_model.onnx",
        "title": "OpenPose Extractor",
        "description": "OpenPose Extractor, for processing ControlNet input.",
        model_meta_data : {"type" : "controlnet_preprocess_model"   }
    },
    "LineArt" : {
        "id": "lineart_model",
        "filename": "lineart_model.onnx",
        "md5": "667d0d7cad9449f3a6db135b526f2279",
        is_stock_model : true,
        "url": "https://huggingface.co/divamgupta/controlnet_tensorflow/resolve/main/linart.onnx",
        "title": "LineArt Extractor",
        "description": "LineArt Extractor, for processing ControlNet input.",
        model_meta_data : {"type" : "controlnet_preprocess_model"   }
    },
}

function is_yes(v){
    return v == true || v == "Yes"
}

// the catalog ControlNet for the selected mode, if the selected model can use it
function selected_controlnet(self , mode){
    if(!mode || mode == "None")
        return undefined
    let meta = self.app.assets_manager.model_meta(self.sd_options.selected_sd_model)
    if(!meta || !meta.supports_controlnet)
        return undefined
    return self.app.assets_manager.controlnet_entry(mode, meta.family)
}

function controlnet_required_assets(self , to_download){
    let mode = self.sd_options.controlnet_model
    let cn = selected_controlnet(self, mode)
    if(!cn)
        return
    to_download.push(cn)
    if(is_yes(self.sd_options.do_controlnet_preprocess) && PREPROCESSORS[mode])
        to_download.push(PREPROCESSORS[mode])
}

function controlnet_proc_form_outputs(self , options){
    let cn = selected_controlnet(self, options.controlnet_model)
    if(!cn)
        return
    options.controlnet_path = self.app.assets_manager.get_downloaded_asset_path(cn.id)
    if(options.do_controlnet_preprocess && PREPROCESSORS[options.controlnet_model])
        options.controlnet_inp_img_preprocesser_model_path = self.app.assets_manager.get_downloaded_asset_path(PREPROCESSORS[options.controlnet_model].id)
}

function controlnet_check_inputs(self , vue){

    if(selected_controlnet(self, self.sd_options.controlnet_model)){
        if((!self.sd_options.controlnet_input_image_path) || self.sd_options.controlnet_input_image_path == ""){
            vue.$toast.default("You have selected a ControlNet model but not specified an image.")
             return false;
        }

    }

    return true
}


export { controlnet_check_inputs , controlnet_proc_form_outputs , controlnet_required_assets }
