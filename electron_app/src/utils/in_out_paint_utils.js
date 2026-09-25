

// Generative Fill with a regular (non-inpainting) model uses the ControlNet inpaint model when the family has one (SD 1.5).
// Dedicated inpainting models, and base models of other families, inpaint natively.
function inpaint_controlnet(self){
    let meta = self.app.assets_manager.model_meta(self.sd_options.selected_sd_model)
    if(!meta || meta.type != "sd_model" || !meta.supports_controlnet)
        return undefined
    return self.app.assets_manager.controlnet_entry("Inpaint", meta.family)
}

function inpaint_assets(self , mode ){
    let to_download = []
    let cn = inpaint_controlnet(self)
    if(mode == "Generative Fill" && cn)
        to_download.push(cn)
    return to_download;
}

function prep_sd_optins(self , sd_options_object, mode , img_mask_url){
    if(!sd_options_object.num_imgs)
        sd_options_object.num_imgs = 1

    if( mode == "Text To Image") {
        sd_options_object.sd_mode_override = "txt2img"
    } else {

        if(mode == "Generative Fill"){
        
            let cn = inpaint_controlnet(self)

            if(cn){
                sd_options_object.do_masking_diffusion = true
                sd_options_object.controlnet_model ="Inpaint"
                sd_options_object.controlnet_inp_img_preprocesser="Inpaint"
                sd_options_object.controlnet_input_image_path="NULL"
                sd_options_object.is_control_net=true
                sd_options_object.controlnet_guess_mode=true
                sd_options_object.guidance_scale = ( (sd_options_object.guidance_scale || 7.5) - 3.5 )
                if(sd_options_object.guidance_scale <= 1)
                    sd_options_object.guidance_scale = 1
                sd_options_object.sd_mode_override = "txt2img"
                sd_options_object.controlnet_path = self.app.assets_manager.get_downloaded_asset_path(cn.id)

            } else {
                // dedicated inpaint model (inpaint_model_path is set by BasicSDApplet) or a base model with native inpainting
                sd_options_object.sd_mode_override = "txt2img"
            }
        } else {
            // Image to Image case
            sd_options_object.infill_alpha = true
        }

        sd_options_object.get_mask_from_image_alpha = true;

        if(self.sd_options.inp_only_update_masked){
            sd_options_object.blur_mask = true
            sd_options_object.do_masking_diffusion = true
            
        }
        sd_options_object.mask_image_path = img_mask_url
    }
}

export { prep_sd_optins ,  inpaint_assets }