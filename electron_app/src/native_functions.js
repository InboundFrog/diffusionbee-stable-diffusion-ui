import { ipcMain, dialog, clipboard, net } from 'electron'
import { app , screen } from 'electron'
import settings from 'electron-settings';

const path = require('path');

var win;


function bind_window_native_functions(w) {
    console.log("browser object binded")
    win = w;
}


let is_windows = process.platform.startsWith('win');



console.log(require('os').freemem()/(1000000000) + " Is the free memory")
console.log(require('os').totalmem()/(1000000000) + " Is the total memory")


ipcMain.on('save_dialog', (event, ...args) => {

    const filename = args[0] ? args[0] : "Untitled"
    const ext = args[1] ? args[1] : "png"
   
    let trimmedFilename = filename.substring(0, 254) // filename size limit
     let save_path = dialog.showSaveDialogSync({
            title: 'Save Image',
            defaultPath: trimmedFilename,
            filters: [{
              name: 'Image',
              extensions: [ext]
            }]
          })

     event.returnValue = save_path;
} )

console.log(require('os').release() + " ohoho")



ipcMain.on('file_dialog', (event, arg) => {
    console.log("file dialog request recieved" + arg) // prints "ping"
    let properties;
    let options;

    if (arg == "folder") // single folder 
    {
        properties = ['openDirectory'];
        options = { properties: properties } ;
    }
    else if(arg == 'img_file') // single image file 
    {
        properties = ['openFile' ]
        options = { filters :[ {name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp']}] , properties: properties } ;
    }
    else if(arg == 'weights_file') // a .safetensors model / LoRA, or a diffusers model folder. No .ckpt: pickle is unsafe
    {
        properties = ['openFile', 'openDirectory' ]
        options = { filters :[ {name: 'Models', extensions: ['safetensors' ]}] , properties: properties } ;
    }
    else if(arg == 'img_files') // multi image files
    {
        properties = ['multiSelections' , 'openFile' ]
        options = { filters :[ {name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp']}] , properties: properties } ;
    }
    else if(arg == 'text_files') // multi image files
    {
        properties = ['multiSelections' , 'openFile' ]
        options = { filters :[ {name: 'Images', extensions: ['txt']}] , properties: properties } ;
    }
    else if(arg == 'audio_files') // multi image files
    {
        properties = ['multiSelections' , 'openFile' ]
        options = { filters :[ {name: 'Images', extensions: ['mp3', 'wav']}] , properties: properties } ;
    }
    else if(arg == 'video_files') // multi image files
    {
        properties = ['multiSelections' , 'openFile' ]
        options = { filters :[ {name: 'Images', extensions: ["mp4", "mov", "avi", "flv", "wmv", "mkv"]}] , properties: properties } ;
    }
    else if(arg == 'any_files') // multi image files
    {
        properties = ['multiSelections' , 'openFile' ]
        options = {  properties: properties } ;
    }
    else
    {
        properties = ['openFile'];
        options = { properties: properties } ;
    }

    // let options = {
    //     See place holder 1 in above image
    //     title : "Custom title bar", 
    //     message : "Custom title bar",

    //     buttonLabel : "Custom button",

    //     See place holder 4 in above image
    //     filters :[
    //      {name: 'Images', extensions: ['jpg', 'png', 'gif']},
    //      {name: 'Movies', extensions: ['mkv', 'avi', 'mp4']},
    //      {name: 'Custom File Type', extensions: ['as']},
    //      {name: 'All Files', extensions: ['*']}
    //     ],
    //     properties: properties
    // }

    // //Synchronous
    let filePaths = dialog.showOpenDialogSync(options)

    if (filePaths && filePaths.length > 0)
        event.returnValue = filePaths.join(";;;");
    else
        event.returnValue = "NULL";
})




ipcMain.on('open_url', (event, url) => {
    if (/^https?:\/\//i.test(url)) // only web links go to the system browser
        require('electron').shell.openExternal(url);
    event.returnValue = '';
})



ipcMain.on('save_file', (event, arg) => {
    let p1 = arg.split("||")[0];
    let p2 = arg.split("||")[1];
    require('fs').copyFileSync(p1, p2);
    event.returnValue = '';
})


ipcMain.on('copy_to_clipboard', (event, arg) => {
    clipboard.writeText(arg)
    event.returnValue = '';
})


ipcMain.on('get_from_clipboard', (event, arg) => {
    event.returnValue = clipboard.readText();
})


ipcMain.on('show_dialog_on_quit', (event, msg) => {
    if(win)
    {
        win.show_dialog_on_quit = true;
        win.dialog_on_msg = msg;
    }
    event.returnValue = 'ok';

})


ipcMain.on('dont_show_dialog_on_quit', (event, arg) => {
    if(win)
        win.show_dialog_on_quit = false;
    event.returnValue = 'ok';

})


ipcMain.on('get_instance_id', (event, arg) => {
    if (settings.hasSync('instance_id')){
        event.returnValue =  settings.getSync('instance_id')
        return;
    }
    let instance_id =  (Math.random() + 1).toString(36);
    settings.set('instance_id', instance_id);
    event.returnValue =   instance_id;

})


ipcMain.on('unfreeze_win', (event, arg) => {

    if (win) {
	win.savable=true;
        const primaryDisplay = screen.getPrimaryDisplay()
        const { width, height } = primaryDisplay.workAreaSize

        if (settings.hasSync('windowPosState')) {
            let windowState = settings.getSync('windowPosState');
            console.log("stateeee")
            console.log(windowState)

            if( windowState.x  >  0.8*width ||  windowState.y  >  0.8*height ||  windowState.x  < -0.2*width || windowState.y  < -0.2*height    ){
                win.setSize(850, 650, false);
            } else {
               win.setPosition( windowState.x  ,  windowState.y  , false);
               win.setMinimumSize(1070, 700);
                win.setSize(windowState.width, windowState.height , false); 
            }

            
        }
        else{
            win.setMinimumSize(1070, 700);
            win.setSize(850, 650, false);
        }

        win.setMinimumSize(1070, 700);
        // win.setResizable(true);
        win.setMaximizable(true);


        
        
    }

    event.returnValue = 'ok';

})



ipcMain.on('freeze_win', (event, arg) => {

    if (win) {
	win.savable=false;
	win.restore()
        win.setMinimumSize(770, 550)
        win.setSize(770, 550, false); 
        // win.setResizable(false);
        win.setMaximizable(false);

        const primaryDisplay = screen.getPrimaryDisplay()
        const { width, height } = primaryDisplay.workAreaSize;

        console.log( width +" " +  height)

        win.setPosition( parseInt((width-770)/2)  , parseInt((height-550)/2), false);

              

    }

    event.returnValue = 'ok';

})



ipcMain.on('show_about', (event, arg) => {

    if (win) {

        if(is_windows)
        {
            let about_content = require('../package.json').name + "\n" + "Version " + require('../package.json').version + " (" + require('../package.json').build_number + ")\n" + require('../package.json').description;
            const choice = require('electron').dialog.showMessageBoxSync(this, {
                buttons: ['Okay'],
                title: require('../package.json').name ,
                message: about_content
            });
        }
        else{
            app.showAboutPanel()
        }

        
    }

    event.returnValue = 'ok';

})




ipcMain.on('native_confirm', (event, arg) => {

    if (win) {
        
        const choice = require('electron').dialog.showMessageBoxSync(this, {
            type: 'question',
            buttons: ['Yes', 'No'],
            title: require('../package.json').name ,
            message: arg
        });
        if (choice === 1) {
            event.returnValue = false ;
        }
        else{
            event.returnValue = true ;
        }

    }
    else{
        event.returnValue = false ;
    }

})



ipcMain.on('close_window', (event, arg) => {

    if (win) {
        
        win.close()
        event.returnValue = true ;
    }
    else{
        event.returnValue = false ;
    }

})




ipcMain.on('native_alert', (event, arg) => {

    if (win) {
        
        const choice = require('electron').dialog.showMessageBoxSync(this, {
            buttons: ['Okay'],
            title: require('../package.json').name ,
            message: arg
        });
        
    }
    
    event.returnValue = true ;

})




ipcMain.on('save_b64_image', (event, b64_str, save_to_tmp ) => {

    const path = require('path');
    const fs = require('fs');

    let base64Data = b64_str.replace(/^data:image\/png;base64,/, "");
    
    const homedir = require('os').homedir();
    let save_dir = path.join(homedir , ".diffusionbee")


    if (!fs.existsSync(save_dir)){
        fs.mkdirSync(save_dir, { recursive: true });
    }

    if(save_to_tmp){
        save_dir = "/tmp/"
    } else {
        save_dir = path.join(save_dir , "inp_images")
    }

    

    if (!fs.existsSync(save_dir)){
        fs.mkdirSync(save_dir, { recursive: true });
    }

    let p = require('path').join(save_dir,  Math.random().toString()+".png");

    require("fs").writeFileSync(p , base64Data, 'base64'); 
    
    event.returnValue = p ;

})



function save_json(data , fname ){
    const path = require('path');
    const fs = require('fs');
    const homedir = require('os').homedir();
    let save_dir = path.join(homedir , ".diffusionbee")


    if (!fs.existsSync(save_dir)){
        fs.mkdirSync(save_dir, { recursive: true });
    }

    let data_path = path.join(homedir , ".diffusionbee" , fname )
    fs.writeFileSync( data_path, JSON.stringify(data) );
}



function load_data(fname){
    const path = require('path');
    const fs = require('fs');
    const homedir = require('os').homedir();
    let data_path = path.join(homedir , ".diffusionbee" , fname );

    if (fs.existsSync(data_path)){
        let json_str = fs.readFileSync( data_path );
        try {
            return JSON.parse(json_str);
          } catch (error) {
            return {} ;
          }
    }
    else{
        return {} ;
    }       
}

ipcMain.on('save_data', (event, arg , fname ) => {
    if(fname)
        save_json(arg, fname)
    else
        save_json(arg, "data.json")
    event.returnValue = true ;
})


ipcMain.on('load_data', (event, fname) => {
    if(fname)
        event.returnValue = load_data(fname)
    else
        event.returnValue = load_data("data.json")
})


ipcMain.on('delete_file', (event, fpath) => {
    const fs = require('fs');
    try{
        fs.unlinkSync(fpath);
        console.log("deleted")
        event.returnValue = true;
    } catch {
        console.log("err in deleting")
        event.returnValue = false;
    }
    
})


// 4x Real-ESRGAN upscale via the backend's `upscale` subcommand (weights: 67 MB, fetched into the HF cache on first use)
function run_realesrgan(input_path , cb ){
    const fs = require('fs');
    let out_path = path.join("/tmp", Math.random() + ".png"); // /tmp is in the dbimg:// allowlist, os.tmpdir() is not
    let proc = spawn_backend_cmd(['upscale', input_path, out_path]);
    proc.on('error', (err) => console.error(`sr error: ${err.message}`)); // missing binary: 'close' still reports the failure
    proc.stderr.on('data', (data) => console.error(`sr stderr: ${data}`));
    proc.on('close', () => cb(fs.existsSync(out_path) ? out_path : ''));
}



// One-shot backend subcommands (download_model, inspect_model). Same dev/prod switch as the main backend:
// dev runs the python script (with backends/.venv python when present), prod the bundled binary.
function spawn_backend_cmd(args, env){
    const fs = require('fs');
    let script_path = process.env.PY_SCRIPT || "../backends/stable_diffusion/diffusionbee_backend.py";
    if (fs.existsSync(script_path)) {
        let venv_python = path.join(path.dirname(script_path), "..", ".venv", "bin", "python");
        return require('child_process').spawn( fs.existsSync(venv_python) ? venv_python : "python3" , [ script_path ].concat(args) , { env: env || process.env });
    }
    let bin_path =  path.join(path.dirname(__dirname), 'core' , 'diffusionbee_backend' );
    return require('child_process').spawn( bin_path , args , { env: env || process.env });
}

// the backend's error is its last stderr line
function last_line(text){
    return text.split("\n").map(l => l.trim()).filter(l => l).pop() || ""
}


// env for backend commands that talk to the Hub: the token from Settings, if any
function hf_env(){
    let env = Object.assign({}, process.env);
    let hf_token = (load_data("app_data_2.json").settings || {}).hf_token;
    if (hf_token)
        env.HF_TOKEN = hf_token;
    return env;
}


// Which of these repos ("repo_id:variant") are already fully in the HF cache: resolves {repo_id: snapshot folder}.
// Downloads nothing.
ipcMain.handle('find_cached_hf_models', async (event, repos) => {
    return await new Promise(resolve => {
        let found = {};
        let proc = spawn_backend_cmd(["cached_models"].concat(repos), hf_env());
        require('readline').createInterface({ input: proc.stdout }).on('line', (line) => {
            let [tag, repo_id, ...rest] = line.split(" ");
            if (tag == "cached")
                found[repo_id] = rest.join(" ");
        });
        proc.stderr.on('data', (data) => console.log(`cached_models: ${data}`));
        proc.on('error', () => resolve({}));
        proc.on('close', () => resolve(found));
    });
})


// Download a HF repo into the HF cache. Progress and result go over the same `to_download` channel as download-file,
// success carries the local snapshot folder.
ipcMain.on('download_hf_model', (event, repo_id, variant, downloadId) => {
    const send = (fn, msg) => {
        try {
            event.sender.send(`to_download`, {fn: fn , download_id: downloadId , msg: msg });
        } catch (err) {
            console.log(err)
        }
    }

    let proc = spawn_backend_cmd( ["download_model", repo_id].concat(variant ? ["--variant", variant] : []) , hf_env() );
    let snapshot_dir = "";
    let errors = "";
    let finished = false;

    require('readline').createInterface({ input: proc.stdout }).on('line', (line) => {
        if (line.startsWith("progress "))
            send('progress', Math.round(Number(line.slice(9))) || 0);
        else if (line.startsWith("done "))
            snapshot_dir = line.slice(5).trim();
    });

    proc.stderr.on('data', (data) => {
        console.error(`download_model stderr: ${data}`);
        errors = (errors + data).slice(-5000);
    });

    proc.on('error', (err) => { // could not start the backend
        if (finished) return;
        finished = true;
        send('error', err.message);
    });

    proc.on('close', (code) => {
        if (finished) return;
        finished = true;
        if (code == 0 && snapshot_dir)
            send('success', snapshot_dir);
        else
            send('error', last_line(errors).slice(-300) || ("download failed (exit code " + code + ")"));
    });
})


// Family/type of a local .safetensors file or diffusers folder. Nothing is converted or copied.
ipcMain.handle('inspect_model', async (event, model_path) => {
    const fs = require('fs');
    let is_dir;
    try {
        is_dir = fs.statSync(model_path).isDirectory();
    } catch {
        return { success: false, error: "file not found" };
    }

    let is_supported = is_dir ? (fs.existsSync(path.join(model_path, "model_index.json")) || fs.existsSync(path.join(model_path, "config.json")))
                              : model_path.toLowerCase().endsWith(".safetensors");
    if (!is_supported)
        return { success: false, error: "only .safetensors files or diffusers model folders can be imported (.ckpt is not supported)" };

    return await new Promise(resolve => {
        let proc = spawn_backend_cmd(["inspect_model", model_path]);
        let out = "";
        let errors = "";
        proc.stdout.on('data', (data) => { out += data });
        proc.stderr.on('data', (data) => { errors += data });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
        proc.on('close', (code) => {
            let json_line = out.split("\n").filter(l => l.trim().startsWith("{")).pop();
            try {
                if (code == 0 && json_line)
                    return resolve({ success: true, info: JSON.parse(json_line) });
            } catch (err) {
                console.log(err)
            }
            resolve({ success: false, error: last_line(errors) || "could not read the model" });
        });
    });
})


ipcMain.on('get_total_ram_gb', (event) => {
    event.returnValue = Math.round(require('os').totalmem() / 2**30);
})



ipcMain.handle('run_realesrgan', async (event, arg) => {
    const result = await new Promise(resolve => run_realesrgan( arg , resolve));
    return result
})

// ipcRenderer.invoke('run_realesrgan', '/Users/divamgupta/Downloads/333.png' ).then((result) => {
//     alert(result)
//   })






ipcMain.on('get_assets_dir', (event, arg) => {
    const path = require('path');
    const fs = require('fs');
    const homedir = require('os').homedir();
    let assets_path = path.join(homedir , ".diffusionbee" , "downloaded_assets");

    if (!fs.existsSync(assets_path)) {
        fs.mkdirSync(assets_path, { recursive: true });
    }

    event.returnValue = assets_path;
});



// Plain file download (md5-checked by AssetsManager). Success carries the md5 of the downloaded bytes.
ipcMain.on('download-file', async (event, url, dest, downloadId) => {
    const fs = require('fs');
    const send = (fn, msg) => {
        try {
            event.sender.send(`to_download`, {fn: fn , download_id: downloadId , msg: msg });
        } catch (err) {
            console.log(err)
        }
    }

    const hash = require('crypto').createHash('md5');
    // 20s without any data aborts, like request's timeout (a whole-download timeout would kill big files)
    const ctrl = new AbortController();
    let timer;
    const kick = () => {
        clearTimeout(timer);
        timer = setTimeout(() => ctrl.abort(new Error("download timed out")), 20000);
    }

    try {
        kick();
        // net.fetch, not Node's fetch: Node 24's undici throws an uncaught assert(!this.paused) when a connection-close
        // server ends the socket while the file write is applying backpressure, which pops a main-process error dialog
        const res = await net.fetch(url, { signal: ctrl.signal }); // follows redirects
        if (!res.ok)
            throw new Error(`download failed (HTTP ${res.status})`);
        const totalBytes = parseInt(res.headers.get('content-length'), 10);
        let downloadedBytes = 0;
        await require('stream/promises').pipeline(res.body, async function* (chunks) {
            for await (const chunk of chunks) {
                kick();
                downloadedBytes += chunk.length;
                hash.update(chunk);
                send('progress', Math.round((downloadedBytes / totalBytes) * 100));
                yield chunk;
            }
        }, fs.createWriteStream(dest));
        send('success', hash.digest('hex'));
    } catch (err) {
        fs.unlink(dest, () => {});
        send('error', err.message);
    } finally {
        clearTimeout(timer);
    }
});


console.log("native functions imported")


export { bind_window_native_functions }