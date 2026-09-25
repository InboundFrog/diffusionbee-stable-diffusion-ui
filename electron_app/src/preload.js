// src/preload.js
// Runs sandboxed with contextIsolation: only plain functions/data cross the bridge.

import { contextBridge, ipcRenderer, webUtils } from 'electron'

// Narrow shim with the same call signatures the renderer already uses.
// (Exposing the ipcRenderer object itself no longer works over contextBridge.)
contextBridge.exposeInMainWorld('ipcRenderer', {
    send: (channel, ...args) => ipcRenderer.send(channel, ...args),
    sendSync: (channel, ...args) => ipcRenderer.sendSync(channel, ...args),
    invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
    // the real IpcRendererEvent can't cross the bridge; listeners get (event-stub, ...args)
    on: (channel, fn) => {
        const listener = (e, ...args) => fn({ senderId: e.senderId }, ...args)
        ipcRenderer.on(channel, listener)
        return () => ipcRenderer.removeListener(channel, listener)
    },
})

// filesystem path of a dropped File (File.path is gone since Electron 32)
contextBridge.exposeInMainWorld('get_path_for_file', (file) => webUtils.getPathForFile(file))

var bind_ipc_renderer_on_fn = undefined;
var bind_ipc_download_on_fns = {}

function bind_ipc_renderer_on(fn) {
    bind_ipc_renderer_on_fn = fn;
}

contextBridge.exposeInMainWorld('bind_ipc_renderer_on', bind_ipc_renderer_on)

ipcRenderer.on("to_renderer", (e, data) => { // the msg channel which is used for electron to send msges to browser / renderer
    if (bind_ipc_renderer_on_fn)
        bind_ipc_renderer_on_fn(data)
});


function bind_ipc_download_on( download_id,  fn_progress , fn_success, fn_error ) {
    bind_ipc_download_on_fns[download_id] = {
        "progress" : fn_progress,
        "success" : fn_success,
        "error" : fn_error,
    }
}

contextBridge.exposeInMainWorld('bind_ipc_download_on', bind_ipc_download_on)


function unbind_ipc_download_on( download_id ){
    bind_ipc_download_on_fns[download_id] = undefined;
}



contextBridge.exposeInMainWorld('unbind_ipc_download_on', unbind_ipc_download_on)


ipcRenderer.on("to_download", (e, data) => { // the msg channel which is used for electron to send msges to browser / download
    if(!bind_ipc_download_on_fns[data.download_id]){
        console.log("no fn di "+ data.download_id)
        return
    }
    if(!bind_ipc_download_on_fns[data.download_id][data.fn]){
        console.log("no fn d ")
        return
    }
    bind_ipc_download_on_fns[data.download_id][data.fn](data.msg)
});
