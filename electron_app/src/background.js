'use strict'


import { app, protocol, net, shell, dialog, BrowserWindow, nativeTheme, Menu } from 'electron'
import contextMenu from 'electron-context-menu'
import settings from 'electron-settings';
import { pathToFileURL } from 'url'
const isDevelopment = process.env.NODE_ENV !== 'production'


import { start_bridge, bind_window_bridge } from './bridge.js'
import { bind_window_native_functions } from "./native_functions.js"

start_bridge();


const path = require('path');

let win;

// Scheme must be registered before the app is ready
protocol.registerSchemesAsPrivileged([
	{ scheme: 'app', privileges: { secure: true, standard: true, supportFetchAPI: true } }
])


import {menu_template} from "./menu_template"
Menu.setApplicationMenu(Menu.buildFromTemplate(menu_template))


const is_web_url = (url) => /^https?:\/\//i.test(url)


function save_window_size() {
	if( ! win.savable )
		return;
	let windowState = win.getBounds();
	windowState.isMaximized = win.isMaximized();
	settings.set('windowPosState', windowState);
}

contextMenu({
	showSaveImageAs: true
});

async function createWindow() {
	// Create the browser window.
	win = new BrowserWindow({
		width: 770,
		height: 550,
		minWidth: 770,
		minHeight: 550,
		titleBarStyle : 'hidden',
		maximizable : false,
		trafficLightPosition: { x: 18, y: 20 },
		webPreferences: {
			// ponytail: renderer shows local generated images via file:// URLs, which needs webSecurity off;
			// serve them through a custom protocol to drop this.
			webSecurity: false,
			nodeIntegration: false,
			contextIsolation: true,
			sandbox: true,
			preload: path.join(__dirname, 'preload.js'),
		}
	});

	// save the window state on resize , move, etc
	['resize', 'move'].forEach(event => {
		win.on(event, save_window_size);
	});


	win.on('close', function(e) {
		if(win.show_dialog_on_quit){

			let message = 'Are you sure you want to quit?';
			if(win.dialog_on_msg)
				message = win.dialog_on_msg;

			const choice = dialog.showMessageBoxSync(win, {
				type: 'question',
				buttons: ['Yes', 'No'],
				title: 'Confirm',
				message: message
			});
			if (choice === 1) {
				e.preventDefault();
			}
		}
	});

	// window.open: http(s) goes to the system browser; data: popups (image viewer in utils.open_popup) stay in-app.
	win.webContents.setWindowOpenHandler(({ url, features }) => {
		if (url.startsWith('data:')) {
			// Chromium blocks renderer-initiated data: navigations, so the main process opens it.
			// No preload; webSecurity off only so the page can show the file:// image.
			new BrowserWindow({ x: 100, y: 100, frame: !features.includes('frame=false'), webPreferences: { webSecurity: false, sandbox: true } }).loadURL(url);
			return { action: 'deny' };
		}
		if (is_web_url(url))
			shell.openExternal(url);
		return { action: 'deny' };
	});

	// Never navigate the app window away from the app.
	win.webContents.on('will-navigate', (e, url) => {
		if (new URL(url).origin === new URL(win.webContents.getURL()).origin)
			return;
		e.preventDefault();
		if (is_web_url(url))
			shell.openExternal(url);
	});

	nativeTheme.themeSource = 'system';

	if (process.env.WEBPACK_DEV_SERVER_URL) {
		// Load the url of the dev server if in development mode
		await win.loadURL(process.env.WEBPACK_DEV_SERVER_URL)
		if (!process.env.IS_TEST) win.webContents.openDevTools()
	} else {
		// Load the index.html when not in development
		win.loadURL('app://./index.html')
	}
}

// Serve the bundled renderer from app:// (replaces the plugin's deprecated registerBufferProtocol helper).
function register_app_protocol() {
	protocol.handle('app', (req) => {
		// host is part of the path: CSS urls come out as app:///img/x.png, which Chromium turns into app://img/x.png
		const { host, pathname } = new URL(req.url)
		const file = path.join(__dirname, host, decodeURIComponent(pathname))
		if (!file.startsWith(__dirname + path.sep))
			return new Response('Not found', { status: 404 })
		return net.fetch(pathToFileURL(file).toString()).catch((err) => {
			console.error(`app:// ${req.url}: ${err.message}`)
			return new Response('Not found', { status: 404 })
		})
	})
}


app.on('activate', () => {
	// On macOS it's common to re-create a window in the app when the
	// dock icon is clicked and there are no other windows open.
	if (BrowserWindow.getAllWindows().length === 0) createWindow()
})




// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', async () => {
	if (!process.env.WEBPACK_DEV_SERVER_URL)
		register_app_protocol();
	createWindow();

	bind_window_bridge(win);

	win.webContents.on('did-finish-load', function() {

		bind_window_native_functions(win);
	});

})

// set the about panel
app.setAboutPanelOptions({
	applicationName: require('../package.json').name,
	applicationVersion: require('../package.json').version,
	version: require('../package.json').build_number,
	credits: require('../package.json').description,
	copyright: "Copyright © 2023 " + require('../package.json').name,
	website: require('../package.json').website
});





// Exit cleanly on request from parent process in development mode.
if (isDevelopment) {
	process.on('SIGTERM', () => {
		app.quit()
	})
}
