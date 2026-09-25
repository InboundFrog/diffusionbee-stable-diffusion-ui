let build_config = {}
try {
    build_config = require('./build_config.json');
} catch (err) {
    // optional
}

const min_os_version = build_config.min_os_version || "14.0"

module.exports = {

    pluginOptions: {
        electronBuilder: {
            preload: './src/preload.js',
            // the plugin marks packages without a "main" field as externals; bundle it so the app needs no node_modules
            externals: ['!electron-context-menu'],

            builderOptions: {
                appId: 'com.diffusionbee.diffusionbee',
                artifactName: "DiffusionBee" + (build_config.build_name || "") + "-${version}-${arch}.${ext}",

                // webpack already bundles every dependency; stop electron-builder from copying/rebuilding node_modules
                beforeBuild: async () => false,

                // Bundled python backend, only when built (access via path.join(path.dirname(__dirname), 'core')).
                extraResources: process.env.BACKEND_BUILD_PATH ? [{
                    from: process.env.BACKEND_BUILD_PATH,
                    to: "core",
                    filter: ["**/*"]
                }] : [],

                mac: {
                    icon: "build/Icon-1024.png",
                    category: "public.app-category.graphics-design",
                    hardenedRuntime: true,
                    entitlements: "build/entitlements.mac.plist",
                    entitlementsInherit: "build/entitlements.mac.plist",
                    minimumSystemVersion: min_os_version,
                    // Notarization is built into electron-builder: it runs only when APPLE_API_KEY/APPLE_API_KEY_ID/APPLE_API_ISSUER
                    // (or APPLE_ID/APPLE_APP_SPECIFIC_PASSWORD/APPLE_TEAM_ID) are set, so local unsigned builds just skip it.
                    target: {
                        target: "dmg",
                        arch: [process.env.BUILD_ARCH || 'arm64'] // 'x64' if ever needed
                    }
                },
            }
        }
    }
}
