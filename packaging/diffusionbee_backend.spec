# PyInstaller spec for the DiffusionBee backend (macOS arm64, --onedir).
# Output: <distpath>/diffusionbee_backend/{diffusionbee_backend,_internal/}
# That folder is BACKEND_BUILD_PATH; electron-builder copies it to Resources/core/.
# Build via ../build_mac.sh, or:
#   pyinstaller --noconfirm --distpath DIST --workpath WORK packaging/diffusionbee_backend.spec
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
ENTRY = os.path.join(ROOT, 'backends', 'stable_diffusion', 'diffusionbee_backend.py')

# diffusers/transformers/peft resolve most classes lazily by string (_LazyModule), so static
# analysis misses them; collect everything. Their import-time `importlib.metadata.version(...)`
# checks need the .dist-info of each dependency too. mlx: collect_all also picks up libmlx.dylib and mlx.metallib.
LAZY_PKGS = ['diffusers', 'transformers', 'huggingface_hub', 'tokenizers', 'safetensors',
             'accelerate', 'peft', 'sentencepiece', 'mlx', 'mflux']
METADATA = ['torch', 'numpy', 'Pillow', 'regex', 'requests', 'tqdm', 'filelock', 'packaging',
            'pyyaml', 'protobuf', 'opencv-python-headless', 'onnxruntime', 'psutil', 'jinja2']

datas, binaries, hiddenimports = [], [], []
for pkg in LAZY_PKGS:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
for dist in METADATA:
    datas += copy_metadata(dist)
# torch.compile on MPS inlines these into its Metal kernels; the torch hook leaves headers out
datas += collect_data_files('torch', includes=['include/c10/metal/*.h'])

a = Analysis(
    [ENTRY],
    pathex=[os.path.dirname(ENTRY)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(SPECPATH, 'rthook_code_filenames.py')],  # torch.compile, see the hook
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'tensorflow', 'jax', 'flax', 'triton'],
    # Keep .py sources next to the bytecode: torch/diffusers/transformers call inspect.getsource.
    module_collection_mode={p: 'pyz+py' for p in LAZY_PKGS},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='diffusionbee_backend',
    console=True,
    target_arch='arm64',
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name='diffusionbee_backend', upx=False)
