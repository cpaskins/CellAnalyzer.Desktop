# -*- mode: python ; coding: utf-8 -*-
# Run from the Python/ directory:
#   venv\Scripts\pyinstaller CellAnalyzerEngine.spec --distpath Engine

from PyInstaller.utils.hooks import collect_data_files

# Plotly needs its bundled templates and package data at runtime
datas = collect_data_files('plotly')

a = Analysis(
    ['cli.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # scipy – connected-component labelling used in basic_contours
        'scipy.ndimage',
        'scipy.ndimage._ni_support',
        'scipy.ndimage._interpolation',
        # scikit-image modules used in scikit_contours / basic_contours
        'skimage.io',
        'skimage.color',
        'skimage.measure',
        'skimage.filters',
        'skimage.morphology',
        # narwhals is a required runtime dep of plotly.express
        'narwhals',
        'narwhals.stable',
        'narwhals.stable.v1',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'matplotlib.pyplot', 'mpl_toolkits'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CellAnalyzerEngine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CellAnalyzerEngine',
)
