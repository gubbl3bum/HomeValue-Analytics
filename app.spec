# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import sys
import os
import glob

hiddenimports = collect_submodules('streamlit') + [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'sklearn',
    'matplotlib',
    'seaborn',
    'scipy'
]

datas = collect_data_files('streamlit')

binaries = []
if sys.platform.startswith('win'):
    import scipy
    import numpy
    
    # Get specific DLL files instead of using wildcards
    scipy_path = os.path.dirname(scipy.__file__)
    numpy_path = os.path.dirname(numpy.__file__)
    
    # Add specific scipy DLLs
    for dll in glob.glob(os.path.join(scipy_path, "_lib", "*.dll")):
        binaries.append((dll, "."))
    
    # Add specific numpy DLLs
    for dll in glob.glob(os.path.join(numpy_path, "core", "*.dll")):
        binaries.append((dll, "."))
    
    # Add BLAS/LAPACK DLLs if present
    for dll in glob.glob(os.path.join(numpy_path, "linalg", "*.dll")):
        binaries.append((dll, "."))

block_cipher = None

a = Analysis(
    ['src/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="HomeValue-Analytics",
    debug=False,  # Changed to True for better error reporting
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None
)
