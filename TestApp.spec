# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['test_build.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\lombi\\anaconda3\\envs\\midi-composer\\Lib\\site-packages\\PyQt6\\Qt6\\plugins\\platforms\\*', 'PyQt6\\Qt6\\plugins\\platforms'), ('C:\\Users\\lombi\\anaconda3\\envs\\midi-composer\\Lib\\site-packages\\PyQt6\\Qt6\\plugins\\styles\\*', 'PyQt6\\Qt6\\plugins\\styles')],
    hiddenimports=['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TestApp',
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
    name='TestApp',
)
