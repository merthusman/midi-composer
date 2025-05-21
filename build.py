# build.py
import PyInstaller.__main__
import os
import sys
import site
import tensorflow as tf
import logging # Import logging

# Configure basic logging for the build process itself
# This logging is separate from the application's runtime logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def build_exe():
    """
    Builds the application into a single executable using PyInstaller.
    """
    # Get the absolute path of the script (build.py is in the project root)
    script_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Build script path: {script_path}")

    # Define the path to the main script
    main_script = os.path.join(script_path, 'src', 'main.py')
    if not os.path.exists(main_script):
        logger.error(f"Main script not found: {main_script}")
        sys.exit(1) # Exit if main script doesn't exist
    logger.info(f"Main script path: {main_script}")


    # --- Determine Paths for Dependencies ---
    # Get site-packages directory of the active environment
    # This is where installed packages are located.
    try:
        site_packages_dirs = site.getsitepackages()
        # In a typical Conda env, this might return two paths, pick the one inside the env
        site_packages = next((s for s in site_packages_dirs if sys.prefix in s), site_packages_dirs[0])
        logger.info(f"Site packages directory: {site_packages}")
    except Exception as e:
        logger.warning(f"Could not determine site-packages directory: {e}")
        # Fallback or handle error


    # Get TensorFlow DLL directory
    # TensorFlow often puts necessary DLLs in its installation directory or a 'python' subdir
    tf_dll_path = None
    try:
        # Try to find the directory containing TensorFlow DLLs. This can vary.
        # Common locations are site-packages/tensorflow/.libs or site-packages/tensorflow/python/dist-packages/tensorflow/.libs
        # Let's try a few common paths relative to the tensorflow package directory
        tf_package_path = os.path.dirname(tf.__file__)
        possible_dll_paths = [
            os.path.join(tf_package_path, '.libs'), # Common location
            os.path.join(tf_package_path, 'python', 'dist-packages', 'tensorflow', '.libs'), # Another possible location
            os.path.join(tf_package_path, 'python'), # Sometimes DLLs are directly in python dir
             tf_package_path # Check the package root itself
        ]
        for path in possible_dll_paths:
             if os.path.exists(path) and any(f.endswith('.dll') for f in os.listdir(path)):
                  tf_dll_path = path
                  logger.info(f"Found TensorFlow DLL path: {tf_dll_path}")
                  break

        if tf_dll_path is None:
             logger.warning("Could not automatically find TensorFlow DLL directory. PyInstaller might need manual help.")
             # Proceed anyway, PyInstaller might find them automatically or fail.
             # You might need to manually specify the path to TensorFlow DLLs using --add-binary

    except ImportError:
        logger.error("TensorFlow is not installed in the active environment. Cannot find TensorFlow DLLs.")
        # Decide if you want to exit or continue build without TF
        # If TF is essential, you should exit.
        # sys.exit(1) # Uncomment to exit if TF is mandatory


    # --- Define PyInstaller Arguments ---
    args = [
        main_script,
        '--name=MIDI_Composer', # Name of the executable
        '--onefile', # Package into a single executable file
        '--windowed', # Create a windowed application (no console window)
        '--icon=resources/icon.ico', # Application icon
        # Add data files/directories (relative to project root)
        '--add-data=resources;resources', # Adds the 'resources' directory and its contents
        '--add-data=src/gui/styles.qss;src/gui', # Add the QSS style file

        # Add necessary DLLs and dependencies that PyInstaller might not find automatically
        # This is often needed for libraries with native components like TensorFlow and Qt (used by PyQt6)
        # If tf_dll_path was found, add its contents.
        # The destination '.' means add to the root of the bundle.
        # The format is SourcePath;DestinationPath
        # SourcePath is relative to the build.py script OR absolute.
        # DestinationPath is relative to the bundle's root.
        # If tf_dll_path is not None, add its DLLs
        # NOTE: PyInstaller's --add-binary format can be tricky. It often works best with
        # absolute paths for sources or paths relative to the spec file directory.
        # Using f'--add-binary={tf_dll_path};.' might work if tf_dll_path is absolute.
        # A more reliable way for DLLs is sometimes collecting the package using --collect-all
        # or --collect-submodules, or explicitly adding known DLL locations relative to site-packages.

        # Let's rely more on --collect-all and --hidden-import first, and if build fails,
        # manually add DLLs based on error messages or known locations.

        # Collect modules that PyInstaller might miss (often needed for dynamic imports)
        # --collect-all includes the package and its submodules, often pulling in native libs
        '--collect-all=tensorflow',
        '--collect-all=pretty_midi',
        '--collect-all=music21',
        '--collect-all=PyQt6',
        '--collect-all=numpy',
        '--collect-all=pandas',
        '--collect-all=scikit-learn',
        '--collect-all=matplotlib', # Matplotlib might need fonts/data files too, collect-all helps.

        # Hidden imports for modules that are imported dynamically or in ways PyInstaller doesn't detect
        '--hidden-import=tensorflow', # Often needed even with --collect-all
        '--hidden-import=pretty_midi',
        '--hidden-import=music21',
        '--hidden-import=PyQt6',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=scikit-learn',
        '--hidden-import=matplotlib',
        # Add hidden imports for specific submodules if the build warns about missing ones
        # '--hidden-import=tensorflow.python.ops.array_ops', # Example from original file
        # '--hidden-import=tensorflow.python.framework.op_def_library', # Example
        # '--hidden-import=pretty_midi.instrument', # Example
        # '--hidden-import=music21.stream', # Example

        # Options for reducing size or improving performance (optional)
        # '--strip', # Remove debug symbols (can reduce size)
        # '--clean', # Clean PyInstaller cache and remove temporary files
        # '--distpath ./dist', # Specify output directory (defaults to ./dist)
        # '--workpath ./build', # Specify build directory (defaults to ./build)

        # Add --add-binary for specific DLLs if --collect-all is insufficient
        # Example (replace with actual paths if needed after build failure):
        # f'--add-binary={os.path.join(site_packages, "scipy/.libs/*.dll")};.', # Example for SciPy DLLs
        # f'--add-binary={os.path.join(site_packages, "numpy.libs/*.dll")};.', # Example for NumPy DLLs
        # f'--add-binary={os.path.join(site_packages, "PyQt6/Qt6/bin/*.dll")};.', # Example for PyQt6 Qt DLLs if not collected


    ]

    logger.info("Running PyInstaller with arguments:")
    for arg in args:
        logger.info(f"  {arg}")

    # Run PyInstaller
    PyInstaller.__main__.run(args)

    logger.info("PyInstaller build process finished.")
    logger.info("Check the 'dist' directory for the executable.")


if __name__ == '__main__':
    # Ensure we are in the directory containing build.py
    # This should be true if running python build.py from the project root
    # build_script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(build_script_dir)
    # logger.info(f"Current working directory set to: {os.getcwd()}") # Should be project root

    build_exe()