"""
ATLAS Dependency Manager

Handles automatic installation and import of ATLAS modules and their dependencies.
"""

import importlib.util
import sys
import logging
import threading
import subprocess

logger = logging.getLogger(__name__)


class ATLASDependencyManager:
    """Manages ATLAS module dependencies and installation."""
    
    def __init__(self):
        self._deps_ready = False
        self._deps_error = None
        self._atlas_available = False
        
    def check_atlas_availability(self):
        """Check if ATLAS modules are available in the Slicer extension path."""
        required_modules = ["BUILDER", "PREDICT", "DATABASE"]
        missing_modules = []
        
        for module_name in required_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    missing_modules.append(module_name)
            except (ImportError, ModuleNotFoundError):
                missing_modules.append(module_name)
        
        self._atlas_available = len(missing_modules) == 0
        
        if not self._atlas_available:
            logger.warning(f"ATLAS modules not found: {', '.join(missing_modules)}")
            logger.info("Please install ATLAS extension from 3D Slicer Extension Manager")
            logger.info("or add ATLAS repository to Slicer's module path")
        
        return self._atlas_available, missing_modules
    
    def check_python_dependencies(self):
        """Check if required Python packages are installed."""
        required_packages = [
            ("tiny3d", "tiny3d"),
            ("biocpd", "biocpd"),
            ("scipy.spatial", "scipy"),
            ("scipy.optimize", "scipy")
        ]
        
        missing = []
        for import_name, pip_name in required_packages:
            try:
                importlib.import_module(import_name)
            except ImportError:
                if pip_name not in [m[1] for m in missing]:
                    missing.append((import_name, pip_name))
        
        return missing
    
    def install_python_dependencies_async(self, missing_packages, on_ready=None, on_error=None):
        """
        Install missing Python packages asynchronously.
        
        Args:
            missing_packages: List of (import_name, pip_name) tuples
            on_ready: Callback function when installation completes successfully
            on_error: Callback function when installation fails (receives exception)
        """
        def worker():
            try:
                if missing_packages:
                    pip_names = list(set([pip for _, pip in missing_packages]))
                    logger.info(f"Installing Python packages: {', '.join(pip_names)}")
                    
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", *pip_names
                    ])
                    
                    # Verify installation
                    for import_name, _ in missing_packages:
                        importlib.import_module(import_name)
                
                self._deps_ready = True
                logger.info("ATLAS Python dependencies installed successfully")
                
                if callable(on_ready):
                    on_ready()
                    
            except Exception as e:
                self._deps_error = e
                logger.error(f"Failed to install ATLAS dependencies: {e}")
                
                if callable(on_error):
                    on_error(e)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def ensure_dependencies(self, confirm_install=True):
        """
        Ensure all dependencies are available, offering installation if needed.
        
        Args:
            confirm_install: Whether to prompt user before installing packages
            
        Returns:
            bool: True if all dependencies are available
        """
        # Check ATLAS modules
        atlas_ok, missing_atlas = self.check_atlas_availability()
        if not atlas_ok:
            return False
        
        # Check Python packages
        missing_py = self.check_python_dependencies()
        
        if missing_py:
            if confirm_install:
                try:
                    import slicer
                    pip_names = list(set([pip for _, pip in missing_py]))
                    msg = f"ATLAS requires these Python packages:\n{', '.join(pip_names)}\n\nInstall now?"
                    
                    if not slicer.util.confirmOkCancelDisplay(msg):
                        logger.warning("User declined package installation")
                        return False
                except ImportError:
                    # Non-Slicer environment, just install
                    pass
            
            # Install synchronously for immediate availability
            pip_names = list(set([pip for _, pip in missing_py]))
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_names])
                self._deps_ready = True
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Package installation failed: {e}")
                return False
        
        self._deps_ready = True
        return True


# Global instance
_dependency_manager = ATLASDependencyManager()


def get_dependency_manager():
    """Get the global dependency manager instance."""
    return _dependency_manager
