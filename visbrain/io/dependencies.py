"""Test if dependencies are installed."""

__all__ = ('is_mne_installed', 'is_nibabel_installed', 'is_opengl_installed')


def is_mne_installed():
    """Test if MNE is installed."""
    try:
        import mne
        return True
    except:
        return False


def is_nibabel_installed():
    """Test if nibabel is installed."""
    try:
        import nibabel
        return True
    except:
        return False


def is_opengl_installed():
    """Test if open gl is installed."""
    try:
        import OpenGL.GL as GL
        return True
    except:
        return False
