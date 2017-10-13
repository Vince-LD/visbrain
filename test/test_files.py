"""Test if needed files are successfully installed with visbrain."""
import os
from distutils.sysconfig import get_python_lib


def _test_file(name, path):
    try:
        # Get python path :
        full_path = [get_python_lib()] + path
        file = os.path.join(*full_path)
        assert os.path.isfile(file)
        print("Distant version passed for *" + name + "* file")
    except:
        file = os.path.join(*path)
        assert os.path.isfile(file)
        print("Local version passed for *" + name + "* file")

###############################################################################
#                                   BRAIN
###############################################################################


def test_brain_templates():
    """Test if templates are installed."""
    name = "Brain template ({})"
    path_to_template = ['visbrain', 'data', 'templates']
    for k in ['B1.npz', 'B2.npz', 'B3.npz']:
        path = path_to_template + [k]
        _test_file(name.format(k), path)


def test_roi_templates():
    """Test if templates are installed."""
    name = "ROI template ({})"
    path_to_template = ['visbrain', 'data', 'roi']
    for k in ['brodmann.npz', 'aal.npz', 'talairach.npz']:
        path = path_to_template + [k]
        _test_file(name.format(k), path)


def test_brain_icon():
    """Test if Sleep icon is installed."""
    path = ['visbrain', 'brain', 'brain_icon.svg']
    _test_file('Brain icon (brain_ico.svg)', path)

###############################################################################
#                                   SLEEP
###############################################################################


def test_sleep_icon():
    """Test if Sleep icon is installed."""
    path = ['visbrain', 'sleep', 'sleep_icon.svg']
    _test_file('Sleep icon (sleep_ico.svg)', path)

###############################################################################
#                                   TOPO
###############################################################################


def test_topo_file():
    """Test if the topo reference file is installed."""
    path = ['visbrain', 'data', 'topo', 'eegref.npz']
    _test_file('Topo reference file (eegref.npz)', path)


def test_topo_icon():
    """Test if the topo icon is installed."""
    path = ['visbrain', 'topo', 'topo_icon.svg']
    _test_file('Topo icon (topo_icon.svg)', path)


###############################################################################
#                                   COLORBAR
###############################################################################


def test_colorbar_icon():
    """Test if the Colorbar icon is installed."""
    path = ['visbrain', 'colorbar', 'colorbar_icon.svg']
    _test_file('Colorbar icon (colorbar_icon.svg)', path)


###############################################################################
#                                   FIGURE
###############################################################################


def test_figure_icon():
    """Test if the Figure icon is installed."""
    path = ['visbrain', 'figure', 'figure_icon.svg']
    _test_file('Figure icon (figure_icon.svg)', path)


###############################################################################
#                                   URL
###############################################################################


def test_data_url():
    """Test if the data_url.txt is installed."""
    path = ['visbrain', 'io', 'data_url.txt']
    _test_file('URL to data (data_url.txt)', path)
