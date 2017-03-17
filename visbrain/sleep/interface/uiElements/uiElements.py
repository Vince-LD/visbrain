"""This script initialize all ui files."""

from .uiSettings import uiSettings
from .uiPanels import uiPanels
from .uiInfo import uiInfo
from .uiTools import uiTools
from .uiScoring import uiScoring
from .uiDetection import uiDetection

__all__ = ['uiElements']


class uiElements(uiSettings, uiPanels, uiInfo, uiTools, uiScoring,
                 uiDetection):
    """Inherit from the diffrent ui files and initialize them."""

    def __init__(self):
        """Init."""
        uiSettings.__init__(self)
        uiPanels.__init__(self)
        uiInfo.__init__(self)
        uiTools.__init__(self)
        uiScoring.__init__(self)
        uiDetection.__init__(self)