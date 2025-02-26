#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from YOLOLogic.YOLOLogic import __version__
    from YOLOLogic.YOLOLogic import __author__
    from YOLOLogic.YOLOLogic import __date__
    from YOLOLogic.YOLOLogic import __url__
    from YOLOLogic.YOLOLogic import __copyright__
    from YOLOLogic.YOLOLogic import YOLOLogic
else:
    from YOLOLogic import __version__
    from YOLOLogic import __author__
    from YOLOLogic import __date__
    from YOLOLogic import __url__
    from YOLOLogic import __copyright__
    from YOLOLogic import YOLOLogic





