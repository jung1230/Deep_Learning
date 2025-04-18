#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from ComputationalGraphPrimer.ComputationalGraphPrimer import __version__
    from ComputationalGraphPrimer.ComputationalGraphPrimer import __author__
    from ComputationalGraphPrimer.ComputationalGraphPrimer import __date__
    from ComputationalGraphPrimer.ComputationalGraphPrimer import __url__
    from ComputationalGraphPrimer.ComputationalGraphPrimer import __copyright__
    from ComputationalGraphPrimer.ComputationalGraphPrimer import ComputationalGraphPrimer
    from ComputationalGraphPrimer.ComputationalGraphPrimer import SGDPlus_ComputationalGraphPrimer
    from ComputationalGraphPrimer.ComputationalGraphPrimer import SGD_ComputationalGraphPrimer
    from ComputationalGraphPrimer.ComputationalGraphPrimer import Adam_ComputationalGraphPrimer

else:
    from ComputationalGraphPrimer import __version__
    from ComputationalGraphPrimer import __author__
    from ComputationalGraphPrimer import __date__
    from ComputationalGraphPrimer import __url__
    from ComputationalGraphPrimer import __copyright__
    from ComputationalGraphPrimer import ComputationalGraphPrimer
    from ComputationalGraphPrimer.ComputationalGraphPrimer import SGDPlus_ComputationalGraphPrimer






