# GNU -*- makefile -*-

#VERSION := ${shell python -c "import YOLOLogic;print YOLOLogic.__version__"}
VERSION := ${shell python3 -c "import os; os.chdir('YOLOLogic'); import YOLOLogic; print(YOLOLogic.__version__)"}

default:
	@echo
	@echo "  *** Welcome to YOLOLogic ${VERSION} ***"
	@echo
	@echo "  docs   -  Build documentation (html)"
	@echo "  help   -  Open the documentation"
	@echo
	@echo "  clean  -  Remove temporary files"
	@echo "  test   -  Run the unittests"
	@echo "  check  -  Look for rough spots"
	@echo "  sdist  -  Build a source distribution tar ball"

docs:
	pydoc3 -w YOLOLogic/YOLOLogic.py

help:
	open YOLOLogic-${VERSION}.html

clean:
	rm -f *.pyc *~

real-clean: clean
	rm -f MANIFEST  *.html YOLOLogic-py.info
	rm -rf build dist

# Run the unittest
test:
	@echo
	@echo Testing...
	@echo
	python3 ./TestYOLOLogic/Test.py 

sdist: test
	@echo
	@echo Building a source distribution...
	@echo
	./setup.py sdist --formats=gztar

# Look for rough spots
check:
	@grep -n FIX *.py *.in PKG-INFO Makefile | grep -v grep
	@echo
	pychecker YOLOLogic


