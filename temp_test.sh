#!/bin/bash
python setup.py bdist_wheel
pip install dist/presumm-1.0.0-py3-none-any.whl
python -c "import presumm; print('Fully imported!')"
pip uninstall presumm
pip list | grep presumm
