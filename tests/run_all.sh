#!/usr/bin/env bash
# set -e
# docker pull pyprob/pyprob_cpp
pytest -rA --cov=./ --cov-report xml
