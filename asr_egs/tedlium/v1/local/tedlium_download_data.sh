#!/bin/bash

# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

mkdir -p db
pushd db

# TED-LIUM database:
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  ln -s /export/corpora5/TEDLIUM_release1
elif [[ $(hostname -f) == compute-*.local ]]; then
  ln -s /data/MM1/corpora/TEDLIUM_release* .
else
  if [ ! -f TEDLIUM_release1.tar.gz ]; then
    wget -q http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz || exit 1
    tar xf TEDLIUM_release1.tar.gz
  fi
  if [ ! -f TEDLIUM_release2.tar.gz ]; then
    wget -q http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz || exit 1;
    tar xf TEDLIUM_release2.tar.gz
  fi
fi

# Language models (Cantab Research):
if [ ! -d cantab-TEDLIUM ]; then
    echo "Downloading \"http://cantabresearch.com/cantab-TEDLIUM.tar.bz2\". "
    wget --no-verbose --output-document=- http://cantabresearch.com/cantab-TEDLIUM.tar.bz2 | bzcat | tar --extract --file=- || exit 1
    gzip cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3
    gzip cantab-TEDLIUM/cantab-TEDLIUM-unpruned.lm4
fi

popd
