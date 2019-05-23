#!/bin/bash

./make_TLG_WFST.sh

./feature.sh

./train.sh

./decode.sh
