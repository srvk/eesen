#!/bin/bash
mkdir temp

for i in *.wav

do 
sox -V $i -t .wav -r 16000 ./temp/$(basename $i)
done
