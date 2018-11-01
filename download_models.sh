#!/bin/bash

mkdir -p model

cd model
if [ ! -f model.zip ] && [ "$MODEL_URL" != "" ]
then
  wget "${MODEL_URL}" -O model.zip
  [ -f model.zip ] && unzip -n model.zip
fi

