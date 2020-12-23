#!/bin/bash

DIR=checkpoints
FILE=fasterrcnn_resnet50_fpn_hicodet_e13.pt
ID=11lS2BQ_In-22Q-SRTRjRQaSLg9nSim9h

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

mv $FILE $DIR/

echo "Done."
