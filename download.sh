#!/bin/bash

FILE=hico_20160224_det.tar.gz
EXTR=hico_20160224_det
ID=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk

if [ -d $EXTR ]; then
  echo "$EXTR already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Extracting..."

tar zxf $FILE
rm $FILE

echo "Done."
