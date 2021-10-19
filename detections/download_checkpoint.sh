#!/bin/bash

DIR=checkpoints
FILE=detr-r50-e632da11.pth

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

python << END
import torch
m = torch.load('detr-r50-e632da11.pth', map_location='cpu')
torch.save(dict(model_state_dict=m['model']), 'detr-r50-e632da11.pth')
END

mv $FILE $DIR/

echo "Done."
