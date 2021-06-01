#!/bin/bash
# Download the data collected & used in ShapeGlot (~218MB)
# We assume you have already accepted the Terms Of Use, else please visit: https://forms.gle/2cd4U9zdBH7r9PyTA

#DATA_LINK=<REPLACE WITH LINK PROVIDED after you accepted the terms of use of ShapeGlot>
wget $DATA_LINK
unzip data.zip
rm data.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=171Rs5BC_ZeaA8tO27HVznpBTpNLk6ZF7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=171Rs5BC_ZeaA8tO27HVznpBTpNLk6ZF7" -O ./data/shapenet_chairs_only_in_game.h5 && rm -rf /tmp/cookies.txt
