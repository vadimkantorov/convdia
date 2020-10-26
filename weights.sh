HUBMODEL=https://raw.githubusercontent.com/pyannote/pyannote-audio-hub/master/models/emb_voxceleb.zip

wget $HUBMODEL
unzip $(basename $HUBMODEL)
