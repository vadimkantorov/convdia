export CUDA_VISIBLE_DEVICES=2
DATASET=dia_dataset/diacalls_short_2020-09-01_2020-10-01

python3 diarization.py ref $@ -i $DATASET -o $DATASET.ref --html --ext .wav

#python3 tools.py cat -i $DATASET.ref -o $DATASET.ref.json

#python3 diarization.py ref $@ -i diarization/stereo -o data/diarization/ref --html
#python3 diarization.py hyp $@ -i diarization/mono -o data/diarization/hyp --html
#python3 diarization.py eval --ref data/diarization/ref --hyp data/diarization/hyp #--audio

#python3 diarization.py ref -i diarization/stereo/00d13c16-ac0d-409c-8a5e-36741a9e750a.mp3 -o data/diarization_test
#python3 diarization.py hyp -i diarization/mono/00d13c16-ac0d-409c-8a5e-36741a9e750a.mp3.wav -o data/diarization_test/

