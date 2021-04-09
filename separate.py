import os
import audio
import json
import argparse
import transcripts
from tqdm.auto import tqdm

def main(args):
	with open(args.diarization_dataset) as data_file:
		for line in tqdm(data_file):
			example = json.loads(line)
			mask = transcripts.intervals_to_mask(example.pop('intervals'), example['sample_rate'], example['duration']).numpy()
			path, ext = os.path.splitext(example['audio_path'])
			signal, sample_rate = audio.read_audio(path + '_mix' + ext, mono=True)
			speaker_1 = signal[:, :mask.shape[-1]] * mask[1, :signal.shape[-1]]
			audio.write_audio(path + '_s1' + ext, speaker_1.T, sample_rate)
			speaker_2 = signal[:, :mask.shape[-1]] * mask[2, :signal.shape[-1]]
			audio.write_audio(path + '_s2' + ext, speaker_2.T, sample_rate)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--diarization-dataset', '-i', required=True)
	args = parser.parse_args()

	main(args)
