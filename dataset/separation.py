import os
import json
import tqdm
import math
import argparse
import vad
import audio
import functools
import numpy as np
import multiprocessing as mp


def slice_audio(dataset_path: str):
	dataset_folder = os.path.dirname(dataset_path)
	mix_folder = os.path.join(dataset_folder, 'mix')
	os.makedirs(mix_folder, exist_ok = True)
	spk1_folder = os.path.join(dataset_folder, 'spk1')
	os.makedirs(spk1_folder, exist_ok = True)
	spk2_folder = os.path.join(dataset_folder, 'spk2')
	os.makedirs(spk2_folder, exist_ok = True)
	with open(dataset_path) as dataset_file:
		for line in dataset_file:
			utterance = json.loads(line)
			signal, _ = audio.read_audio(utterance['audio_path'], mono=False)


def make_separation_dataset(input_path: str, output_path: str, sample_rate: int, utterance_duration: float, vad_type: str, device: str, processes: int, stride: int):
	if os.path.isdir(input_path):
		audio_files = [os.path.join(input_path, audio_name) for audio_name in os.listdir(input_path)]
	else:
		audio_files = [input_path]

	if vad_type == 'simple':
		_vad = vad.PrimitiveVAD(device = device)
	elif vad_type == 'webrtc':
		_vad = vad.WebrtcVAD(sample_rate = sample_rate)
	else:
		raise RuntimeError(f'VAD for type {vad_type} not found.')

	os.makedirs(os.path.join(output_path, 'mix'), exist_ok = True)
	os.makedirs(os.path.join(output_path, 'spk1'), exist_ok = True)
	os.makedirs(os.path.join(output_path, 'spk2'), exist_ok = True)

	if processes > 0:
		parametrized_generate = functools.partial(generate_utterances, output_path=output_path, sample_rate=sample_rate, vad=_vad, utterance_duration=utterance_duration, stride=stride)
		with mp.Pool(processes=processes) as pool:
			list(tqdm.tqdm(pool.imap(parametrized_generate, audio_files), total=len(audio_files)))
	else:
		for audio_path in tqdm.tqdm(audio_files):
			generate_utterances(audio_path, output_path, sample_rate, _vad, utterance_duration, stride)

def generate_utterances(audio_path: str, output_path: str, sample_rate: int, vad, utterance_duration: float, stride: int):
	signal, _ = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = vad.required_type, __array_wrap__ = vad.required_wrapper)
	speaker_masks = vad.detect(signal, keep_intersections = True)
	utterance_duration = math.ceil(utterance_duration * sample_rate)
	assert utterance_duration % stride == 0

	## https://habr.com/ru/post/489734/#1d
	sliding_window = np.lib.stride_tricks.as_strided(speaker_masks,
	                                                 shape = (speaker_masks.shape[0], int((speaker_masks.shape[-1] - utterance_duration) / stride) + 1, utterance_duration,),
	                                                 strides = (speaker_masks.strides[0], stride, 1))
	n_samples_by_speaker = sliding_window.sum(-1)
	# speaker ratio in range [0;1] - silence ratio in range [0;1]
	utterance_scores = n_samples_by_speaker[1:].min(0)/(n_samples_by_speaker[1:].max(0) + 1) - n_samples_by_speaker[0]/utterance_duration

	n = 0
	audio_name, extension = os.path.splitext(os.path.basename(audio_path))
	while utterance_scores.max() > 0.25:
		i = np.argmax(utterance_scores)
		utterance = signal[:, i * stride : i * stride + utterance_duration]
		utterance_scores[max(0, i - int(utterance_duration/stride) + 1): i + int(utterance_duration/stride)] = 0.0
		audio.write_audio(os.path.join(output_path, 'mix', f'{audio_name}.{n}{extension}'), utterance.T, sample_rate, mono = True)
		audio.write_audio(os.path.join(output_path, 'spk1', f'{audio_name}.{n}{extension}'), utterance[0:1, :].T, sample_rate, mono = True)
		audio.write_audio(os.path.join(output_path, 'spk2', f'{audio_name}.{n}{extension}'), utterance[1:2, :].T, sample_rate, mono = True)
		n += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-path', '-i', required=True)
	parser.add_argument('--output-path', '-o', required=True)
	parser.add_argument('--sample-rate', type = int, default = 8_000)
	parser.add_argument('--duration', dest = 'utterance_duration', type = float, default = 15.0)
	parser.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc'], default = 'webrtc')
	parser.add_argument('--device', type = str, default = 'cpu')
	parser.add_argument('--processes', type = int, default = 0)
	parser.add_argument('--stride', type = int, default = 1000)

	args = vars(parser.parse_args())
	make_separation_dataset(**args)
