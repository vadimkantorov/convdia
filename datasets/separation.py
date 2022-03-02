import os
import tqdm
import math
import argparse
import vad
import audio
import functools
import torch
import numpy as np
import multiprocessing as mp
from typing import Callable

def make_separation_dataset(input_path: str, output_path: str, sample_rate: int, utterance_duration: float, vad_type: str, num_workers: int, stride: int, min_utterance_score: float, mode: str):
	if os.path.isdir(input_path):
		audio_files = [os.path.join(input_path, audio_name) for audio_name in os.listdir(input_path)]
	else:
		audio_files = [input_path]

	if vad_type == 'simple':
		_vad = vad.SimpleVAD()
	elif vad_type == 'webrtc':
		_vad = vad.WebrtcVAD(sample_rate = sample_rate)
	elif vad_type == 'silero':
		_vad = vad.SileroVAD(sample_rate = sample_rate)
	else:
		raise RuntimeError(f'VAD for type {vad_type} not found.')

	os.makedirs(os.path.join(output_path, 'mix'), exist_ok = True)
	os.makedirs(os.path.join(output_path, 'spk1'), exist_ok = True)
	os.makedirs(os.path.join(output_path, 'spk2'), exist_ok = True)

	if mode == 'balanced':
		score_func = balanced_score
	elif mode == 'single':
		score_func = single_speaker_score
	else:
		raise RuntimeError(f'Unexpected mode: "{mode}"')

	if num_workers > 0:
		parametrized_generate = functools.partial(generate_utterances, output_path=output_path, sample_rate=sample_rate, vad=_vad, utterance_duration=utterance_duration, stride=stride, min_utterance_score=min_utterance_score, score_func=score_func)
		with mp.Pool(processes=num_workers) as pool:
			list(tqdm.tqdm(pool.imap(parametrized_generate, audio_files), total=len(audio_files)))
	else:
		for audio_path in tqdm.tqdm(audio_files):
			generate_utterances(audio_path, output_path, sample_rate, _vad, utterance_duration, stride, min_utterance_score, score_func=score_func)

def balanced_score(n_samples_by_speaker, utterance_duration):
	# Use this score if you want to extract utterances with two speakers
	# speakers ratio in range [0;1] - silence ratio in range [0;1]
	return n_samples_by_speaker[1:].min(0)/(n_samples_by_speaker[1:].max(0) + 1) - n_samples_by_speaker[0]/utterance_duration

def single_speaker_score(n_samples_by_speaker, utterance_duration):
	# Use this score if you want to extract utterances with only one speaker
	# make primary speaker score as ratio between samples with longest speech in one channel and utterance duration
	primary_speaker_score = n_samples_by_speaker[1:].max(0) / utterance_duration
	# make secondary speaker penalty, this penalty will be smaller than -1 if we have more than one speaker in utterance. Used to make total score below zero in this case.
	secondary_speaker_penalty = n_samples_by_speaker[1:].min(0) * (-1.0)
	return np.clip(primary_speaker_score + secondary_speaker_penalty, 0.0, 1.0)

def generate_utterances(audio_path: str, output_path: str, sample_rate: int, vad, utterance_duration: float, stride: int, min_utterance_score: float, score_func: Callable):
	signal, _ = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = vad.input_dtype, __array_wrap__ = vad.input_type)
	# speaker_masks - array, shape [3, n_samples], speaker_masks[0] - is true if nobody talks, speaker_masks[1] - is true if somebody talks in first channel, speaker_masks[2] - is true if somebody talks in second channel.
	speaker_masks = vad.detect(signal, allow_overlap = True)
	if vad.input_type == torch.tensor:
		speaker_masks = speaker_masks.cpu().numpy()
	# utterance_duration – required utterance duration in samples
	utterance_duration = math.ceil(utterance_duration * sample_rate)
	assert utterance_duration % stride == 0

	# sliding_window - array, shape [3, n_utterances, utterance_duration], mask over utterances selected by sliding window with size utterance_duration and stride args.stride
	# for each utterance, sliding_window[0, n_utterance] - is true if nobody talks, sliding_window[1, n_utterance]  - is true if somebody talks in first channel, sliding_window[2, n_utterance]  - is true if somebody talks in second channel.
	## https://habr.com/ru/post/489734/#1d
	sliding_window = np.lib.stride_tricks.as_strided(speaker_masks,
	                                                 shape = (speaker_masks.shape[0], int((speaker_masks.shape[-1] - utterance_duration) / stride) + 1, utterance_duration,),
	                                                 strides = (speaker_masks.strides[0], stride, 1))
	# n_samples_by_speaker – array, shape [3, n_utterances], contains amount of samples with speech of each speaker
	# n_samples_by_speaker[0, n_utterances] – amount of samples where nobody talks, n_samples_by_speaker[1, n_utterances] – amount of samples where speaker talks in first channel, n_samples_by_speaker[2, n_utterances] – amount of samples where speaker talks in second channel.
	n_samples_by_speaker = sliding_window.sum(-1)
	# speakers ratio in range [0;1] - silence ratio in range [0;1]
	utterance_scores = score_func(n_samples_by_speaker, utterance_duration)

	n = 0
	audio_name, extension = os.path.splitext(os.path.basename(audio_path))
	while utterance_scores.max() > min_utterance_score:
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
	parser.add_argument('--mode', choices=['balanced', 'single'], default='balanced', help='Use balanced mode to extract utterances with two speakers and single mode to extract utterances with only one speaker.')
	parser.add_argument('--sample-rate', type = int, default = 8_000)
	parser.add_argument('--duration', dest = 'utterance_duration', type = float, default = 15.0)
	parser.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc', 'silero'], default = 'webrtc')
	parser.add_argument('--num-workers', type = int, default = 0)
	parser.add_argument('--stride', type = int, default = 1000)
	parser.add_argument('--min-utterance-score', dest = 'min_utterance_score', type = float, default = 0.25, help = 'Threshold value to accept utterance. Utterance score: speakers ratio in range [0;1] - silence ratio in range [0;1].')

	args = vars(parser.parse_args())
	make_separation_dataset(**args)
