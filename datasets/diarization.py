import os
import json
import tqdm
import argparse
import vad
import audio
import torch
import functools
import transcripts
import multiprocessing as mp


def make_diarization_dataset(input_path: str, output_path: str, sample_rate: int, allow_overlap: bool, vad_type: str, num_workers: int):
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

	if num_workers > 0:
		parametrized_generate = functools.partial(generate_intervals, sample_rate=sample_rate, vad=_vad, allow_overlap=allow_overlap)
		with mp.Pool(processes=num_workers) as pool:
			dataset = list(tqdm.tqdm(pool.imap(parametrized_generate, audio_files), total=len(audio_files)))
	else:
		dataset = [generate_intervals(audio_path, sample_rate, _vad, allow_overlap) for audio_path in tqdm.tqdm(audio_files)]

	with open(output_path, 'w') as output:
		output.write('\n'.join([json.dumps(example, ensure_ascii = False) for example in dataset]))


def generate_intervals(audio_path: str, sample_rate: int, vad, allow_overlap: bool):
	signal, _ = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = vad.input_dtype, __array_wrap__ = vad.input_type)
	speaker_masks = vad.detect(signal, allow_overlap)
	if vad.input_type == torch.tensor:
		speaker_masks = speaker_masks.cpu().numpy()
	intervals = transcripts.mask_to_intervals(speaker_masks, sample_rate)
	return dict(audio_path = audio_path, audio_name = os.path.basename(audio_path), intervals = intervals, sample_rate = sample_rate, duration = signal.shape[-1] / sample_rate)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-path', '-i', required=True)
	parser.add_argument('--output-path', '-o', required=True)
	parser.add_argument('--sample-rate', type = int, default = 8_000)
	parser.add_argument('--allow-overlap', action = 'store_true', default = False)
	parser.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc', 'silero'], default = 'webrtc')
	parser.add_argument('--num-workers', type = int, default = 0)

	args = vars(parser.parse_args())
	make_diarization_dataset(**args)
