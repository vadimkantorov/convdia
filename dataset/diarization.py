import os
import json
import tqdm
import argparse
import vad
import audio
import functools
import transcripts
import multiprocessing as mp


def make_diarization_dataset(input_path: str, output_path: str, sample_rate: int, keep_intersections: bool, vad_type: str, device: str, processes: int):
	if os.path.isdir(input_path):
		audio_files = [os.path.join(input_path, audio_name) for audio_name in os.listdir(input_path)]
	else:
		audio_files = [input_path]

	if vad_type == 'simple':
		_vad = vad.PrimitiveVAD(device = device)
	elif vad_type == 'webrtc' and processes == 0:
		_vad = vad.WebrtcVAD(sample_rate = sample_rate)
	elif vad_type == 'webrtc' and processes > 0:
		_vad = vad.PicklableWebrtcVAD(sample_rate = sample_rate)
	else:
		raise RuntimeError(f'VAD for type {vad_type} not found.')

	if processes > 0:
		parametrized_generate = functools.partial(generate_markup, sample_rate=sample_rate, vad=_vad, keep_intersections=keep_intersections)
		with mp.Pool(processes=processes) as pool:
			dataset = list(tqdm.tqdm(pool.imap(parametrized_generate, audio_files), total=len(audio_files)))
	else:
		dataset = [generate_markup(audio_path, sample_rate, _vad, keep_intersections) for audio_path in tqdm.tqdm(audio_files)]

	with open(output_path, 'w') as output:
		output.write('\n'.join([json.dumps(example, ensure_ascii = False) for example in dataset]))


def generate_markup(audio_path: str, sample_rate: int, vad, keep_intersections: bool):
	signal, _ = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = vad.required_type, __array_wrap__ = vad.required_wrapper)
	speaker_masks = vad.detect(signal, keep_intersections)
	markup = transcripts.mask_to_intervals(speaker_masks, sample_rate)
	return dict(audio_path = audio_path, audio_name = os.path.basename(audio_path), markup = markup, sample_rate = sample_rate, duration = signal.shape[-1] / sample_rate)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('diarization')
	cmd.add_argument('--input-path', '-i', required=True)
	cmd.add_argument('--output-path', '-o', required=True)
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--keep-intersections', action = 'store_true', default = False)
	cmd.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc'], default = 'webrtc')
	cmd.add_argument('--device', type = str, default = 'cpu')
	cmd.add_argument('--processes', type = int, default = 0)
	cmd.set_defaults(func = make_diarization_dataset)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
