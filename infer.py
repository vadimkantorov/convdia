import os
import tqdm
import json
import argparse
import torch
import functools
import models
import shapes
import audio
import vad
import transcripts
import multiprocessing as mp


def get_transcript(model, audio_path: str, sample_rate: int):
	signal: shapes.BT; sample_rate: int
	signal, sample_rate = audio.read_audio(audio_path, sample_rate=sample_rate, dtype='float32', mono=True, __array_wrap__=torch.as_tensor)

	speaker_mask = model.get_speaker_mask(signal, sample_rate)

	return dict(
		audio_path = audio_path,
		audio_name = os.path.basename(audio_path),
		markup = transcripts.mask_to_intervals(speaker_mask, sample_rate),
		sample_rate = sample_rate,
		duration = signal.shape[-1] / sample_rate
	)


def main(args):
	# load vad
	if args.vad_type == 'simple':
		_vad = vad.PrimitiveVAD(device=args.device)
	elif args.vad_type == 'webrtc':
		_vad = vad.WebrtcVAD(sample_rate=args.sample_rate)
	else:
		raise RuntimeError(f'VAD for type {args.vad_type} not found.')

	# loading model
	if args.model == 'spectral':
		model = models.SpectralClusteringDiarizationModel(vad = _vad, vad_sensitivity = 0.90, weights_path = args.weights_path, device = args.device, sample_rate = args.sample_rate)
	elif args.model == 'pyannote':
		model = models.PyannoteDiarizationModel(vad = _vad, vad_sensitivity = 0.90, device = args.device, sample_rate = args.sample_rate)
	else:
		raise RuntimeError(f'Diarization model for name "{args.model}" not found.')

	if args.audio_path[-5:] == '.json':
		with open(args.audio_path) as dataset_file:
			paths = [json.loads(line)['audio_path'] for line in dataset_file]
	else:
		paths = [args.audio_path]

	# make transcripts
	if args.processes > 0:
		parametrized_transcript = functools.partial(get_transcript, model, sample_rate = args.sample_rate)
		with mp.Pool(processes=args.processes) as pool:
			_transcripts = list(tqdm.tqdm(pool.imap(parametrized_transcript, paths), total=len(paths)))
	else:
		_transcripts = [get_transcript(model, path, args.sample_rate) for path in tqdm.tqdm(paths)]

	with open(args.transcript_path, 'w') as output_file:
		output_file.write('\n'.join(json.dumps(t, ensure_ascii=False) for t in _transcripts))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio-path', '-i', required=True)
	parser.add_argument('--transcript-path', '-o', required=True)
	parser.add_argument('--weights-path', default = '_emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt')
	parser.add_argument('--device', default = 'cpu')
	parser.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc'], default = 'webrtc')
	parser.add_argument('--model', choices = ['pyannote', 'spectral'], default = 'pyannote')
	parser.add_argument('--sample-rate', type = int, default = 16_000)
	parser.add_argument('--num-speakers', type = int, default = 2)
	parser.add_argument('--processes', type=int, default=0)
	args = parser.parse_args()

	main(args)
