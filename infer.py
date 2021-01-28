import os
import json
import argparse
import torch
import models
import shapes
import audio
import vad
import transcripts


def main(args):
	# load vad
	if args.vad_type == 'simple':
		_vad = vad.PrimitiveVAD(device = args.device)
	elif args.vad_type == 'webrtc':
		_vad = vad.WebrtcVAD(sample_rate = args.sample_rate)
	else:
		raise RuntimeError(f'VAD for type {args.vad_type} not found.')

	# loading model
	model = models.SpectralClusteringModel(vad = _vad, vad_sensitivity = 0.90, weights_path = args.weights_path, device = args.device, sample_rate = args.sample_rate)

	# loading audio
	signal: shapes.BT; sample_rate: int
	signal, sample_rate = audio.read_audio(args.audio_path, sample_rate = args.sample_rate, dtype = 'float32', mono = True, __array_wrap__ = torch.as_tensor)
	print('read_audio done', signal.shape)

	speaker_mask = model.get_speaker_mask(signal, sample_rate)

	# save transcript
	transcript = dict(
		audio_path = args.audio_path,
		audio_name = os.path.basename(args.audio_path),
		markup = transcripts.mask_to_intervals(speaker_mask, sample_rate)
	)
	with open(args.transcript_path, 'w') as output_file:
		output_file.write(json.dumps(transcript, ensure_ascii=False))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio-path', '-i', required=True)
	parser.add_argument('--transcript-path', '-o', required=True)
	parser.add_argument('--weights-path', default = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt')
	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--vad', dest = 'vad_type', choices = ['simple', 'webrtc'], default = 'webrtc')
	parser.add_argument('--sample-rate', type = int, default = 16_000)
	parser.add_argument('--num-speakers', type = int, default = 2)
	args = parser.parse_args()

	main(args)
