import argparse
import os
import json
import numpy as np
import soundfile
import torch
import models

#pipeline = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')

def read_audio(audio_path, sample_rate, mono = False, dtype = 'float32'):
	smax = np.iinfo(np.int16).max
	f2s_numpy = lambda signal, max = np.float32(smax): np.multiply(signal, max).astype('int16')
	s2f_numpy = lambda signal, max = np.float32(smax): np.divide(signal, max, dtype = 'float32')
	
	signal, sample_rate_ = soundfile.read(audio_path, dtype = 'int16')
	signal = signal[:, None] if len(signal.shape) == 1 else signal
	signal = signal.T
	
	if signal.dtype == np.int16 and dtype == 'float32':
		signal = s2f_numpy(signal)
	
	if mono and len(signal) > 1:
		assert signal.dtype == np.float32
		signal = signal.mean(0, keepdims = True)

	#assert sample_rate_ == sample_rate

	return torch.as_tensor(signal), sample_rate

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights-path', default = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt'
	parser.add_argument('--transcript-path', default '../dia_dataset/diacalls_short_2020-09-01_2020-10-01.ref')
	parser.add_argument('--output-path', '-o', default = './embeddings')
	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--sample-rate', type = int, default = 16_000)
	args = parser.parse_args()

	os.makedirs(output_path, exist_ok = True)
	
	#kernel_size_pool, stride_pool = int(4.0 * sample_rate), int(1.0 * sample_rate)

	torch.set_grad_enabled(False)
	model = models.SincTDNN(padding_same = True, tdnn = dict(kernel_size_pool = 21, stride_pool = 21)) # dict(kernel_size_pool = kernel_size_pool, stride_pool = stride_pool))
	diag = model.load_state_dict(torch.load(args.weights_path), strict = False)
	print(diag.missing_keys, diag.unexpected_keys)
	model.eval()
	model.to(args.device)
	
	#res = pipeline(dict(waveform = channels.t().numpy(), sample_rate = sample_rate)).data

	transcript = [t for transcript_name in os.listdir(args.transcript_path) if transcript_name.endswith('.json') for transcript in json.load(open(os.path.join(args.transcript_path, transcript_name))) for t in transcript]
	audio_paths = sorted({t['audio_path'] for t in transcript})
	for i, audio_path in enumerate(audio_paths):
		channels, _ = read_audio(audio_path, sample_rate = args.sample_rate, dtype = 'float32', mono = True)
		features = model(channels.to(args.device))[0]

		torch.save(features.cpu(), os.path.join(args.output_path, os.path.basename(audio_path) + '.pt'))
		print(i, '/', len(audio_paths), features.shape)
