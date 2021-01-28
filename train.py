import argparse
import os
import json
import torch
import models
import audio

#pipeline = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights-path', default = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt')
	parser.add_argument('--transcript-path', default = '../dia_dataset/diacalls_short_2020-09-01_2020-10-01.ref')
	parser.add_argument('--output-path', '-o', default = './embeddings')
	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--sample-rate', type = int, default = 16_000)
	args = parser.parse_args()

	os.makedirs(args.output_path, exist_ok = True)
	
	
	torch.set_grad_enabled(False)
	model = models.common.SincTDNN(sample_rate = args.sample_rate, padding_same = True, tdnn = dict(kernel_size_pool = 501, stride_pool = 101))#501, 101)) 101, 11
	print('stride:', model.stride / args.sample_rate, 'kernel_size:', model.kernel_size / args.sample_rate) # kernel_size, stride = int(4.0 * args.sample_rate), int(1.0 * args.sample_rate)
	diag = model.load_state_dict(torch.load(args.weights_path), strict = False)
	assert diag.missing_keys == ['sincnet_.conv1d_.0.window', 'sincnet_.conv1d_.0.sinct'] and diag.unexpected_keys == ['tdnn_.segment7.weight', 'tdnn_.segment7.bias']	
	
	model.eval()
	model.to(args.device)
	
	#res = pipeline(dict(waveform = channels.t().numpy(), sample_rate = sample_rate)).data

	transcript = [t for transcript_name in os.listdir(args.transcript_path) if transcript_name.endswith('.json') for t in json.load(open(os.path.join(args.transcript_path, transcript_name)))]
	
	audio_paths = sorted({t['audio_path'] for t in transcript})
	
	for i, audio_path in enumerate(audio_paths):
		audio_path = '_ES2004a.Mix-Headset.wav'
		channels, _ = audio.read_audio(audio_path, sample_rate = args.sample_rate, dtype = 'float32', mono = True, __array_wrap__ = torch.as_tensor)
		features = model(channels.to(args.device), args.sample_rate)[0]

		torch.save(features.cpu(), os.path.join(args.output_path, os.path.basename(audio_path) + '.pt'))
		print(i, '/', len(audio_paths), features.shape, audio_path)
		break
