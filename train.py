import os
import json
import numpy as np
import soundfile
import torch
import models

def read_audio(audio_path, mono = False, dtype = 'float32'):
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

	signal = torch.as_tensor(signal)

	return signal, sample_rate_

if __name__ == '__main__':
	weights_path = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt'
	transcript_path = '../dia_dataset/diacalls_2020-09-01_2020-10-01.ref.json'
	output_path = './embeddings'
	device = 'cuda'

	os.makedirs(output_path, exist_ok = True)

	torch.set_grad_enabled(False)
	model = models.SincTDNN(tdnn = dict(kernel_size_pool = 21, stride_pool = 21))
	
	model.load_state_dict(torch.load(weights_path), strict = False)
	model.eval()
	model.to(device)

	audio_paths = sorted(set(t['audio_path'] for t in json.load(open(transcript_path))))
	for i, audio_path in enumerate(audio_paths):
		channels, sample_rate = read_audio(audio_path, dtype = 'float32', mono = True)  #(torch.rand(16, 4*1024) - 0.5) * 2
		features = model(channels.to(device))[0]
		torch.save(features.cpu(), os.path.join(output_path, os.path.basename(audio_path) + '.pt'))
		print(i, '/', len(audio_paths), features.shape)

