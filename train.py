import json
import numpy as np
import soundfile
import torch
import models

def read_audio(audio_path, mono = False, dtype = 'float32'):
	smax = torch.iinfo(torch.int16).max
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
	transcript_path = 'subset.json'
	weights_path = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt'

	torch.set_grad_enabled(False)
	model = models.SincTDNN(tdnn = dict(kernel_size_pool = 21, stride_pool = 3))
	
	model.load_state_dict(torch.load(weights_path))
	model.eval()

	transcript = json.load(open(transcript_path))
	#embeddings = open('embeddings.tsv', 'w')
	#speaker_names = open('speaker_names.tsv', 'w')
	for i, t in enumerate(transcript):
		channels, sample_rate = read_audio(t['audio_path'], dtype = 'float32', mono = False)  #(torch.rand(16, 4*1024) - 0.5) * 2
		features = model(channels)
		break
		#embeddings.write('\t'.join(map('{:.02f}'.format, features[0].cpu().clone().tolist())) + '\n')
		#speaker_names.write(t['speaker_name'] + '\n')
		print(i, '/', len(transcript))

	#print(waveforms.shape, features.shape)
	#features.backward(torch.ones_like(features))
