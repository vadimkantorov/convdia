import argparse
import torch
import models
import shapes
import audio
import transcripts

def main(args):
	# loading model
	torch.set_grad_enabled(False)
	model = models.SincTDNN(sample_rate = args.sample_rate, padding_same = True, tdnn = dict(kernel_size_pool = 501, stride_pool = 101))
	print('stride:', model.stride / args.sample_rate, 'kernel_size:', model.kernel_size / args.sample_rate)
	diag = model.load_state_dict(torch.load(args.weights_path), strict = False)
	assert diag.missing_keys == ['sincnet_.conv1d_.0.window', 'sincnet_.conv1d_.0.sinct'] and diag.unexpected_keys == ['tdnn_.segment7.weight', 'tdnn_.segment7.bias']	
	model.eval()
	model.to(args.device)

	# loading audio
	# type: (shapes.BT, int) 
	signal, sample_rate = audio.read_audio(args.audio_path, sample_rate = args.sample_rate, dtype = 'float32', mono = True, __array_wrap__ = torch.as_tensor)
	print('read_audio done', signal.shape)
	features : shapes.BCt = model(signal.to(args.device), args.sample_rate)
	print('model done', features.shape)

	# compute affinity
	emb = features[0].t()
	K = models.wang_affinity(emb)
	W = (K > 0.5).float() # ???
	L = models.normalized_symmetric_laplacian(W)
	eigvals, eigvecs = L.symeig(eigenvectors = True)
	num_eigvecs = 1
	eigvecs = eigvecs[:, 1 : (1 + num_eigvecs)]
	eigvecs_max = eigvecs.max()
	fiedler_vector = eigvecs[:, 0] / eigvecs_max
	print('fiedler_vector', fiedler_vector.shape)

	silence = torch.zeros_like(fiedler_vector, dtype = torch.bool)
	speaker1 = fiedler_vector < 0
	speaker2 = fiedler_vector > 0

	speaker_mask = torch.stack([silence, speaker1, speaker2])

	# save transcript
	transcript = transcripts.from_speaker_mask(speaker_mask, sample_rate)
	print(transcripts.save(args.transcript_path, transcript))
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio-path', '-i')
	parser.add_argument('--transcript-path', '-o')
	parser.add_argument('--weights-path', default = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt')
	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--sample-rate', type = int, default = 16_000)
	parser.add_argument('--num-speakers', type = int, default = 2)
	args = parser.parse_args()

	main(args)
