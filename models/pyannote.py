import shapes
import torch
import numpy as np


class PyannoteDiarizationModel:
	def __init__(self, vad, vad_sensitivity: float, device: str, sample_rate: int):
		assert sample_rate == 16_000, 'Only 16 kHz sample rate supported'
		self.vad = vad
		self.vad_sensitivity = vad_sensitivity
		self.device = device
		self.sample_rate = sample_rate
		self.model = torch.hub.load('pyannote/pyannote-audio', 'dia', device = torch.device(device), batch_size = 1)

	def get_silence_mask(self, signal: shapes.BT):
		smax = np.iinfo(np.int16).max
		f2s = lambda signal, max=np.float32(smax): torch.mul(signal, max).to(torch.int16)
		s2f = lambda signal, max=np.float32(smax): torch.div(signal, max, dtype = torch.float32)

		if signal.dtype == torch.float32 and self.vad.required_type == 'int16':
			signal = f2s(signal)
		if signal.dtype == torch.int16 and self.vad.required_type == 'float32':
			signal = s2f(signal)

		if self.vad.required_wrapper == np.array:
			signal = signal.cpu().numpy()
			silence_mask = torch.as_tensor(self.vad.detect(signal)[0], dtype = torch.bool, device = self.device)
		else:
			silence_mask = self.vad.detect(signal)[0]
		return silence_mask

	def get_speaker_mask(self, signal: shapes.BT, sample_rate: int):
		assert sample_rate == self.sample_rate
		# extract features
		num_speakers = 0
		speakers_id = dict()
		with torch.no_grad():
			silence: shapes.T = self.get_silence_mask(signal)
			annotation = self.model(dict(waveform=signal.t().numpy(), sample_rate=sample_rate))
			speaker_mask = torch.zeros(2, signal.shape[-1], dtype = torch.bool, device = signal.device)
			for turn, _, speaker in annotation.itertracks(yield_label=True):
				if speaker not in speakers_id:
					speakers_id[speaker] = num_speakers
					num_speakers += 1
				speaker_id = speakers_id[speaker]
				speaker_mask[speaker_id, int(turn.start * sample_rate) : int(turn.end * sample_rate)] = True
			silence = silence | ~speaker_mask.any(dim = 0)
			speaker_mask = torch.cat([silence.unsqueeze(0), speaker_mask & (~silence.unsqueeze(0))])
		return speaker_mask
