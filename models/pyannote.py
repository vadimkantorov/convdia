# pip install pyannote.metrics
# pip install pyannote.pipeline
# pip install pescador
# pip install optuna
# pip install filelock
# pip install git+https://github.com/pyannote/pyannote-audio
# pip install sortedcontainers simplejson typing_extensions pyannote.database pyannote.metrics
# pip install pyannote.core --no-dependencies

import shapes
import torch
import numpy as np


class PyannoteDiarizationModel:
	def __init__(self, vad, vad_sensitivity: float, sample_rate: int):
		assert sample_rate == 16_000, 'Only 16 kHz sample rate supported'
		self.vad = vad
		self.vad_sensitivity = vad_sensitivity
		self.sample_rate = sample_rate
		self.model = torch.hub.load('pyannote/pyannote-audio', 'dia', batch_size = 1)

	@property
	def input_dtype(self):
		return self.vad.input_dtype

	def get_silence_mask(self, signal: shapes.BT):
		if self.vad.input_type == np.array:
			device = signal.device
			signal = signal.cpu().numpy()
			silence_mask = torch.as_tensor(self.vad.detect(signal)[0], dtype = torch.bool, device = device)
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
				# only 2 speaker markup supports now
				if speaker_id < 2:
					speaker_mask[speaker_id, int(turn.start * sample_rate) : int(turn.end * sample_rate)] = True
			silence = silence | ~speaker_mask.any(dim = 0)
			speaker_mask = torch.cat([silence.unsqueeze(0), speaker_mask & (~silence.unsqueeze(0))])
		return speaker_mask
