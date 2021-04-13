import shapes
import numpy as np
import torch
import torch.nn.functional as F
import functools


def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())


class SimpleVAD:
	def __init__(self,
	             kernel_size_smooth_silence: int = 4096,
	             kernel_size_smooth_signal: int = 128,
	             kernel_size_smooth_speaker: int = 4096,
	             silence_absolute_threshold: float = 0.05,
	             silence_relative_threshold: float = 0.2,
	             eps: float = 1e-9,
	             normalization_percentile: float = 0.9):
		self.kernel_size_smooth_silence = kernel_size_smooth_silence
		self.kernel_size_smooth_signal = kernel_size_smooth_signal
		self.kernel_size_smooth_speaker = kernel_size_smooth_speaker
		self.silence_absolute_threshold = silence_absolute_threshold
		self.silence_relative_threshold = silence_relative_threshold
		self.eps = eps
		self.normalization_percentile = normalization_percentile
		self.input_type = torch.tensor
		self.input_dtype = 'float32'

	def detect(self, signal: shapes.BT, allow_overlap: bool = False) -> shapes.BT:
		assert len(signal) <= 2

		padding = self.kernel_size_smooth_signal // 2
		stride = 1
		smoothed_for_diff = F.max_pool1d(signal.abs().unsqueeze(1), self.kernel_size_smooth_signal, stride=stride, padding=padding).squeeze(1)

		padding = self.kernel_size_smooth_silence // 2
		stride = 1

		# dilation
		smoothed_for_silence = F.max_pool1d(signal.abs().unsqueeze(1), self.kernel_size_smooth_silence, stride=stride, padding=padding).squeeze(1)

		# erosion
		smoothed_for_silence = -F.max_pool1d(-smoothed_for_silence.unsqueeze(1), self.kernel_size_smooth_silence, stride=stride, padding=padding).squeeze(1)

		# primitive VAD
		signal_max = smoothed_for_diff.kthvalue(int(self.normalization_percentile * smoothed_for_diff.shape[-1]), dim=-1, keepdim=True).values
		silence_absolute = smoothed_for_silence < self.silence_absolute_threshold
		silence_relative = smoothed_for_silence / (self.eps + signal_max) < self.silence_relative_threshold
		silence = silence_absolute | silence_relative

		if allow_overlap or len(signal) == 1:
			speech = ~silence
		else:
			diff_flat = smoothed_for_diff[0] - smoothed_for_diff[1]
			speaker_id_bipole = diff_flat.sign()

			padding = self.kernel_size_smooth_speaker // 2
			stride = 1
			speaker_id_bipole = F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size=self.kernel_size_smooth_speaker, stride=stride, padding=padding).view(-1).sign()

			# removing 1 sample silence at 1111-1-1-1-1 boundaries, replace by F.conv1d (patterns -101, 10-1)
			speaker_id_bipole = torch.where((speaker_id_bipole == 0) & (F.avg_pool1d(speaker_id_bipole.abs().view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1) == 2 / 3) & (
						F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1) == 0), torch.ones_like(speaker_id_bipole), speaker_id_bipole)

			resize_to_min_size_(silence, speaker_id_bipole, dim=-1)

			bipole = torch.tensor([1, -1], dtype=speaker_id_bipole.dtype, device=speaker_id_bipole.device)
			speech = (~silence) * (speaker_id_bipole.unsqueeze(0) == bipole.unsqueeze(1))
		speech = torch.cat([~speech.any(dim = 0).unsqueeze(0), speech])
		return speech


class WebrtcVAD:
	def __init__(self, aggressiveness: int = 3, sample_rate: int = 8_000, window_size: float = 0.01):
		assert sample_rate in [8_000, 16_000, 32_000, 48_000]
		assert window_size in [0.01, 0.02, 0.03]
		assert aggressiveness in [0, 1, 2, 3] # 3 is the most aggressive
		self.sample_rate = sample_rate
		self.window_size = window_size
		self.frame_len = int(window_size * sample_rate)
		self.aggressiveness = aggressiveness
		self.input_type = np.array
		self.input_dtype = 'int16'

	def detect(self, signal: shapes.BT, allow_overlap: bool = False) -> shapes.BT:
		assert signal.dtype == np.int16
		import webrtcvad
		speech_length = np.zeros(signal.shape, dtype = np.int)
		for channel in range(len(signal)):
			vad = webrtcvad.Vad(self.aggressiveness)
			frames = np.pad(signal[channel], (0, self.frame_len - signal.shape[-1] % self.frame_len), 'constant', constant_values = (0, 0))
			frames = bytearray(frames)
			start = None
			amount = 0

			for i in range(0, int(len(frames) / 2), self.frame_len):
				is_speech = vad.is_speech(frames[i * 2: (i + self.frame_len) * 2], self.sample_rate)
				if is_speech and start is None:
					start = i
					amount = 1
				elif is_speech:
					amount += 1
				elif not is_speech and start is not None:
					speech_length[channel, start: start + amount * self.frame_len] = amount * self.frame_len
					start = None
					amount = 0

			if start is not None:
				speech_length[channel, start: start + amount * self.frame_len] = amount * self.frame_len

		if allow_overlap:
			speech = speech_length > 0
		else:
			speech_max_length = speech_length.max(axis=0)
			speech = (speech_length == speech_max_length[np.newaxis, :]) & (speech_max_length != 0)
		speech = np.vstack([~speech.any(axis = 0), speech])
		return speech


class SileroVAD:
	def __init__(self, sample_rate: int = 8_000, use_micro = True, device = 'cpu'):
		self.device = device
		self.sample_rate = sample_rate
		if sample_rate == 8_000:
			assert use_micro, 'Only "micro" model exists for 8 kHz sample rate.'
			self.model_name = 'silero_vad_micro_8k'
		elif sample_rate == 16_000 and use_micro:
			self.model_name = 'silero_vad_micro'
		elif sample_rate == 16_000 and not use_micro:
			self.model_name = 'silero_vad'
		else:
			raise RuntimeError(f'No model exist for sample rate {sample_rate} Hz.')

		self.input_type = torch.tensor
		self.input_dtype = 'float32'

	@functools.lru_cache()
	def _get_model(self):
		model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model=self.model_name, batch_size=2000)
		get_speech_ts, _, _, _, _, _ = utils
		return model.to(self.device), get_speech_ts

	def detect(self, signal: shapes.BT, allow_overlap: bool = False) -> shapes.BT:
		signal = signal.to(self.device)
		model, get_speech_ts = self._get_model()
		speech_length = torch.zeros_like(signal, dtype = torch.int64)
		for i, channel_signal in enumerate(signal):
			intervals = get_speech_ts(channel_signal, model)
			for interval in intervals:
				speech_length[i, interval['start']:interval['end']] = torch.arange(0, interval['end'] - interval['start'], dtype = torch.int64)

		if allow_overlap:
			speech = speech_length > 0
		else:
			speech_max_length = speech_length.amax(dim=0)
			speech = (speech_length == speech_max_length.unsqueeze(0)) & (speech_max_length != 0)
		speech = torch.cat([~speech.any(dim = 0).unsqueeze(0), speech])
		return speech
