import shapes
import torch
import torch.nn.functional as F


def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())


class PrimitiveVAD:
	def __init__(self,
	             kernel_size_smooth_silence : int = 128,
	             kernel_size_smooth_signal: int = 4096,
	             kernel_size_smooth_speaker: int = 4096,
	             silence_absolute_threshold: float = 0.2,
	             silence_relative_threshold: float = 0.5,
	             eps: float = 1e-9,
	             normalization_percentile: float = 0.9):
		self.kernel_size_smooth_silence = kernel_size_smooth_silence
		self.kernel_size_smooth_signal = kernel_size_smooth_signal
		self.kernel_size_smooth_speaker = kernel_size_smooth_speaker
		self.silence_absolute_threshold = silence_absolute_threshold
		self.silence_relative_threshold = silence_relative_threshold
		self.eps = eps
		self.normalization_percentile = normalization_percentile

	def detect(self, signal: shapes.BT, keep_intersections: bool = False) -> shapes.BT:
		assert len(signal) == 2

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

		if keep_intersections:
			return ~silence
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
			return (~silence) * (speaker_id_bipole.unsqueeze(0) == bipole.unsqueeze(1))


class WebrtcVAD:
	def __init__(self, aggressiveness: int = 1, sample_rate: int = 8_000, window_size: float = 0.02):
		'''
		Aggressiveness mode, which is an integer between 0 and 3.
		0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
		'''
		import webrtcvad
		assert sample_rate in [8_000, 16_000, 32_000, 48_000]
		assert window_size in [0.01, 0.02, 0.03]
		self.sample_rate = sample_rate
		self.window_size = window_size
		self.frame_len = int(window_size * sample_rate)
		self.vad = webrtcvad.Vad(aggressiveness)
	
	def detect(self, signal: shapes.BT, keep_intersections: bool = False) -> shapes.BT:
		assert signal.dtype == torch.int16
		speech = []
		for channel in range(len(signal)):
			speech.append([])
			for frame in signal[channel].split(self.frame_len):
				chunk_size = len(frame)
				if chunk_size < self.frame_len:
					frame = torch.stack([frame, torch.zeros(self.frame_len - chunk_size, dtype = torch.int16, device = frame.device)])
				if self.vad.is_speech(bytearray(frame.numpy()), self.sample_rate):
					speech[-1].extend([True] * chunk_size)
				else:
					speech[-1].extend([False] * chunk_size)

		speech = torch.tensor(speech, device = signal.device)
		if keep_intersections:
			return speech
		else:
			# prefer started earlier speaker, could be optimized to longest speech algorithm
			simultaneous_speakers = speech.sum(dim = -1)
			for i in range(speech.shape[-1]):
				if simultaneous_speakers[i] > 1:
					speech[i, :] = False
					speech[i, speech[i-1, :].max()] = True
		return speech
