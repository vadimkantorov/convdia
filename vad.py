import shapes

def select_speaker(signal : shapes.BT, kernel_size_smooth_silence : int, kernel_size_smooth_signal : int, kernel_size_smooth_speaker : int, silence_absolute_threshold : float = 0.2, silence_relative_threshold : float = 0.5, eps : float = 1e-9, normalization_percentile = 0.9) -> shapes.T:
	#TODO: remove bipole processing, smooth every speaker, conditioned on the other speaker

	assert len(signal) == 2

	padding = kernel_size_smooth_signal // 2
	stride = 1
	smoothed_for_diff = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_signal, stride = stride, padding = padding).squeeze(1)

	padding = kernel_size_smooth_silence // 2
	stride = 1
	
	# dilation
	smoothed_for_silence = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	# erosion
	smoothed_for_silence = -F.max_pool1d(-smoothed_for_silence.unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	# primitive VAD
	signal_max = smoothed_for_diff.kthvalue(int(normalization_percentile * smoothed_for_diff.shape[-1]), dim = -1, keepdim = True).values
	silence_absolute = smoothed_for_silence < silence_absolute_threshold
	silence_relative = smoothed_for_silence / (eps + signal_max) < silence_relative_threshold
	silence = silence_absolute | silence_relative
	
	diff_flat = smoothed_for_diff[0] - smoothed_for_diff[1]
	speaker_id_bipole = diff_flat.sign()
	
	padding = kernel_size_smooth_speaker // 2
	stride = 1
	speaker_id_bipole = F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size = kernel_size_smooth_speaker, stride = stride, padding = padding).view(-1).sign()

	# removing 1 sample silence at 1111-1-1-1-1 boundaries, replace by F.conv1d (patterns -101, 10-1)
	speaker_id_bipole = torch.where((speaker_id_bipole == 0) & (F.avg_pool1d(speaker_id_bipole.abs().view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 2/3) & (F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 0), torch.ones_like(speaker_id_bipole), speaker_id_bipole)
	
	resize_to_min_size_(silence, speaker_id_bipole, dim = -1)
	
	silence_flat = silence.all(dim = 0)
	speaker_id_categorical = models.convert_speaker_id(speaker_id_bipole, from_bipole = True) * (~silence_flat)

	bipole = torch.tensor([1, -1], dtype = speaker_id_bipole.dtype, device = speaker_id_bipole.device)
	speaker_id_mask = (~silence) * (speaker_id_bipole.unsqueeze(0) == bipole.unsqueeze(1))
	return speaker_id_categorical, torch.cat([silence_flat.unsqueeze(0), speaker_id_mask])

	#speaker_id = torch.where(silence.any(dim = 0), torch.tensor(0, device = signal.device, dtype = speaker_id.dtype), speaker_id)
	#return speaker_id, silence_flat, torch.stack((speaker_id * 0.5, diff, smoothed_for_silence[0], smoothed_for_silence[1] , silence_flat.float() * 0.5))

def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())

class WebrtcSpeechActivityDetectionModel(nn.Module):
	def __init__(self, aggressiveness):
		import webrtcvad
		self.vad = webrtcvad.Vad(aggressiveness)
	
	def forward(self, signal, sample_rate, window_size = 0.02, extra = {}):
		assert sample_rate in [8_000, 16_000, 32_000, 48_000] and signal.dtype == torch.int16 and window_size in [0.01, 0.02, 0.03]
		frame_len = int(window_size * sample_rate)
		speech = torch.as_tensor([[len(chunk) == frame_len and self.vad.is_speech(bytearray(chunk.numpy()), sample_rate) for chunk in channel.split(frame_len)]	for channel in signal])
		transcript = [dict(begin = float(begin) * window_size, end = (float(begin) + float(duration)) * window_size, speaker = 1 + channel, speaker_name = transcripts.default_speaker_names[1 + channel], **extra) for channel in range(len(signal)) for begin, duration, mask in zip(*rle1d(speech[speaker])) if mask == 1]
		return transcript
