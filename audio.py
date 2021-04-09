import torch
import numpy as np
import soundfile
import librosa
import subprocess


def read_audio(audio_path, sample_rate = None, mono = False, dtype = 'float32', __array_wrap__ = None):
	smax = np.iinfo(np.int16).max
	f2s_numpy = lambda signal, max = np.float32(smax): np.multiply(signal, max).astype('int16')
	s2f_numpy = lambda signal, max = np.float32(smax): np.divide(signal, max, dtype = 'float32')
	
	signal, sample_rate_ = soundfile.read(audio_path, dtype = 'int16', always_2d = True)
	signal = signal.T
	
	if signal.dtype == np.int16 and dtype == 'float32':
		signal = s2f_numpy(signal)
	
	if mono and len(signal) > 1:
		assert signal.dtype == np.float32
		signal = signal.mean(0, keepdims = True)

	if sample_rate is not None and sample_rate_ != sample_rate:
		assert signal.dtype == np.float32
		mono_ = len(signal) == 1
		signal = librosa.resample(signal[0 if mono_ else ...], sample_rate_, sample_rate)[None if mono_ else ...]
	else:
		sample_rate = sample_rate_

	return signal if __array_wrap__ is None else __array_wrap__(signal), sample_rate


def write_audio(audio_path, signal, sample_rate, mono = False, format = 'wav'):
	signal = signal.cpu().numpy() if torch.is_tensor(signal) else signal
	if  signal.dtype == np.float32:
		subtype = 'FLOAT'
	elif signal.dtype == np.int16:
		subtype = 'PCM_16'
	else:
		raise RuntimeError(f'Signal with type {signal.dtype} is not supported.')

	if mono and len(signal) != 1:
		signal = signal.mean(-1, keepdims = True, dtype = signal.dtype)

	soundfile.write(audio_path, signal, endian = 'LITTLE', samplerate = sample_rate, subtype = subtype, format = format.upper())
	return audio_path


def compute_duration(audio_path, backend = None):
	assert backend in [None, 'scipy', 'ffmpeg', 'sox']

	if backend is None:
		if audio_path.endswith('.wav'):
			backend = 'scipy'
		else:
			backend = 'ffmpeg'

	if backend == 'scipy':
		signal, sample_rate = read_audio(audio_path, sample_rate = None, dtype = None, mono = False)
		return signal.shape[-1] / sample_rate

	elif backend == 'ffmpeg':
		cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
			   'default=noprint_wrappers=1:nokey=1']
		return float(subprocess.check_output(cmd + [audio_path]))

	elif backend == 'sox':
		cmd = ['soxi', '-D']
		return float(subprocess.check_output(cmd + [audio_path]))
