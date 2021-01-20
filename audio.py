import torch
import numpy as np
import soundfile
import librosa
import subprocess

def read_audio(audio_path, sample_rate, mono = False, dtype = 'float32', __array_wrap__ = None):
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

	if sample_rate_ != sample_rate:
		assert signal.dtype == np.float32
		mono_ = len(signal) == 1
		signal = librosa.resample(signal[0 if mono_ else ...], sample_rate_, sample_rate)[None if mono_ else ...]

	return signal if __array_wrap__ is None else __array_wrap__(signal), sample_rate

def write_audio(audio_path, signal, sample_rate, mono = False, format = 'wav'):
	assert signal.dtype == torch.float32 or len(signal) == 1 or (not mono)

	signal = signal if (not mono or len(signal) == 1) else signal.mean(dim = 0, keepdim = True)
	
	assert not isinstance(audio_path, str) or audio_path.endswith('.' + format)
	assert signal.dtype == torch.float32 or signal.dtype == torch.int16
	subtype = 'FLOAT' if signal.dtype == torch.float32 else 'PCM_16'
	soundfile.write(audio_path, signal.numpy(), endian = 'LITTLE', samplerate = sample_rate, subtype = subtype, format = format.upper()) 
	return audio_path

def compute_duration(audio_path, backend = None):
	assert backend in [None, 'scipy', 'ffmpeg', 'sox']

	if backend is None:
		if audio_path.endswith('.wav'):
			backend = 'scipy'
		else:
			backend = 'ffmpeg'

	if backend == 'scipy':
		signal, sample_rate = read_audio(audio_path, sample_rate = None, dtype = None, mono = False, backend = 'scipy')
		return signal.shape[-1] / sample_rate

	elif backend == 'ffmpeg':
		cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
			   'default=noprint_wrappers=1:nokey=1']
		return float(subprocess.check_output(cmd + [audio_path]))

	elif backend == 'sox':
		cmd = ['soxi', '-D']
		return float(subprocess.check_output(cmd + [audio_path]))
