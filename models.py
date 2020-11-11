# Code is adapted from pyannote.audio. All credit goes to pyannote.audio
# https://github.com/pyannote/pyannote-audio/tree/master/pyannote/audio/models

import math
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

class SincTDNN(nn.Module):
	def __init__(self, sincnet: dict = {}, tdnn: dict = {}):
		super().__init__()
		self.sincnet_ = SincNet(**sincnet)
		self.tdnn_ = XVectorNet(self.sincnet_.out_channels[-1], **tdnn)

	def forward(self, waveforms : torch.Tensor, eps : float = 1e-6):
		x = self.sincnet_(waveforms)
		x = self.tdnn_(x)
		return x / (x.norm(p = 2, dim = -1, keepdim = True) + eps) 

class SincConv1d(torch.nn.Conv1d):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = False, padding_mode = 'zeros', sample_rate = 16_000, min_low_hz = 50, min_band_hz = 50, low_hz = 30):
		assert in_channels == 1 and kernel_size % 2 == 1 and bias is False and groups == 1
		super().__init__(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias, padding_mode = padding_mode)
		self.register_parameter('weight', None)

		to_mel = lambda hz: 2595 * torch.log10(1 + hz / 700)
		to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
		ediff1d = lambda t, prepend = False: t[1:] - t[:-1]

		self.sample_rate = sample_rate
		self.min_low_hz = min_low_hz
		self.min_band_hz = min_band_hz

		high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)

		hz = to_hz(torch.linspace(to_mel(torch.tensor(float(low_hz))), to_mel(torch.tensor(float(high_hz))), steps = out_channels + 1))

		self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(-1))
		self.band_hz_ = nn.Parameter(ediff1d(hz).unsqueeze(-1))

		self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * math.pi * torch.linspace(0, kernel_size / 2 - 1, steps=int((kernel_size / 2))) / kernel_size))
		#self.window = torch.hamming_window(kernel_size)[:kernel_size // 2]
		
		self.register_buffer('sinct', 2 * math.pi * torch.arange(-(kernel_size // 2), 0, dtype = torch.float32) / sample_rate)

	@property
	def weight(self):
		if self._buffers.get('weight') is not None:
			return self._buffers['weight']

		lo = self.min_low_hz + self.low_hz_.abs()
		hi = (lo + self.min_band_hz + self.band_hz_.abs()).clamp(min = self.min_low_hz, max = self.sample_rate / 2)
		sincarg_hi, sincarg_lo = hi * self.sinct, lo * self.sinct

		band_pass_left = (sincarg_hi.sin() - sincarg_lo.sin()) / (self.sinct / 2) * self.window
		band_pass_center = (hi - lo) * 2
		band_pass_right = band_pass_left.flip(dims = [1])
		band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim = 1) / band_pass_center
		
		return band_pass.unsqueeze(1)
		

class SincNet(nn.Module):
	def __init__(
		self,
		waveform_normalize = True,
		sample_rate = 16_000,
		min_low_hz = 50,
		min_band_hz = 50,
		out_channels = [80, 60, 60],
		kernel_size = [251, 5, 5],
		stride = [5, 1, 1],
		max_pool = [3, 3, 3],
		instance_normalize = True,
		activation = 'leaky_relu',
		dropout = 0.0,
		padding_same = True
	):
		super().__init__()
		self.out_channels = out_channels
		self.waveform_normalize_ = nn.InstanceNorm1d(1, affine = True) if waveform_normalize else nn.Identity()
		self.activation_ = nn.LeakyReLU(negative_slope = 0.2) if activation == 'leaky_relu' else nn.Identity()
		self.dropout_ = nn.Dropout(p = dropout) if dropout else nn.Identity()
		self.conv1d_ = nn.ModuleList()
		self.max_pool1d_ = nn.ModuleList()
		self.instance_norm1d_ = nn.ModuleList()

		in_channels = 1
		for i, (out_channels, kernel_size, stride, max_pool) in enumerate(zip(out_channels, kernel_size, stride, max_pool)):
			conv_kwargs = dict(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2 if padding_same else 0)
			self.conv1d_.append(nn.Conv1d(**conv_kwargs, bias = True) if i > 0 else SincConv1d(**conv_kwargs, bias = False, sample_rate = sample_rate, min_low_hz = min_low_hz, min_band_hz = min_band_hz))
			self.max_pool1d_.append(nn.MaxPool1d(max_pool, stride = max_pool, padding = max_pool // 2 if padding_same else 0))
			self.instance_norm1d_.append(nn.InstanceNorm1d(out_channels, affine = True) if instance_normalize else nn.Identity())
			in_channels = out_channels

	def forward(self, waveforms):
		x = waveforms.unsqueeze(1)
		x = self.waveform_normalize_(x)
		for i, (conv1d, max_pool1d) in enumerate(zip(self.conv1d_, self.max_pool1d_)):
			x = conv1d(x)
			x = x.abs() if i == 0 else x
			x = max_pool1d(x)
			x = self.instance_norm1d_[i](x)
			x = self.activation_(x)
			x = self.dropout_(x)
		return x

class WeightNormConv1d(nn.Conv1d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.weight_v = nn.Parameter(self.weight.detach())
		self.weight_g = nn.Parameter(self.weight.norm(dim = [1, 2], keepdim = True).detach())
		self.register_parameter('weight', None)

	@property
	def weight(self):
		return self.weight_v * self.weight_g / self.weight_v.norm(dim = [1, 2], keepdim = True)

class WeightNormConv1dReLU(nn.Module):
	def __init__(self, *args, **kwargs):
		super(WeightNormConv1dReLU, self).__init__()
		self.temporal_conv = WeightNormConv1d(*args, **kwargs)

	def forward(self, x):
		x = self.temporal_conv(x)
		x = F.relu(x)
		return x

class MeanStdPool1d(nn.Module):
	def __init__(self, kernel_size = None, stride = None, interpolation_mode = 'nearest'):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.interpolation_mode = interpolation_mode

	def forward(self, x):
		kwargs = dict(kernel_size = self.kernel_size, padding = self.kernel_size // 2, stride = self.stride or 1) if self.kernel_size is not None else {}
		mean = F.avg_pool1d(x, **kwargs) if kwargs else F.adaptive_avg_pool1d(x, output_size = 1)
		mean_interpolated = F.interpolate(mean, x.shape[-1], mode = self.interpolation_mode) if kwargs and kwargs['stride'] != 1 else mean
		zero_mean_squared = (x - mean_interpolated) ** 2
		std_squared = F.avg_pool1d(zero_mean_squared, **kwargs) if kwargs else F.adaptive_avg_pool1d(zero_mean_squared, output_size = 1) 
		std = std_squared.sqrt()
		return torch.cat((mean, std), dim = 1)

class XVectorNet(nn.Module):
	def __init__(self, input_dim: int = 24, embedding_dim: int = 512, hidden_dim_small: int = 512, hidden_dim_large: int = 1500, padding_same: bool = True, kernel_size = [5, 3, 3, 1, 1], dilation = [1, 2, 3, 1, 1], kernel_size_pool: typing.Optional[int] = None, stride_pool: typing.Optional[int] = None):
		super().__init__()
		conv_args = lambda k: dict(kernel_size = kernel_size[k], padding = dilation[k] * (kernel_size[k] // 2) if padding_same else 0, dilation = dilation[k], stride = 1)
		self.tdnn = nn.Sequential(
			WeightNormConv1dReLU(input_dim,        hidden_dim_small, **conv_args(0)),
			WeightNormConv1dReLU(hidden_dim_small, hidden_dim_small, **conv_args(1)),
			WeightNormConv1dReLU(hidden_dim_small, hidden_dim_small, **conv_args(2)),
			WeightNormConv1dReLU(hidden_dim_small, hidden_dim_small, **conv_args(3)),
			WeightNormConv1dReLU(hidden_dim_small, hidden_dim_large, **conv_args(4))
		)
		self.pool = MeanStdPool1d(kernel_size = kernel_size_pool, stride = stride_pool)
		self.segment6 = nn.Linear(hidden_dim_large * 2, embedding_dim)
		self.segment7 = nn.Linear(embedding_dim, embedding_dim)
	
	def forward(self, x: torch.Tensor):
		x = self.tdnn(x)

		x = self.pool(x)
		
		x = F.conv1d(x, self.segment6.weight.unsqueeze(-1))
		x = F.relu(x)
		
		x = F.conv1d(x, self.segment7.weight.unsqueeze(-1))
		x = F.relu(x)
		
		return x

