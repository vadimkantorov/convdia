import math
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

class Scaling(nn.Module):
	def __init__(self, n_features : int, method : typing.Literal['fixed', 'logistic']):
		super().__init__()
		self.method = method
		self.linear = nn.Linear(1, 1, bias = True) if self.method == 'logistic' else nn.Identity()
		self.activation = nn.Sigmoid() if self.method == 'logistic' else nn.Identity()

	def forward(self, features : torch.Tensor):
		if self.method is None:
			return features
		norm = features.norm(p = 2, dim = -1, keepdim = True)
		new_norm = self.activation(self.linear(norm)) if self.method == 'logistic' else 1.0
		return new_norm / (norm + 1e-6) * features

class Embedding(nn.Module):
	def __init__(self, n_features : int, batch_normalize : bool = False, scale : str = None):
		super().__init__()
		self.batch_normalize_ = nn.BatchNorm1d(n_features, eps = 1e-5, momentum = 0.1, affine = False) if batch_normalize else nn.Identity()
		self.scaling = Scaling(n_features, method = scale)

	def forward(self, embedding : torch.Tensor):
		return self.scaling(self.batch_normalize_(embedding))

class SincTDNN(nn.Module):
	def __init__(self, sincnet: typing.Optional[dict] = None, tdnn: typing.Optional[dict] = None, embedding: typing.Optional[dict] = None):
		super().__init__()
		self.sincnet_ = SincNet(**sincnet)
		n_features = self.sincnet_.out_channels[-1]
		self.tdnn_ = XVectorNet(n_features, **tdnn)
		self.embedding_ = Embedding(n_features, **embedding)

	def forward(self, waveforms : torch.Tensor):
		output = self.sincnet_(waveforms)
		output = self.tdnn_(output)
		return self.embedding_(output)

class SincConv1d(nn.Conv1d):
	def __init__(
		self,
		in_channels,
		out_channels,
		kernel_size,
		sample_rate=16000,
		min_low_hz=50,
		min_band_hz=50,
		**kwargs,
		):	
		super().__init__(in_channels, out_channels, kernel_size, **kwargs)

		to_mel = lambda hz: 2595 * math.log10(1 + hz / 700)
		to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
		ediff1d = lambda t: t[1:] - t[:-1]

		self.sample_rate = sample_rate
		self.min_low_hz = min_low_hz
		self.min_band_hz = min_band_hz

		low_hz = 30
		high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)

		hz = to_hz(torch.linspace(to_mel(low_hz), to_mel(high_hz), out_channels + 1))

		self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
		self.band_hz_ = nn.Parameter(torch.Tensor(ediff1d(hz)).view(-1, 1))

		# Half Hamming half window
		n_lin = torch.linspace(0, kernel_size / 2 - 1, steps=int((kernel_size / 2)))
		self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
		
		n = (kernel_size - 1) / 2.0
		self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate

	@property
	def weight_computed(self):
		low = self.min_low_hz + torch.abs(self.low_hz_)
		high = clamp(low + self.min_band_hz + torch.abs(self.band_hz_)).clamp(min = self.min_low_hz, max = self.sample_rate / 2)
		band = (high - low)[:, 0]

		f_times_t_low = torch.matmul(low, self.n_)
		f_times_t_high = torch.matmul(high, self.n_)

		band_pass_left = (f_times_t_high.sin() - f_times_t_low.sin()) / (self.n_ / 2) * self.window_
		band_pass_center = 2 * band.view(-1, 1)
		band_pass_right = band_pass_left.flip(dims=[1])

		band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
		band_pass = band_pass / (2 * band[:, None])
		return band_pass.view_as(self.weight)

	def forward(self, x):
		self.weight.copy_(self.weight_computed)
		return super().forward(x)
		

class SincNet(nn.Module):
	def __init__(
		self,
		waveform_normalize=True,
		sample_rate=16000,
		min_low_hz=50,
		min_band_hz=50,
		out_channels=[80, 60, 60],
		kernel_size: typing.List[int] = [251, 5, 5],
		stride=[1, 1, 1],
		max_pool=[3, 3, 3],
		instance_normalize=True,
		activation="leaky_relu",
		dropout=0.0,
	):
		super().__init__()
		self.out_channels = out_channels
		self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine = True) if waveform_normalize else nn.Identity()
		self.activation_ = nn.LeakyReLU(negative_slope = 0.2) if activation == 'leaky_relu' else nn.Identity()
		self.dropout_ = nn.Dropout(p = dropout) if dropout else nn.Identity()
		self.conv1d_ = nn.ModuleList()
		self.max_pool1d_ = nn.ModuleList()
		self.instance_norm1d_ = nn.ModuleList()

		config = zip(out_channels, kernel_size, stride, max_pool)
		in_channels = 1
		for i, (out_channels, kernel_size, stride, max_pool) in enumerate(config):
			conv_kwargs = dict(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 0, dilation = 1, groups = 1)
			self.conv1d_.append(nn.Conv1d(**conv_kwargs, bias = True) if i > 0 else SincConv1d(**conv_kwargs, bias = False, sample_rate = sample_rate, min_low_hz = min_low_hz, min_band_hz = min_band_hz))
			self.max_pool1d_.append(nn.MaxPool1d(max_pool, stride=max_pool, padding=0, dilation=1))
			self.instance_norm1d_.append(nn.InstanceNorm1d(out_channels, affine=True) if instance_normalize else nn.Identity())
			in_channels = out_channels

	def forward(self, waveforms):
		output = waveforms.transpose(1, 2)
		output = self.waveform_normalize_(output)
		for i, (conv1d, max_pool1d) in enumerate(zip(self.conv1d_, self.max_pool1d_)):
			output = conv1d(output)
			output = torch.abs(output) if i == 0 else output
			output = max_pool1d(output)
			output = self.instance_norm1d_[i](output)
			output = self.activation_(output)
			output = self.dropout_(output)

		return output.transpose(1, 2)

class TDNN(nn.Module):
	def __init__(
		self,
		context: list,
		input_channels: int,
		output_channels: int,
		full_context: bool = True,
	):
		super(TDNN, self).__init__()
		context = sorted(context)

		if full_context:
			kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
			dilation = 1
		else:
			dilation = context[1] - context[0]
			kernel_size = len(context)
		
		self.temporal_conv = torch.nn.utils.weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size, dilation = dilation))

	def forward(self, x):
		x = self.temporal_conv(torch.transpose(x, 1, 2))
		return F.relu(torch.transpose(x, 1, 2))

class StatsPool(nn.Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
		return torch.cat((mean, std), dim=1)

class XVectorNet(nn.Module):
	def __init__(self, input_dim: int = 24, embedding_dim: int = 512):
		super(XVectorNet, self).__init__()
		frame1 = TDNN(
			context=[-2, 2],
			input_channels=input_dim,
			output_channels=512,
			full_context=True,
		)

		frame2 = TDNN(
			context=[-2, 0, 2],
			input_channels=512,
			output_channels=512,
			full_context=False,
		)
		frame3 = TDNN(
			context=[-3, 0, 3],
			input_channels=512,
			output_channels=512,
			full_context=False,
		)
		frame4 = TDNN(context=[0], input_channels=512, output_channels=512, full_context=True)
		frame5 = TDNN(context=[0], input_channels=512, output_channels=1500, full_context=True)
		self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4, frame5, StatsPool())
		self.segment6 = nn.Linear(3000, embedding_dim)
		self.segment7 = nn.Linear(embedding_dim, embedding_dim)
		self.embedding_dim = embedding_dim
	
	def forward(self, x: torch.Tensor, return_intermediate: typing.Optional[str] = None):
		x = self.tdnn(x)
		if return_intermediate == 'stats_pool': return x
		x = self.segment6(x)
		if return_intermediate == 'segment6': return x
		x = self.segment7(F.relu(x))
		return F.relu(x)
