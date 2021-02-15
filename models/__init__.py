import itertools
import math

import torch
from torch.nn import functional as F

from models.spectral_clustering import SpectralClusteringDiarizationModel, cosine_kernel
from models.pyannote import PyannoteDiarizationModel

# Code is adapted from pyannote.audio. All credit goes to pyannote.audio
# https://github.com/pyannote/pyannote-audio/tree/master/pyannote/audio/models

def rle1d(tensor):
	#_notspeech_ = ~F.pad(speech, [1, 1])
	#channel_i_channel_j = torch.cat([(speech & _notspeech_[..., :-2]).nonzero(), (speech & _notspeech_[..., 2:]).nonzero()], dim = -1)
	#return [dict(begin = i / sample_rate, end = j / sample_rate, channel = channel) for channel, i, _, j in channel_i_channel_j.tolist()]

	assert tensor.ndim == 1
	starts = torch.cat(( torch.tensor([0], dtype = torch.long, device = tensor.device), (tensor[1:] != tensor[:-1]).nonzero(as_tuple = False).add_(1).squeeze(1), torch.tensor([tensor.shape[-1]], dtype = torch.long, device = tensor.device)))
	starts, lengths, values = starts[:-1], (starts[1:] - starts[:-1]), tensor[starts[:-1]]
	return starts, lengths, values


def pdist(A, squared = False, eps = 1e-4):
	normAsq = A.pow(2).sum(dim = -1, keepdim = True)
	res = torch.addmm(normAsq.transpose(-2, -1), A, A.transpose(-2, -1), alpha = -2).add_(normAsq)
	return res.clamp_(min = 0) if squared else res.clamp_(min = eps).sqrt_()


def shrink(tensor, min):
	return torch.where(tensor >= min, tensor, torch.zeros_like(tensor))


def acf(K):
	arange = torch.arange(len(K), device = K.device)
	Z = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
	ACF = torch.zeros_like(arange, dtype = K.dtype).scatter_add_(0, Z.flatten(), K.flatten()).div_(Z.flatten().bincount(minlength = len(arange)))
	M = ACF[Z.flatten()].view_as(Z)
	return ACF, M


def gaussian_kernel(kernel_size, sigma):
	kernel = 1
	meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
	for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
		mean = (size - 1) / 2
		kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
	return (kernel / kernel.sum())[None, None, ...]


def cdist(A, B, squared = False, eps = 1e-4):
	normAsq = A.pow(2).sum(dim = -1, keepdim = True)
	normBsq = B.pow(2).sum(dim = -1, keepdim = True)
	res = torch.addmm(normBsq.transpose(-2, -1), A, B.transpose(-2, -1), alpha = -2).add_(normAsq)
	return res.clamp_(min = 0) if squared else res.clamp_(min = eps).sqrt_()


def kmeans(E, k = 5, num_iter = 10):
	torch.manual_seed(1)
	centroids = E[torch.randperm(len(E), device = E.device)[:k]]
	for i in range(num_iter):
		assignment = cdist(E, centroids, squared = True).argmin(dim = 1)
		centroids.zero_().scatter_add_(0, assignment.unsqueeze(-1).expand(-1, E.shape[-1]), E)
		centroids /= assignment.bincount(minlength = k).unsqueeze(-1)
	return assignment


def reassign_speaker_id(speaker_id, targets):
	perm = min((float((torch.tensor(perm)[speaker_id] - targets).abs().sum()), perm) for perm in itertools.permutations([0, 1, 2]))[1]
	return torch.tensor(perm)[speaker_id]


def convert_speaker_id(speaker_id, to_bipole = False, from_bipole = False):
	k, b = (1 - 3/2, 3 / 2) if from_bipole else (-2, 3) if to_bipole else (None, None)
	return (speaker_id != 0) * (speaker_id * k + b)
