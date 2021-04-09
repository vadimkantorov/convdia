import math
import torch
import audio
import shapes


speaker_missing = 0


def intervals_to_mask(intervals, sample_rate: int, duration: float):
	mask = torch.zeros(max(x['speaker'] for x in intervals) + 1, math.ceil(sample_rate * duration), dtype = torch.bool)
	for interval in intervals:
		mask[interval['speaker'], int(interval['begin'] * sample_rate): int(interval['end'] * sample_rate)] = True
	return mask


def mask_to_intervals(speaker_masks: shapes.BT, sample_rate: int):
	intervals = []
	for speaker, mask in enumerate(speaker_masks):
		interval = None
		for i, sample in enumerate(mask):
			if sample and interval is None:
				interval = dict(speaker = speaker, begin = i / sample_rate)
			elif not sample and interval is not None:
				interval['end'] = i / sample_rate
				intervals.append(interval)
				interval = None
		if interval is not None:
			interval['end'] = i / sample_rate
			intervals.append(interval)
	intervals = sorted(intervals, key = lambda x: x['end'])
	return intervals


def compute_duration(t, hours = False):
	seconds = None

	if 'begin' in t or 'end' in t:
		seconds = t.get('end', 0) - t.get('begin', 0)
	elif 'hyp' in t or 'ref' in t:
		seconds = max(t_['end'] for k in ['hyp', 'ref'] for t_ in t.get(k, []))
	elif 'audio_path' in t:
		seconds = audio.compute_duration(t['audio_path'])

	assert seconds is not None

	return seconds / (60 * 60) if hours else seconds
