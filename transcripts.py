import math
import torch
import audio
import shapes


speaker_missing = 0


def transcript_to_mask(transcript, sample_rate: int, duration: float):
	num_speakers = max(t['speaker'] for t in transcript)+1
	mask = torch.zeros(max(3, num_speakers), math.ceil(sample_rate * duration), dtype = torch.bool)
	for t in transcript:
		mask[t['speaker'], int(t['begin'] * sample_rate): int(t['end'] * sample_rate)] = True
	return mask


def mask_to_transcript(speaker_masks: shapes.BT, sample_rate: int):
	transcript = []
	for speaker, mask in enumerate(speaker_masks):
		t = None
		for i, sample in enumerate(mask):
			if sample and t is None:
				t = dict(speaker = speaker, begin = i / sample_rate)
			elif not sample and t is not None:
				t['end'] = i / sample_rate
				transcript.append(t)
				t = None
		if t is not None:
			t['end'] = i / sample_rate
			transcript.append(t)
	transcript = sorted(transcript, key = lambda x: x['end'])
	return transcript


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
