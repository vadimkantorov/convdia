import torch

def speaker_mask(transcript, num_speakers, duration, sample_rate):
	mask = torch.zeros(1 + num_speakers, int(duration * sample_rate), dtype = torch.bool)
	for t in transcript:
		mask[t['speaker'], int(t['begin'] * sample_rate) : int(t['end'] * sample_rate)] = 1
	mask[0] = mask[1] & mask[2]
	return mask

def speaker_error(ref, hyp, num_speakers, sample_rate = 8000, hyp_speaker_mapping = None, ignore_silence_and_overlapped_speech = True):
	assert num_speakers == 2
	duration = transcripts.compute_duration(dict(ref = ref, hyp = hyp))
	ref_mask = speaker_mask(ref,  num_speakers, duration, sample_rate)
	hyp_mask_ = speaker_mask(hyp, num_speakers, duration, sample_rate)

	print('duration', duration)
	vals = []
	for hyp_perm in ([[0, 1, 2], [0, 2, 1]] if hyp_speaker_mapping is None else hyp_speaker_mapping):
		hyp_mask = hyp_mask_[hyp_perm]
		speaker_mismatch = (ref_mask[1] != hyp_mask[1]) | (ref_mask[2] != hyp_mask[2])
		if ignore_silence_and_overlapped_speech:
			silence_or_overlap_mask = ref_mask[1] == ref_mask[2]
			speaker_mismatch = speaker_mismatch[~silence_or_overlap_mask]

		confusion = (hyp_mask[1] & ref_mask[2] & (~ref_mask[1])) | (hyp_mask[2] & ref_mask[1] & (~ref_mask[2]))
		false_alarm = (hyp_mask[1] | hyp_mask[2]) & (~ref_mask[1]) & (~ref_mask[2])
		miss = (~hyp_mask[1]) & (~hyp_mask[2]) & (ref_mask[1] | ref_mask[2])
		total = ref_mask[1] | ref_mask[2]

		confusion, false_alarm, miss, total = [float(x.float().mean()) * duration for x in [confusion, false_alarm, miss, total]]

		print('my', 'confusion', confusion, 'false_alarm', false_alarm, 'miss', miss, 'total', total)
		err = float(speaker_mismatch.float().mean())
		vals.append((err, hyp_perm))

	return min(vals)

def pyannote_der(ref_rttm_path, hyp_rttm_path, metric = None):
	import pyannote.database.util 
	import pyannote.metrics.diarization
	metric = metric if metric is not None else pyannote.metrics.diarization.DiarizationErrorRate()
	ref, hyp = map(pyannote.database.util.load_rttm, [ref_rttm_path, hyp_rttm_path])
	ref, hyp = [next(iter(anno.values())) for anno in [ref, hyp]]
	return metric(ref, hyp)
