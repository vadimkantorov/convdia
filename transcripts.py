import audio

time_missing = -1
speaker_missing = 0

default_speaker_names = '_' + ''.join(chr(ord('A') + i) for i in range(26))
default_channel_names = {channel_missing : 'channel_', 0 : 'channel0', 1 : 'channel1'}

def summary(transcript, ij = False):
	res = dict(
		begin = min(w.get('begin', 0.0) for w in transcript),
		end = max(w.get('end', 0.0) for w in transcript),
		i = min([w['i'] for w in transcript if 'i' in w] or [0]),
		j = max([w['j'] for w in transcript if 'j' in w] or [0])
	) if len(transcript) > 0 else dict(begin = time_missing, end = time_missing, i = 0, j = 0)
	if not ij:
		del res['i']
		del res['j']
	return res

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

def segment_by_time(transcript, max_segment_seconds, break_on_speaker_change = True, break_on_channel_change = True):
	transcript = [t for t in transcript if t['begin'] != time_missing and t['end'] != time_missing]
	ind_last_taken = -1
	for j, t in enumerate(transcript):
		first, last = ind_last_taken == -1, j == len(transcript) - 1

		if last or (t['end'] - transcript[ind_last_taken + 1]['begin'] > max_segment_seconds) \
				or (break_on_speaker_change and j >= 1 and t['speaker'] != transcript[j - 1]['speaker']) \
				or (break_on_channel_change and j >= 1 and t['channel'] != transcript[j - 1]['channel']):

			ind_last_taken, transcript_segment = take_between(transcript, ind_last_taken, t, first, last, sort_by_time=False)
			if transcript_segment:
				yield transcript_segment

def take_between(transcript, ind_last_taken, t, first, last, sort_by_time = True, soft = True, set_speaker = False):
	if sort_by_time:
		lt = lambda a, b: a['end'] < b['begin']
		gt = lambda a, b: a['end'] > b['begin']
	else:
		lt = lambda a, b: sort_key(a) < sort_key(b)
		gt = lambda a, b: sort_key(a) > sort_key(b)

	if soft:
		res = [(k, u) for k, u in enumerate(transcript)	if (first or ind_last_taken < 0 or lt(transcript[ind_last_taken], u)) and (last or gt(t, u))]
	else:
		intersects = lambda t, begin, end: (begin <= t['end'] and t['begin'] <= end)
		res = [(k, u) for k, u in enumerate(transcript) if ind_last_taken < k and intersects(t, u['begin'], u['end'])] if t else []

	ind_last_taken, transcript = zip(*res) if res else ([ind_last_taken], [])

	if set_speaker:
		for u in transcript:
			u['speaker'] = t.get('speaker', speaker_missing)
			if t.get('speaker_name') is not None:
				u['speaker_name'] = t['speaker_name']

	return ind_last_taken[-1], list(transcript)

def save(data_path, transcript):
	with open(data_path, 'w') as f:

		if data_path.endswith('.json'):
			json.dump(transcript, f, ensure_ascii = False, sort_keys = True, indent = 2)

		elif data_path.endswith('.rttm'):
			f.writelines('SPEAKER {audio_name} 1 {begin:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n'.format(audio_name = audio_name(t), begin = t['begin'], duration = compute_duration(t), speaker = t['speaker']) for t in transcript if t['speaker'] != speaker_missing)

	return data_path

def load(data_path):
	assert os.path.exists(data_path)

	if data_path.endswith('.rttm'):
		with open(data_path) as f:
			transcript = [dict(audio_name = splitted[1], begin = float(splitted[3]), end = float(splitted[3]) + float(splitted[4]), speaker_name = splitted[7]) for splitted in map(str.split, f)]

	elif data_path.endswith('.json') or data_path.endswith('.json.gz'):
		with open_maybe_gz(data_path) as f:
			transcript = json.load(f)

	elif os.path.exists(data_path + '.json'):
		with open(data_path + '.json') as f:
			transcript = json.load(f)
			for t in transcript:
				t['audio_path'] = data_path
	else:
		transcript = [dict(audio_path = data_path)]

	return transcript

def open_maybe_gz(data_path, mode = 'r'):
	return gzip.open(data_path, mode + 't') if data_path.endswith('.gz') else open(data_path, mode)
