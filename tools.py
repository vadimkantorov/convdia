# pip install pyannote.metrics
# pip install pyannote.pipeline
# pip install pescador
# pip install optuna
# pip install filelock
# pip install git+https://github.com/pyannote/pyannote-audio
#pip install sortedcontainers simplejson typing_extensions pyannote.database pyannote.metrics
#pip install pyannote.core --no-dependencies

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import audio
import shapes
import models
import vis
import transcripts



def ref(input_path, output_path, sample_rate, window_size, device, max_duration, debug_audio, html, ext):
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		noextname = audio_name[:-len(ext)]
		transcript_path = os.path.join(output_path, noextname + '.json')
		rttm_path = os.path.join(output_path, noextname + '.rttm')

		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = 'float32', duration = max_duration)

		speaker_id_ref, speaker_id_ref_ = select_speaker(signal.to(device), silence_absolute_threshold = 0.05, silence_relative_threshold = 0.2, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096, kernel_size_smooth_silence = 4096)

		transcript = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker = speaker, speaker_name = transcripts.default_speaker_names[speaker]) for speaker in range(1, len(speaker_id_ref_)) for begin, duration, mask in zip(*rle1d(speaker_id_ref_[speaker])) if mask == 1]

		if not transcript or len({t['speaker_name'] for t in transcript}) == 1:
			continue
		
		#transcript = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker_name = str(int(speaker)), speaker = int(speaker)) for begin, duration, speaker in zip(*rle1d(speaker_id_ref.cpu()))]

		transcript_without_speaker_missing = [t for t in transcript if t['speaker'] != transcripts.speaker_missing]
		transcripts.save(transcript_path, transcript_without_speaker_missing)
		print(transcript_path)

		transcripts.save(rttm_path, transcript_without_speaker_missing)
		print(rttm_path)
		
		if debug_audio:
			audio.write_audio(transcript_path + '.wav', torch.cat([signal[..., :speaker_id_ref.shape[-1]], convert_speaker_id(speaker_id_ref[..., :signal.shape[-1]], to_bipole = True).unsqueeze(0).cpu() * 0.5, speaker_id_ref_[..., :signal.shape[-1]].cpu() * 0.5]), sample_rate, mono = False)
			print(transcript_path + '.wav')

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)

def hyp(input_path, output_path, device, batch_size, html, ext, sample_rate, max_duration):
	
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	model = PyannoteDiarizationModel(device = device, batch_size = batch_size)
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		noextname = audio_name[:-len(ext)]
		transcript_path = os.path.join(output_path, noextname + '.json')
		rttm_path = os.path.join(output_path, noextname + '.rttm')
	
		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = True, dtype = 'float32', duration = max_duration)
		transcript = model(signal, sample_rate = sample_rate, extra = dict(audio_path = audio_path))
		transcripts.collect_speaker_names(transcript, set_speaker_data = True)
		
		transcripts.save(transcript_path, transcript)
		print(transcript_path)

		transcripts.save(rttm_path, transcript)
		print(rttm_path)

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)

		
def eval(ref, hyp, html, debug_audio, sample_rate = 100):
	if os.path.isfile(ref) and os.path.isfile(hyp):
		print(pyannote_der(ref_rttm_path = ref, hyp_rttm_path = hyp))

	elif os.path.isdir(ref) and os.path.isdir(hyp):
		errs = []
		diarization_transcript = []
		for rttm in os.listdir(ref):
			if not rttm.endswith('.rttm'):
				continue

			print(rttm)
			audio_path = transcripts.load(os.path.join(hyp, rttm).replace('.rttm', '.json'))[0]['audio_path']

			ref_rttm_path, hyp_rttm_path = os.path.join(ref, rttm), os.path.join(hyp, rttm)
			ref_transcript, hyp_transcript = map(transcripts.load, [ref_rttm_path, hyp_rttm_path])
			ser_err, hyp_perm = speaker_error(ref = ref_transcript, hyp = hyp_transcript, num_speakers = 2, sample_rate = sample_rate, ignore_silence_and_overlapped_speech = True)
			der_err, *_ = speaker_error(ref = ref_transcript, hyp = hyp_transcript, num_speakers = 2, sample_rate = sample_rate, ignore_silence_and_overlapped_speech = False)
			der_err_ = pyannote_der(ref_rttm_path = ref_rttm_path, hyp_rttm_path = hyp_rttm_path)
			transcripts.remap_speaker(hyp_transcript, hyp_perm)

			err = dict(
				ser = ser_err,
				der = der_err,
				der_ = der_err_
			)
			diarization_transcript.append(dict(
				audio_path = audio_path,
				audio_name = transcripts.audio_name(audio_path),
				ref = ref_transcript, 
				hyp = hyp_transcript,
				**err
			))
			print(rttm, '{ser:.2f}, {der:.2f} | {der_:.2f}'.format(**err))
			print()
			errs.append(err)
		print('===')
		print({k : sum(e) / len(e) for k in errs[0] for e in [[err[k] for err in errs]]})
		
		if html:
			print(vis.diarization(sorted(diarization_transcript, key = lambda t: t['ser'], reverse = True), html, debug_audio))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	
	cmd = subparsers.add_parser('ref')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--window-size', type = float, default = 0.02)
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--max-duration', type = float)
	cmd.add_argument('--audio', dest = 'debug_audio', action = 'store_true')
	cmd.add_argument('--html', action = 'store_true')
	cmd.add_argument('--ext', default = '.mp3')
	cmd.set_defaults(func = ref)
	
	cmd = subparsers.add_parser('hyp')
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--batch-size', type = int, default = 8)
	cmd.add_argument('--sample-rate', type = int, default = 16_000)
	cmd.add_argument('--html', action = 'store_true')
	cmd.add_argument('--ext', default = '.mp3.wav')
	cmd.add_argument('--max-duration', type = float)
	cmd.set_defaults(func = hyp)
	
	cmd = subparsers.add_parser('eval')
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--html', default = 'data/diarization.html')
	cmd.add_argument('--audio', dest = 'debug_audio', action = 'store_true')
	cmd.set_defaults(func = eval)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)

