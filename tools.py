# pip install pyannote.metrics
# pip install pyannote.pipeline
# pip install pescador
# pip install optuna
# pip install filelock
# pip install git+https://github.com/pyannote/pyannote-audio
# pip install sortedcontainers simplejson typing_extensions pyannote.database pyannote.metrics
# pip install pyannote.core --no-dependencies

import os
import argparse
import torch
import audio
import models
import vad
import vis
import transcripts


def genref(input_path, output_path, sample_rate, window_size, device, max_duration, debug_audio, html, ext):
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	transcript_refs = []
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		noextname = audio_name[:-len(ext)]
		transcript_path = os.path.join(output_path, noextname + '.json')
		rttm_path = os.path.join(output_path, noextname + '.rttm')

		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = 'float32', __array_wrap__ = torch.tensor)
		speaker_id_ref, speaker_id_ref_ = vad.select_speaker(signal.to(device), silence_absolute_threshold = 0.05, silence_relative_threshold = 0.2, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096, kernel_size_smooth_silence = 4096)
		transcript = transcripts.from_speaker_mask(speaker_id_ref_, sample_rate, audio_path = audio_path)
		if not transcript or len({t['speaker_name'] for t in transcript}) == 1:
			continue
		transcript = [t for t in transcript if t['speaker'] != transcripts.speaker_missing]

		print(transcripts.save(transcript_path, transcript))
		print(transcripts.save(rttm_path, transcript))
		
		if debug_audio:
			debug_signal = torch.cat([signal[..., :speaker_id_ref.shape[-1]], models.common.convert_speaker_id(speaker_id_ref[..., :signal.shape[-1]], to_bipole = True).unsqueeze(0).cpu() * 0.5, speaker_id_ref_[..., :signal.shape[-1]].cpu() * 0.5]).T
			print(audio.write_audio(transcript_path + '.wav', debug_signal, sample_rate, mono = False))

		transcript_refs.append(dict(
			audio_path = audio_path,
			audio_name = audio_name,
			ref = transcript
		))
	if html:
		print(vis.diarization(transcript_refs, os.path.join(output_path, audio_name + '.html'), debug_audio=True))

def genhyppyannote(input_path, output_path, device, batch_size, html, ext, sample_rate, max_duration):
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	model = models.common.PyannoteDiarizationModel(device = device, batch_size = batch_size)
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	
	cmd = subparsers.add_parser('genref')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--window-size', type = float, default = 0.02)
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--max-duration', type = float)
	cmd.add_argument('--audio', dest = 'debug_audio', action = 'store_true')
	cmd.add_argument('--html', action = 'store_true')
	cmd.add_argument('--ext', default = '.mp3')
	cmd.set_defaults(func = genref)

	cmd = subparsers.add_parser('genhyppyannote')
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--batch-size', type = int, default = 8)
	cmd.add_argument('--sample-rate', type = int, default = 16_000)
	cmd.add_argument('--html', action = 'store_true')
	cmd.add_argument('--ext', default = '.mp3.wav')
	cmd.add_argument('--max-duration', type = float)
	cmd.set_defaults(func = genhyppyannote)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)

