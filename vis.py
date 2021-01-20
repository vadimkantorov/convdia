import os
import collections
import glob
import json
import io
import sys
import math
import typing
import itertools
import argparse
import base64
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import audio
import transcripts

meta_charset = '<meta charset="UTF-8">'

onclick_img_script = '''
function onclick_img(evt)
{
	const img = evt.target;
	const dim = img.getBoundingClientRect();
	let begin = (evt.clientX - dim.left) / dim.width;
	let relative = true;
	if(img.dataset.begin != null && img.dataset.begin != '' && img.dataset.end != null && img.dataset.end != '')
	{
		begin = parseFloat(img.dataset.begin) + (parseFloat(img.dataset.end) - parseFloat(img.dataset.begin)) * begin;
		relative = false;
	}
	const channel = img.dataset.channel || 0;
	play(evt, channel, begin, 0, false);
}
'''

onclick_svg_script = '''
function onclick_svg(evt)
{
	const rect = evt.target;
	const channel = rect.dataset.channel || 0;
	play(evt, channel, parseFloat(rect.dataset.begin), parseFloat(rect.dataset.end));
}
'''

play_script = '''
var playTimeStampMillis = 0.0;

function download_audio(evt, channel)
{
	const a = evt.target;
	a.href = document.getElementById(`audio${channel}`).src;
	return true;
}

function play(evt, channel, begin, end, relative)
{
	Array.from(document.querySelectorAll('audio')).map(audio => audio.pause());
	const audio = document.querySelector(`#audio${channel}`);
	if(relative)
		[begin, end] = [begin * audio.duration, end * audio.duration];
	audio.currentTime = begin;
	audio.dataset.endTime = end;
	playTimeStampMillis = evt.timeStamp;
	audio.play();
	return false;
}

function onpause_(evt)
{
	if(evt.timeStamp - playTimeStampMillis > 10)
		evt.target.dataset.endTime = null;
}

'''

channel_colors = ['violet', 'lightblue']
speaker_colors = ['gray', 'violet', 'lightblue']
#speaker_colors = ['gray', 'red', 'blue']

def diarization(diarization_transcript, html_path, debug_audio):
	with open(html_path, 'w') as html:
		html.write('<html><head>' + meta_charset + '<style>.nowrap{white-space:nowrap} table {border-collapse:collapse} .border-hyp {border-bottom: 2px black solid}</style></head><body>\n')
		html.write(f'<script>{play_script}</script>\n')
		html.write(f'<script>{onclick_img_script}</script>')
		html.write('<table>\n')
		html.write('<tr><th>audio_name</th><th>duration</th><th>refhyp</th><th>ser</th><th>der</th><th>der_</th><th>audio</th><th>barcode</th></tr>\n')
		avg = lambda l: sum(l) / len(l)
		html.write('<tr class="border-hyp"><td>{num_files}</td><td>{total_duration:.02f}</td><td>avg</td><td>{avg_ser:.02f}</td><td>{avg_der:.02f}</td><td>{avg_der_:.02f}</td><td></td><td></td></tr>\n'.format(
			num_files = len(diarization_transcript),
			total_duration = sum(map(transcripts.compute_duration, diarization_transcript)),
			avg_ser = avg([dt['ser'] for dt in diarization_transcript]) if all('ser' in dt for dt in diarization_transcript) else -1.0,
			avg_der = avg([dt['der'] for dt in diarization_transcript]) if all('der' in dt for dt in diarization_transcript) else -1.0,
			avg_der_ = avg([dt['der_'] for dt in diarization_transcript]) if all('der_' in dt for dt in diarization_transcript) else -1.0
		))
		for i, dt in enumerate(diarization_transcript):
			audio_html = fmt_audio(dt['audio_path'], channel = i) if debug_audio else ''
			begin, end = 0.0, transcripts.compute_duration(dt)
			for refhyp in ['ref', 'hyp']:
				if refhyp in dt:
					html.write('<tr class="border-{refhyp}"><td class="nowrap">{audio_name}</td><td>{end:.02f}</td><td>{refhyp}</td><td>{ser:.02f}</td><td>{der:.02f}</td><td>{der_:.02f}</td><td rospan="{rowspan}">{audio_html}</td><td>{barcode}</td></tr>\n'.format(audio_name = dt['audio_name'], audio_html = audio_html if refhyp == 'ref' else '', rowspan = 2 if refhyp == 'ref' else 1, refhyp = refhyp, end = end, ser = dt.get('ser', -1.0), der = dt.get('der', -1.0), der_ = dt.get('der_', -1.0), barcode = fmt_img_speaker_barcode(dt[refhyp], begin = begin, end = end, onclick = None if debug_audio else '', dataset = dict(channel = i))))

		html.write('</table></body></html>')
	return html_path

def fmt_img_speaker_barcode(transcript, begin = None, end = None, colors = speaker_colors, onclick = None, dataset = {}):
	if begin is None:
		begin = 0
	if end is None:
		end = max(t['end'] for t in transcript)
	if onclick is None:
		onclick = 'onclick_img(event)'
	color = lambda s: colors[s] if s < len(colors) else transcripts.speaker_missing

	plt.figure(figsize = (8, 0.2))
	plt.xlim(begin, end)
	plt.yticks([])
	plt.axis('off')
	for t in transcript:
		plt.axvspan(t['begin'], t['end'], color = color(t.get('speaker', transcripts.speaker_missing)))
	plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
	buf = io.BytesIO()
	plt.savefig(buf, format = 'jpg', dpi = 150, facecolor = color(transcripts.speaker_missing))
	plt.close()
	uri_speaker_barcode = base64.b64encode(buf.getvalue()).decode()
	dataset = ' '.join(f'data-{k}="{v}"' for k, v in dataset.items())
	return f'<img onclick="{onclick}" src="data:image/jpeg;base64,{uri_speaker_barcode}" style="width:100%" data-begin="{begin}" data-end="{end}" {dataset}></img>'


def fmt_svg_speaker_barcode(transcript, begin, end, colors = speaker_colors, max_segment_seconds = 60, onclick = None):
	if onclick is None:
		onclick = 'onclick_svg(event)'
	color = lambda s: colors[s] if s < len(colors) else transcripts.speaker_missing
	html = ''
	
	segments = transcripts.segment_by_time(transcript, max_segment_seconds = max_segment_seconds, break_on_speaker_change = False, break_on_channel_change = False)
	
	for segment in segments:
		summary = transcripts.summary(segment)
		duration = transcripts.compute_duration(summary)
		if duration <= max_segment_seconds:
			duration = max_segment_seconds
		header = '<div style="width: 100%; height: 15px; border: 1px black solid"><svg viewbox="0 0 1 1" style="width:100%; height:100%" preserveAspectRatio="none">'
		body = '\n'.join('<rect data-begin="{begin}" data-end="{end}" x="{x}" width="{width}" height="1" style="fill:{color}" onclick="{onclick}"><title>speaker{speaker} | {begin:.2f} - {end:.2f} [{duration:.2f}]</title></rect>'.format(onclick = onclick, x = (t['begin'] - summary['begin']) / duration, width = (t['end'] - t['begin']) / duration, color = color(t['speaker']), duration = transcripts.compute_duration(t), **t) for t in transcript)
		footer = '</svg></div>'
		html += header + body + footer
	return html

def audio_data_uri(audio_path, sample_rate = None, audio_backend = 'scipy', audio_format = 'wav'):
	data_uri = lambda audio_format, audio_bytes: f'data:audio/{audio_format};base64,' + base64.b64encode(audio_bytes).decode()
	
	if isinstance(audio_path, str):
		assert audio_path.endswith('.wav')
		audio_bytes, audio_format = open(audio_path, 'rb').read(), 'wav'
	else:
		audio_bytes = audio.write_audio(io.BytesIO(), audio_path, sample_rate, backend = audio_backend, format = audio_format).getvalue()
		
	return data_uri(audio_format = audio_format, audio_bytes = audio_bytes)

def fmt_audio(audio_path, channel = 0):
	return f'<audio id="audio{channel}" style="width:100%" controls src="{audio_data_uri(audio_path)}"></audio>\n'

