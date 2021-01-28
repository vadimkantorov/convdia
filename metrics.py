import json
import torch
import shapes
import argparse
import transcripts


def align_hypref(hyp: shapes.BT, ref: shapes.BT):
	original_order_score = (hyp & ref).sum()
	inversed_order_score = (hyp[(0,2,1),:] & ref).sum()
	if original_order_score > inversed_order_score:
		return hyp, ref
	else:
		return hyp[(0,2,1),:], ref


def der(hyp: shapes.BT, ref: shapes.BT):
	# ignore silence
	false_alarm = (ref[0] & (hyp[1] | hyp[2])).sum().item()
	missed_detection = (hyp[0] & (ref[1] | ref[2])).sum().item()
	confusion = (hyp[1] & ref[2] & ~ref[1] | hyp[2] & ref[1] & ~ref[2]).sum().item()
	total = (~ref[0]).sum().item()
	return (false_alarm + missed_detection + confusion) / total


def ser(hyp: shapes.BT, ref: shapes.BT):
	# ignore silence
	return der(hyp[:, ~ref[0]], ref[:, ~ref[0]])


def main(args):
	data_files = [('ref', args.ref_path), ('hyp', args.hyp_path)]
	examples = None
	for hypref, path in data_files:
		buffer = dict()
		with open(path) as data_file:
			for line in data_file:
				example = json.loads(line)
				if examples is not None and example['audio_path'] not in examples:
					continue
				mask = transcripts.intervals_to_mask(example.pop('markup'), example['sample_rate'], example['duration'])
				if examples is not None:
					buffered = examples[example['audio_path']]
					if example['sample_rate'] > buffered['sample_rate']:
						mask = mask[:, ::int(example['sample_rate'] / buffered['sample_rate'])]
					elif example['sample_rate'] < buffered['sample_rate']:
						mask = torch.nn.functional.interpolate(mask[None, ...], mask.shape[-1] * int(buffered['sample_rate'] / example['sample_rate'])).squeeze()
					buffer[example['audio_path']] = buffered
					buffer[example['audio_path']][hypref] = mask
				else:
					example[hypref] = mask
					buffer[example['audio_path']] = example
		examples = buffer
	examples = list(examples.values())
	with open(args.output_path, 'w') as metrics_file:
		for i, example in enumerate(examples):
			example['hyp'], example['ref'] = align_hypref(example['hyp'], example['ref'])
			example['der'] = der(example['hyp'], example['ref'])
			example['ser'] = ser(example['hyp'], example['ref'])
			example['hyp'] = transcripts.mask_to_intervals(example['hyp'], example['sample_rate'])
			example['ref'] = transcripts.mask_to_intervals(example['ref'], example['sample_rate'])
			metrics_file.write(json.dumps(example, ensure_ascii=False))
			print(f"{i+1}/{len(examples)}. {example['audio_name']} DER:{example['der']:.2f} SER:{example['ser']:.2f}")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ref-path', '-rp', required = True)
	parser.add_argument('--hyp-path', '-hp', required = True)
	parser.add_argument('--output-path', '-o', required = True)
	parser.add_argument('--sample-rate', '-sr', type = int, default = 16_000)

	args = parser.parse_args()
	main(args)
