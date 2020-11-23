import sys; sys.path.append('../uis-rnn')
import uisrnn
import argparse
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--val-iteration-interval', type = int, default = 1000) 
parser.add_argument('--speaker-embeddings', default = './embeddings')
parser.add_argument('--transcripts', default = 'dia_dataset/diacalls_short_2020-09-01_2020-10-01.ref')
parser.add_argument('--fraction-train', type = float, default = 0.9)
parser.add_argument('--default-speaker-names', default = '_' + ''.join(chr(ord('A') + i) for i in range(26)))
args, _ = parser.parse_known_args()

train_sequences, train_cluster_ids = [], []
test_sequences, test_cluster_ids = [], []
input_files = os.listdir(args.speaker_embeddings)
cutoff = int(len(input_files) * args.fraction_train)
for i, emb_name in enumerate(input_files):
	features_TC = torch.load(os.path.join(args.speaker_embeddings, emb_name), 'cpu').t().to(torch.float64)
	speaker_id = torch.load(os.path.join(args.transcripts, emb_name.replace('.wav.pt', '.pt')), 'cpu')
	speaker_id = F.interpolate(speaker_id.view(1, 1, -1).to(torch.float32), len(features_TC)).flatten().to(torch.int8)
	(train_sequences if i < cutoff else test_sequences).append(np.asarray(features_TC))
	(train_cluster_ids if i < cutoff else test_cluster_ids).append(np.array([args.default_speaker_names[i] for i in speaker_id.tolist()], dtype = 'U1'))

print('Prepared dataset')

train_sequences, train_cluster_ids = test_sequences[:1], test_cluster_ids[:1]
test_sequences, test_cluster_ids = test_sequences[:1], test_cluster_ids[:1]

model_args, training_args, inference_args = uisrnn.parse_arguments()
model = uisrnn.UISRNN(model_args)

def on_validation_epoch_start(iteration):
	model.rnn_model.eval()
	torch.set_grad_enabled(False)
	print('Validation @', iteration, '. Num test sequences: ', len(test_sequences))
	test_records, predicted_cluster_ids = [], []
	for test_sequence, test_cluster_id in zip(test_sequences, test_cluster_ids):
		predicted_cluster_id = model.predict(test_sequence, inference_args)
		predicted_cluster_ids.append(predicted_cluster_id)
		accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_id.tolist(), predicted_cluster_id)
		test_records.append((accuracy, len(test_cluster_id)))

		print('Ground truth labels:', test_cluster_id)
		print('Predicted    labels:', predicted_cluster_id)

	print('Validation @', iteration, 'Done:', uisrnn.output_result(model_args, training_args, test_records))
	print()
	model.rnn_model.train()
	torch.set_grad_enabled(True)

model.fit(train_sequences, train_cluster_ids, training_args, args.val_iteration_interval, on_validation_epoch_start)
#model.save('model_uisrnn.pt')
