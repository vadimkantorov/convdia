import torch
import models

weights_path = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt'
weights = torch.load(weights_path)

sincnet = dict(stride = [5, 1, 1], waveform_normalize = True, instance_normalize = True)
tdnn = dict(embedding_dim = 512)
embedding = dict(batch_normalize = False)

model = models.SincTDNN(sincnet = sincnet, tdnn = tdnn, embedding = embedding)
model.load_state_dict(weights, strict = False)
print(model)
