import torch
import models

weights_path = 'emb_voxceleb/train/X.SpeakerDiarization.VoxCeleb.train/weights/0326.pt'
weights = torch.load(weights_path)

sincnet = dict(stride = [5, 1, 1], waveform_normalize = True, instance_normalize = True)
tdnn = dict(embedding_dim = 512)
embedding = dict(batch_normalize = False)

model = models.SincTDNN(sincnet = sincnet, tdnn = tdnn, embedding = embedding)
model.load_state_dict(weights)
print(model)

waveforms = (torch.rand(16, 4*1024) - 0.5) * 2
print('Before forward')
features = model(waveforms)
print('Before backward')

print(waveforms.shape, features.shape)

features.backward(torch.ones_like(features))
print('After backward')
