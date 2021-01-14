# [WIP]
Simplification of [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker embedding model for TorchScript export.

Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) [emb_voxeceleb](https://raw.githubusercontent.com/pyannote/pyannote-audio-hub/master/models/emb_voxceleb.zip) TorchHub weights.

We got:
1. Download 2-channel audio dataset from Mongo
2. Generate ground truth from 2-channel audio + primitive VAD + functions for evaluating diarization quality (der + speaker error)
3. Imported PyAnnote model with weights pretrained on VoxCeleb (speaker embeddings)
4. Wang Affinity for spectral clustering. Speaker Diarization with LSTM, Wang et al, https://arxiv.org/abs/1710.10468, https://github.com/wq2012/SpectralCluster/blob/master/spectralcluster/spectral_clusterer.py
5. Basic Spectral Clustering implementation
6. Jupyter Notebook showcasing affinity for short and long English audio
