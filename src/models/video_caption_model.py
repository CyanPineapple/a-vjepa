import torch.nn as nn
class VideoCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VideoCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, video_frames, captions):
        # Encode video frames
        features = self.encoder(video_frames)
        # Decode to generate captions
        outputs = self.decoder(features, captions)
        return outputs
