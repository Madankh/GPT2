import torch
import torchaudio
from snac import SNAC
import soundfile as sf

class SpeechTokenizer():
    def __init__(self, device='cpu')-> None:
        self.model =torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device))
        self.sample_rate = 24000
        self.device =  device

    def flatten_tensors(self, tensors, seperator=4097):
        """Safely flattens a list of tensors into a flat list of integers."""
        flattened  =[]

        for batch in range(tensors[0].size(0)):
            flattened_list = []
            if len(tensors) == 3:
                for i in range(tensors[0].size()[1]):
                    flattened_list.append(seperator)