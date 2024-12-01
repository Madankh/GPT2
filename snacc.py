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
                    flattened_list.append(tensors[0][batch][i].item())
                    for j in range(2):
                        flattened_list.append(tensors[1][batch][j+i*2].item())
                        for k in  range(2):
                            flattened_list.append(
                                tensors[2][batch][k+j*2+i*4].item()
                            )
            if len(tensors) == 4:
                for i in range(tensors[0].size()[1]):
                    flattened_list.append(seperator)
                    flattened_list.append(tensors[0][batch][i].item())
                    for j in range(2):
                        flattened_list.append(tensors[1][batch][j + i * 2].item())
                        for k in range(2):
                            flattened_list.append(
                                tensors[2][batch][k+j*2+i*4].item()
                            )
                            for l in range(2):
                                flattened_list.append(
                                    tensors[3][batch][l + k*2+j*4+i*8].item()
                                )
                flattened_list.append(seperator)
                flattened.append(flattened_list)
        return flattened
    
    def reconstruct_single_tensors(self, flattened_output, seperator=4097):
        """
        Reconstructs the list of tensors from the flattened output
        """
        def count_elements_between_hashes(lst):
            try:
                # find the index of the first 
                first_index = lst.index(seperator)
                # fine the index of the second '#' after the first 
                second_index = lst.index(seperator, first_index + 1)
                return second_index + first_index

            except ValueError:
                return f"List does not contain two '{seperator}' seperator"
            
        def  remove_elements_before_hash(flattened_list):
            