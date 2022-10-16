import os
import shutil
from pathlib import Path

import kenlm
from pyctcdecode import build_ctcdecoder
from speechbrain.utils.data_utils import download_file
import gzip

from hw_asr.utils import ROOT_PATH

def setup_lm_decoder(vocab, lm = "3-gram.pruned.1e-7"):
    base_path = ROOT_PATH / "data" / "lm"
    base_lm_path = base_path / lm #/ "lowered" + lm + ".arpa"
    lm_gzip_path = base_lm_path / (lm+".arpa.gz")
    if not lm_gzip_path.exists():
        lm_url = f'http://www.openslr.org/resources/11/{lm}.arpa.gz'
        download_file(lm_url, lm_gzip_path)
    uppercase_lm_path = base_lm_path / (lm+".arpa")
    
    if not uppercase_lm_path.exists():
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
                
    lm_path = base_lm_path / f'lowered_{lm}.arpa'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    
    unigram_path = base_path / "librispeech-vocab.txt"
    if not unigram_path.exists():
        download_file("http://www.openslr.org/resources/11/librispeech-vocab.txt", unigram_path)
    with open(unigram_path) as f:
        unigram_list = [t.lower() for t in f.read().strip().split("\n")]
        
    # kenlm_model = kenlm.Model(str(lm_path))
    decoder = build_ctcdecoder(vocab, str(lm_path), unigram_list)
    
    return decoder
