#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers
import numpy as np


class Dataset:
    """
    sadsadsadsadsa
    """
    def __init__(self):
        """
        sadasdsadsa
        """
        examples, _ = tfds.load(
            'ted_hrlr_translate/pttoen',
            as_supervised=True,
            with_info=True
        )

        self.datatrain = examples['train']
        self.datavalid = examples['validation']
        a, b = self.tokenizedataset(self.datatrain)
        self.tokenizerpt, self.tokenizeren = a, b

    def tokenizedataset(self, data):
        """
        sadsads
        """
        abv = "neuralmind/bertbaseportuguesecased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bertbaseuncased")
        
        ptsen = []
        ensen = []
        
        for pt, en in data:
            ptsen.append(pt.numpy().decode('utf8'))
            ensen.append(en.numpy().decode('utf8'))
        
        tpt = tpt.trainnewfromiterator(ptsen, vocab_size=2**13)
        ten = ten.trainnewfromiterator(ensen, vocab_size=2**13)

        return tpt, ten

    def encode(self, pt, en):
        """
        sadsadsa
        """
        pttxt = pt.numpy().decode('utf8')
        entxt = en.numpy().decode('utf8')
        
        pttok = self.tokenizerpt.encode(pttxt)
        entok = self.tokenizeren.encode(entxt)
        
        pttok = np.array(pttok)
        entok = np.array(entok)
        
        vspt = self.tokenizerpt.vocabsize
        vsen = self.tokenizeren.vocabsize
        
        pttok = np.concatenate([[vspt], pttok, [vspt + 1]])
        entok = np.concatenate([[vsen], entok, [vsen + 1]])
        
        return pttok, entok
