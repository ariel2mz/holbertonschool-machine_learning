#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


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
        
        self.datatrain = self.datatrain.map(self.tfencode)
        self.datavalid = self.datavalid.map(self.tfencode)

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
        
        vspt = self.tokenizerpt.vocabsize
        vsen = self.tokenizeren.vocabsize
        
        pttok = [vspt] + pttok + [vspt + 1]
        entok = [vsen] + entok + [vsen + 1]
        
        return pttok, entok

    def tfencode(self, pt, en):
        """
        sadsadsa
        """
        resultpt, resulten = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        resultpt.set_shape([None])
        resulten.set_shape([None])
        
        return resultpt, resulten
