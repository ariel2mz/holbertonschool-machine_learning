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
            'ted_hrlr_translate/pt_to_en',
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
        abv = "neuralmind/bert-base-portuguese-cased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        
        ptsen = []
        ensen = []
        
        for pt, en in data.take(1000):
            ptsen.append(pt.numpy().decode('utf-8'))
            ensen.append(en.numpy().decode('utf-8'))
        
        tpt = tpt.train_new_from_iterator(ptsen, vocab_size=2**13)
        ten = ten.train_new_from_iterator(ensen, vocab_size=2**13)

        return tpt, ten

    def encode(self, pt, en):
        """
        sadsadsa
        """
        pttxt = pt.numpy().decode('utf-8')
        entxt = en.numpy().decode('utf-8')
        
        pttok = self.tokenizerpt.encode(pttxt)
        entok = self.tokenizeren.encode(entxt)
        
        vspt = self.tokenizerpt.vocab_size
        vsen = self.tokenizeren.vocab_size
        
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
