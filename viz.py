import os
import json
import matplotlib.pyplot as plt

from utils import *

def hist_plot(manifest_path, output_path):
    '''
    plot a bar plot of the frequency (histogram) for the given col.
    eg: word rate, char rate
    Input:
        - manifest_path: path to the manifest json file
        - col_name selected column name
    '''

    manifest = open(manifest_path, "r").readlines()

    char_rate = []
    word_rate = []

    for idx, item in enumerate(manifest):
        
        item = json.loads(item)

        print(idx, item["audio_filepath"])

        sentence = item["sentence"]
        duration = librosa.core.get_duration(filename=item["audio_filepath"])

        chars = 0
        trans_length = len(sentence.split(" "))
        for i in sentence.split(" "):
            chars = chars + len(i)
        
        ch_rate = chars/duration
        wd_rate = trans_length/duration
        
        char_rate.append(ch_rate)
        word_rate.append(wd_rate)
       
    
    plt.hist(char_rate, bins=50)
    plt.save_fig("/data/chanie/commonvoice/11/rw/test_char_hist.png")

    plt.hist(word_rate, bins=50)
    plt.save_fig("/data/chanie/commonvoice/11/rw/test_word_hist.png")


if __name__ =="__main__":

    manifest_path = "/data/chanie/commonvoice/11/rw/test_manifest_v2.json"
    col_name = ""
    output_path = "/data/chanie/commonvoice/11/rw/test_hist.png"

    hist_plot(manifest_path, output_path)


