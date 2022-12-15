import os
import re
from tracemalloc import start
import unidecode
import subprocess
import json
import librosa
import time

import pandas as pd


def to_wav(path, meta_df, dest_dir):
    '''
    Convert raw mp3 file to wav format

    Input:
        - path: root path for the dataframe and clips folder. Clips folder contains raw audio mp3 dataset.
        - meta_df: meta data for the audiofiles (test.tsv). Should be inside the root path
        - dest_dir: Output directory for the wav audio output
    '''

    test_df = pd.read_csv(f"{path}/{meta_df}", sep="\t")

    for _, item in enumerate(test_df["path"]):
        wav_path = f"{path}/{dest_dir}/{item[:-4]}.wav"
        cmd = ["sox", f"{path}/clips/{item}", wav_path]
        subprocess.run(cmd)

def create_manifest(df, manifest_path, PATH="/data/chanie/commonvoice/11/rw", transcription_df=None):
    '''
    Inputs:
        - df: a dataframe containing dataset information (sentence, audio path)
        - PATH [Optional]: root path for the audio files folder. This folder should contain clips for audio files
        - transcription_df [Optional]: a dataframe for transcription.
            - Columns: transcription, audio path 
    '''
    paths = df['path']
    transcriptions = df['sentence']

    from_idx = int(0.5*len(transcriptions))
    size = len(transcriptions)# - from_idx

    with open(manifest_path, 'w') as fout:
        for idx, (path, transcription) in enumerate(zip(paths, transcriptions)):
            
            start_time = time.time()

            transcript = transcription.strip()

            file_id = path
            audio_path = os.path.join(PATH, "clips", file_id)

            duration = librosa.core.get_duration(filename=audio_path)

            chars = 0
            trans_length = len(transcript.split(" "))
            for i in transcript.split(" "):
                chars = chars + len(i)
            
            try:
                char_rate = chars/duration
                word_rate = trans_length/duration
            except:
                print("Err", duration)

            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": remove_special_characters(transcript),
                "char_rate_calc": char_rate,
                "word_rate_calc": word_rate
            }
            if transcription_df is not None:
                metadata["pred_text"] = transcription.lower()

            json.dump(metadata, fout)
            fout.write('\n')

            elapsed = time.time() - start_time
            if idx % 1000 == 0:
                rem_time = ((size - idx) * elapsed)/3600
                print(f"[{idx}/{size}] Remaining Time: {rem_time:.2f} hrs")

def remove_special_characters(text):
    '''
    Input:
        - text: a text input

    Source: https://huggingface.co/lucio/wav2vec2-large-xlsr-kinyarwanda-apostrophied
    '''
    
    chars_to_ignore_regex = r'[!"#$%&()*+,./:;<=>?@\[\]\\_{}|~£¤¨©ª«¬®¯°·¸»¼½¾ðʺ˜˝ˮ‐–—―‚“”„‟•…″‽₋€™−√�]'

    text = re.sub(r'[ʻʽʼ‘’´`]', r"'", text)  # normalize apostrophes
    text = re.sub(chars_to_ignore_regex, "", text).lower().strip()  # remove all other punctuation
    text = re.sub(r"([b-df-hj-np-tv-z])' ([aeiou])", r"\1'\2", text)  # remove spaces where apostrophe marks a deleted vowel
    text = re.sub(r"(-| '|' |  +)", " ", text)  # treat dash and other apostrophes as word boundary
    
    # from https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/examples/kinyarwanda_asr.rst
    # not necessary
    # text = re.sub(r" '", " ", text)  # delete apostrophes at the beginning of word
    # text = re.sub(r"' ", " ", text)  # delete apostrophes at the end of word
    # text = re.sub(r" +", " ", text)  # merge multiple spaces
    
    text = unidecode.unidecode(text)  # strip accents from loanwords
    
    return text


def clean_data(manifest_path, output_path, threshold=3):

    # with open(manifest_path) as manifest:
    manifest = open(manifest_path)

    with open(output_path, 'w') as fout:
        
        items = manifest.readlines()
        
        for idx, item in enumerate(items):
            
            item = json.loads(item)

            print(f"{idx}/{len(items)}")
            
            # if item["word_rate_calc"] <= threshold:
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": item["audio_filepath"],
                "duration": item["duration"],
                "text": remove_special_characters(item["text"]) # TODO add remove special characters
            }

            if item.get("pred_text") is not None:
                # pred = re.sub("\.", "", item["pred_text"])
                # pred = item["pred_text"]
                pred = remove_special_characters(item["pred_text"])
                metadata["pred_text"] = pred

            json.dump(metadata, fout)
            fout.write('\n')


def combine_transcription(manifest_path, output_path, transcription_df):
    # with open(manifest_path) as manifest:
    manifest = open(manifest_path)

    with open(output_path, 'w') as fout:
        
        items = manifest.readlines()
        
        for idx, item in enumerate(items):
            
            item = json.loads(item)

            print(f"{idx}/{len(items)}")
            
            # if item["word_rate_calc"] <= threshold:
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": item["audio_filepath"],
                "duration": item["duration"],
                "text": item["text"],
                "pred_text": transcription_df[transcription_df["path"] ==  item["audio_filepath"]].iloc[0]["transcription"]
            }

            json.dump(metadata, fout)
            fout.write('\n')

def test(f="/data/chanie/asr/test_decoded_finetune_final.json"):
    manifest = open(f)
    items = manifest.readlines()
    for idx, item in enumerate(items):
        item = json.loads(item)
        item["audio_filepath"].split()


# python NeMo/tools/speech_data_explorer/data_explorer.py manifest.json
# Convert tsv info to json for processing and training
if __name__ == "__main__":
    import pandas as pd
    import json
    import csv

    PATH = "/data/chanie/commonvoice/11/rw"
    # train_manifest.json, test_manifest.json
    MANIFEST_NAME = f"{PATH}/manifest_exp/train.json"

    # df = pd.read_csv(f"{PATH}/train.tsv", sep="\t", encoding="utf-8", quoting = csv.QUOTE_NONE)
    df = pd.read_csv(f"{PATH}/train.tsv", sep="\t", encoding="utf-8", quoting = csv.QUOTE_NONE)
    transcription_df = pd.read_csv("/data/chanie/asr/test_transcription__finetune_ckpt.csv") # this is for test.tsv

    # create_manifest(df, MANIFEST_NAME, transcription_df=None)

    # manifest files with calculated word rate and char rate vals
    # manifest_files = ["test_manifest_v3.json", "dev_manifest.json", "train_manifest.json"]
    # manifest_files = ["manifest_exp/test_manifest_raw.json"]
    # for file in manifest_files:
    #     MANIFEST_NAME = f"{PATH}/{file}"
    #     clean_data(MANIFEST_NAME, f"{PATH}/{file[:-5]}_with_full_stop.json")
    #     print("Done", file)

    # combine_transcription(
    #     "/data/chanie/asr/manifest/test_manifest_clean_decoded.json",
    #     "/data/chanie/asr/test_decoded_finetune_final.json",
    #     transcription_df
    # )

    test()