import os
import copy
import json
import pandas as pd

import nemo.collections.asr as nemo_asr

# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_rw_conformer_ctc_large")

def get_audio_path(manifest_path):
    '''
    Accepts manifest path to return the path to the audio files
    '''

    paths = []

    manifest = open(manifest_path)
    items = manifest.readlines()
    
    for _, item in enumerate(items):
        item = json.loads(item)
        paths.append(item["audio_filepath"])

    return paths


def evaluate(model_path, manifest_path, output_path):
    '''
    Performs inference and returns a new manifest with the transcription from the model
    '''
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=model_path) # restore_from(restore_path)

    audio_path = get_audio_path(manifest_path)

    print("-"*50)
    print(f"Starting: {model_path}")
    print(len(audio_path))
    print(audio_path[:5])
    print("DATA INFO:\nLength: {} \nSamples: \n-{}".format(len(audio_path), '\n-'.join(audio_path[:5])))

    transcription = model.transcribe(audio_path)

    # save along with the transcription
    manifest = open(manifest_path)
    items = manifest.readlines()
    
    with open(output_path, 'w') as fout:

        for idx, item in enumerate(items):

            item = json.loads(item)
            item["pred_text"] = transcription[idx]

            json.dump(item, fout)
            fout.write('\n')

def clean(manifest_path):
    manifest = open(manifest_path)
    items = manifest.readlines()

    with open("t.json", 'w') as fout:
        for idx, item in enumerate(items):

            item = json.loads(item)

            try:
                if(len(item["text"]) == 0 or len(item["pred_text"]) == 0):
                    pass
                else:
                    json.dump(item, fout)
                    fout.write('\n')
            except:
                pass
        

if __name__ == "__main__":
    model_paths = ["RW_Finetune"]#, "Kinyarwanda ASR Run 3", "Kinyarwanda ASR Medium", "Finetune_2"]
    manifest_path = "/data/chanie/asr/manifest/dev_manifest_clean_decoded.json"
    # manifest_path = "/data/elamin/test_manifest_clean_decoded_cleaner.json"
    # manifest_path = "/data/chanie/asr/manifest/train_manifest_clean_decoded.json"

    MODEL_ROOT_PATH = "/data/chanie/asr/results"
    OUTPUT_ROOT_PATH = "/data/chanie/asr/manifest/final"

    for path in model_paths:
        full_path = os.path.join(MODEL_ROOT_PATH, path, "checkpoints", f"{path}.nemo")

        evaluate(full_path, manifest_path, f"{OUTPUT_ROOT_PATH}/{path}_mo_final.json")


