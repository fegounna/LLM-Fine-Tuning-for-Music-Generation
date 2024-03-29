# Llama2 Music Generation

This repository focuses on fine-tuning Llama2 for music generation tasks. The process involves converting MIDI files to a text-based representation using `midi_textefinal.py` script. The text representation encodes musical notes as quadruplets, where each variable is separated by a colon (`:`). Within each quadruplet, the four variables represent:

- `p`: Pitch
- `v`: Velocity
- `d`: Duration
- `t`: Time

After converting MIDI files to text, we proceed to fine-tune the Llama2 model for improved music generation performance.

## Usage

### Converting a MIDI File to Text Representation

To convert a MIDI file to text format, use the `midi_textefinal.py` script as follows:

1. Install dependencies:
```bash
pip install py_midicsv==4.0.0
```
2. Run the script:
```bash
python midi_textefinal.py ./input.mid output.txt
```

### Converting Text to MIDI File

To convert text to a MIDI file, use the `text_to_midi.py` script as follows:

1. Install dependencies:
```bash
pip install mido==1.3.2
```
2. Run the script:
```bash
python text_to_midi.py ./input.txt output.mid
```
### Generating Music
To complete the generation of music from a text representation, use the `inference.py` script as follows:
1. Install dependencies:
```bash
pip install -r requirements_training.txt
```
2. Run the script:
```bash
python inference.py ./input.txt output.txt
```
## Example:
Given the Prompt :


https://github.com/fegounna/LLM-Fine-Tuning-for-Music-Generation/assets/130999532/1b53d5de-0b93-4ce8-89cc-200fb1d13171

We have the completion :

https://github.com/fegounna/LLM-Fine-Tuning-for-Music-Generation/assets/130999532/27ebf1d4-08ee-4bdf-b3cb-067b936a8445





## Acknowledgments

- Special thanks to our tutor, [Ghadjeres](https://github.com/Ghadjeres), for his guidance and support throughout this project.

## Contributors

- Yessin Moakher ([fegounna](https://github.com/fegounna))
- Marc Janthial ([EchoSlash](https://github.com/EchoSlash))
- Charles Benichou--chaffanjon ([chafflarch](https://github.com/chafflarch))
- Cyprien Laruelle
- Jules Cognon
- Damien Fromilhague




