# Llama2 Music Generation

<img src="https://github.com/fegounna/LLM-Fine-Tuning-for-Music-Generation/assets/130999532/c8c4f29c-7f24-4840-ac15-da371894b925" width="400">


This repository focuses on fine-tuning Llama2 for music generation tasks. The process involves converting MIDI files to a text-based representation using `midi_textefinal.py` script. The text representation encodes musical notes as quadruplets, where each variable is separated by a colon (`:`). Within each quadruplet, the four variables represent:

- `p`: Pitch
- `v`: Velocity
- `d`: Duration
- `t`: Time

After converting MIDI files to text, we proceed to fine-tune the Llama2 model for improved music generation performance.

## Usage

### Converting a MIDI File to Text Representation

To convert a MIDI file to text format, use the `midi_textefinal.py` 

### Converting Text to MIDI File

To convert text to a MIDI file, use the `text_to_midi.py` 
### Generating Music

## Example
Given the Prompt :


https://github.com/fegounna/LLM-Fine-Tuning-for-Music-Generation/assets/130999532/1b53d5de-0b93-4ce8-89cc-200fb1d13171

We have the completion :

https://github.com/fegounna/LLM-Fine-Tuning-for-Music-Generation/assets/130999532/447f61fb-10bc-43f9-ba5d-b3793919f352






## Acknowledgments

- Special thanks to our tutor, [Ghadjeres](https://github.com/Ghadjeres), for his guidance and support throughout this project.

## Contributors

- Yessin Moakher ([fegounna](https://github.com/fegounna))
- Marc Janthial ([EchoSlash](https://github.com/EchoSlash))
- Charles Benichou--chaffanjon ([chafflarch](https://github.com/chafflarch))
- Cyprien Laruelle
- Jules Cognon
- Damien Fromilhague




