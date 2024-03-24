# Llama2 Music Generation

This repository focuses on fine-tuning Llama2, for music generation tasks. The process involves converting MIDI files to a text-based representation using `midi_textefinal.py` script. The text representation encodes musical notes as quadruplets, where each variable is separated by a colon (`:`). Within each quadruplet, the four variables represent:

- `p`: Pitch
- `v`: Velocity
- `d`: Duration
- `t`: Time

## Usage

### Converting MIDI Files to Text Representation

To convert MIDI files to text format, use the `midi_textefinal.py` script as follows:

1. Install dependencies:
```bash
pip install py_midicsv==4.0.0
```
2. Run the script:
```bash
python midi_textefinal.py /path/to/midis/input.mid output.txt
```
## Acknowledgments

- Special thanks to our tutor, [Ghadjeres](https://github.com/Ghadjeres), for his guidance and support throughout this project.

## Contributors

- Yessin Moakher ([@fegounna](https://github.com/fegounna))
- Marc Janthial ([@EchoSlash](https://github.com/EchoSlash))
- Charles Benichou--chaffanjon ([@chafflarch](https://github.com/chafflarch))
- Cyprien Laruelle
- Jules Cognon
- Damien Fromilhague




