This repository focuses on fine-tuning Llama2, for music generation tasks. The process involves converting MIDI files to a text-based representation using midi_textefinal.py script. The text representation encodes musical notes as quadruplets, where each variable is separated by a colon (:). Within each quadruplet, the four variables represent:

p: Pitch
v: Velocity
d: Duration
t: Time
