from llmlingua import PromptCompressor

prompt = """We will provide you with pieces of music, which are sets of music notes represented by quadruplets, where each variable is separated by ":". Within each quadruplet, the 4 variables are p (pitch), v (velocity), d (duration), and t (time), each followed by their value (for example, you may encounter the quadruplet p52:v57:d1895:t212). Here is what each variable corresponds to:
- p corresponds to the pitch of the note (for example, p60 corresponds to a C3). The pitch difference between 2 notes corresponds to the number of semitones that separate them.
- v corresponds to the velocity of the note, meaning the intensity or force with which the note is played.
- d is the duration of the note, meaning the duration (in milliseconds) during which the note can be heard.
- t is the time that separates the instant when the note is played from the instant when the next note will be played. In our syntax, we have arbitrarily chosen, for the last note of each piece, to assign the value -1 to its variable t, in order to detect the end of the respective piece."""

llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", question="", target_token=200)

with open("../generated files/compressed_prompt.txt", "w") as file:
    file.write(compressed_prompt['compressed_prompt'])
file.close()