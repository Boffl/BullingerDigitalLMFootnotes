from generate_instruction import *

with open(os.path.join(DATA_PATH, "downsized_letters", "10454.xml")) as infile:
    test_letter = infile.read()

test_fn = 18

print(len(test_letter))
print((get_letter_context(test_letter, test_fn)))