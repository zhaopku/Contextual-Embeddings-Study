path = 'snli_1.0_train.txt'

with open(path, 'r') as f_in, open('text.txt', 'w') as f_out:
	lines = f_in.readlines()[1:]
	sents = []
	lines = list(set(lines))

	new_lines = []
	for line in lines:
		new_lines.append(line.strip().lower())
	lines = new_lines

	for line in lines:
		splits = line.split('\t')

		sents.append(splits[5].strip())
		sents.append(splits[6].strip())


	print(len(sents))

	for s in sents:
		f_out.write(s+'\n')

