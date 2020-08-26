# 
# Decompression library using dynamic arithmetic coding
#
# Original Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#
# edited by Arxya to fit simpleLstmEncoderDecoder needs

import sys
import arithmeticcoding

class ArithmeticDecompress():

	def __init__(self, input_file, output_file):
		self.inputfile = input_file
		self.outputfile = output_file

	def start(self):
		self.inp = open(self.inputfile, "rb")
		self.out = open(self.outputfile, "wb")
		self.bitin = arithmeticcoding.BitInputStream(self.inp)
		self.freqsTable = arithmeticcoding.SimpleFrequencyTable([1] * 257)
		self.decoder = arithmeticcoding.ArithmeticDecoder(32, self.bitin)

	def stop(self):
		self.out.close()
		self.inp.close()

	""" 
	Decompresses the file bit by bit and writes into the output file.
	Also returns the symbol
	"""
	def decompress_next(self, freq_pred):
		symbol = self.decoder.read(self.freqsTable)
		self.freqsTable.set(symbol, freq_pred)
		self.out.write(bytes((symbol,)))

		return symbol




#def read_frequencies(bitin):
#	def read_int(n):
#		result = 0
#		for _ in range(n):
#			result = (result << 1) | bitin.read_no_eof()  # Big endian
#		return result
#
#	freqs = [read_int(32) for _ in range(256)]
#	freqs.append(1)  # EOF symbol
#	return arithmeticcoding.SimpleFrequencyTable(freqs)
#
#
#def decompress(freqs, bitin, out):
#	dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
#	while True:
#		symbol = dec.read(freqs)
#		if symbol == 256:  # EOF symbol
#			break
#		out.write(bytes((symbol,)))
#
#
## Command line main application function.
#def main(args):
#	# Handle command line arguments
#	if len(args) != 2:
#		sys.exit("Usage: python arithmeticdecompress.py InputFile OutputFile")
#	inputfile, outputfile = args
#
#	# Perform file decompression
#	with open(outputfile, "wb") as out, open(inputfile, "rb") as inp:
#		bitin = arithmeticcoding.BitInputStream(inp)
#		freqs = read_frequencies(bitin)
#		decompress(freqs, bitin, out)
#
#
## Main launcher
#if __name__ == "__main__":
#	main(sys.argv[1 : ])
