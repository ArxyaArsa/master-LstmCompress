#
# Compression application using dynamic arithmetic coding
#
# Then use the corresponding arithmeticdecompress.py application to recreate the original input file.
# Note that the application uses an alphabet of 257 symbols - 256 symbols for the byte
# values and 1 symbol for the EOF marker. The compressed file format starts with a list
# of 256 symbol frequencies, and then followed by the arithmetic-coded data.
#
# Original Copyright (c) Project Nayuki
#
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#
# edited by Arxya to fit simpleLstmEncoderDecoder needs


import contextlib, sys
import arithmeticcoding

class ArithmeticCompress():

    def __init__(self, output_file):
        self.outputfile = output_file

    def start(self, dictionary_size=256):
        self.dictionary_size = dictionary_size
        self.bitout = arithmeticcoding.BitOutputStream(open(self.outputfile, "wb"))
        self.freqsTable = arithmeticcoding.SimpleFrequencyTable([1] * (dictionary_size + 1))
        self.encoder = arithmeticcoding.ArithmeticEncoder(32, self.bitout)

    def stop(self):
        self.encoder.write(self.freqsTable, self.dictionary_size)  # EOF
        self.encoder.finish()  # Flush remaining code bits
        self.bitout.close()

    def compress_next(self, new_freq_table_256, symbol_number):
        if isinstance(new_freq_table_256, (list, set)):
            new_table_copy = list(new_freq_table_256)
            new_table_copy.extend([1])
            self.freqsTable = arithmeticcoding.SimpleFrequencyTable(new_table_copy)

        self.encoder.write(self.freqsTable, symbol_number)
        ## set new frequency for the symbol
        #self.freqsTable.set(symbol_number, freq_pred)


## Returns a frequency table based on the bytes in the given file.
## Also contains an extra entry for symbol 256, whose frequency is set to 0.
# def get_frequencies(self, filepath):
#    freqs = arithmeticcoding.SimpleFrequencyTable([0] * 257)
#    with open(filepath, "rb") as input:
#        while True:
#            b = input.read(1)
#            if len(b) == 0:
#                break
#            freqs.increment(b[0])
#    return freqs

# def write_frequencies(self, bitout, freqs):
#    for i in range(256):
#        write_int(bitout, 32, freqs.get(i))

# def compress(self, freqs, inp, bitout):
#    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
#    while True:
#        symbol = inp.read(1)
#        if len(symbol) == 0:
#            break
#        enc.write(freqs, symbol[0])
#    enc.write(freqs, 256)  # EOF
#    enc.finish()  # Flush remaining code bits


## Writes an unsigned integer of the given bit width to the given stream.
#def write_int(bitout, numbits, value):
#    for i in reversed(range(numbits)):
#        bitout.write((value >> i) & 1)  # Big endian
#
#
## Command line main application function.
#def main(args):
#    # Handle command line arguments
#    if len(args) != 2:
#        sys.exit("Usage: python arithmeticcompress.py InputFile OutputFile")
#    inputfile, outputfile = args
#
#    # Read input file once to compute symbol frequencies
#    freqs = get_frequencies(inputfile)
#    freqs.increment(256)  # EOF symbol gets a frequency of 1
#
#    # Read input file again, compress with arithmetic coding, and write output file
#    with open(inputfile, "rb") as inp, \
#            contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
#        write_frequencies(bitout, freqs)
#        compress(freqs, inp, bitout)
#
#
## Main launcher
#if __name__ == "__main__":
#    main(sys.argv[1:])