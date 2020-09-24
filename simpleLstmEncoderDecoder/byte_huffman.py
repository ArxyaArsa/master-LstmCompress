
import heapq
import os


class HuffmanCoding:

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.dictionary_size = -1

    def make_heap(self, frequency):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        if not isinstance(frequency, (list, set)):
            frequency = [(i) for i in range(self.dictionary_size)]

        for i in range(len(frequency)):
            node = self.HeapNode(i, frequency[i])
            heapq.heappush(self.heap, node)
        node = self.HeapNode(len(frequency), float(0))
        heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

class HuffmanEncoder(HuffmanCoding):

    def __init__(self, output_file_path):
        super().__init__()
        self.encoded = ""
        self.output_file = output_file_path

    def start(self, dictionary_size=256):
        self.dictionary_size = dictionary_size

    def compress_next(self, freq_table, symbol):
        self.make_heap(freq_table)
        self.merge_nodes()
        self.make_codes()
        self.encoded += self.codes[symbol]

    def compress_next_simple(self, symbol):
        self.encoded += self.codes[symbol]

    def stop(self):
        padded = self.pad_encoded_text(self.encoded)
        b = self.get_byte_array(padded)
        with open(self.output_file, 'wb') as output:
            output.write(bytes(b))

        #with open('./data/huffman_encoder.txt', 'wt') as log:
        #    log.write('padded:   ' + padded + '\n')
        #    log.write('unpadded: ________' + self.encoded + '\n')

class HuffmanDecoder(HuffmanCoding):

    def __init__(self, input_file_path, output_file_path):
        super().__init__()
        self.input_file = input_file_path
        self.output_file = output_file_path

    def start(self, dictionary_size=256):
        self.dictionary_size = dictionary_size

        with open(self.input_file, 'rb') as file:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            self.encoded = self.remove_padding(bit_string)

            #with open('./data/huffman_decoder.txt', 'wt') as log:
            #    log.write('padded:   ' + bit_string + '\n')
            #    log.write('unpadded: ________' + self.encoded + '\n')

        self.encoded_len = len(self.encoded)
        self.out = open(self.output_file, "wb")
        self.position = 0


    def decompress_next(self, freq_table):
        self.make_heap(freq_table)
        self.merge_nodes()
        self.make_codes()

        symbol = -1

        stop = False
        code = ""
        while not stop:
            if self.position >= self.encoded_len:
                return self.dictionary_size

            code += self.encoded[self.position]
            self.position += 1

            if code in self.reverse_mapping:
                character = self.reverse_mapping[code]
                symbol = character
                stop = True

        self.out.write(bytes((symbol,)))

        return symbol

    def stop(self):
        self.out.close()

