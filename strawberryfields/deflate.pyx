"""Deflate, unpacking the stream that closes a stored pitch track.

The payload behind the header is a zlib stream: a two byte wrapper, the compressed
blocks, and the adler32 of what they hold. This module reads one — the stream the
``pitch_track`` column carries — bit by bit, in the order RFC 1951 defines, instead of
handing the work to :mod:`zlib`.

The blocks come in the three shapes the format allows. A *stored* block is its bytes
plain. A *fixed* block writes every symbol through the code table the format fixes for
it. A *dynamic* block carries its own table, which is what a stream written at the
format's highest level looks like: the table arrives as the lengths of its codes, runs
of equal lengths arrive run-length encoded, and the symbols arrive one at a time
against the table that has just been built.

A stream that ends early, names a block type the format has no room for, reaches
further back than the bytes it holds, or fails its checksum raises :class:`ValueError`,
so a damaged column is found where it is read rather than further down.
"""

LENGTH_BASE = (3, 4, 5, 6, 7, 8, 9, 10,
               11, 13, 15, 17,
               19, 23, 27, 31,
               33, 41, 49, 57,
               65, 81, 97, 113,
               129, 157, 185, 213,
               258)

LENGTH_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
                5, 5, 5, 5,
                0)

DISTANCE_BASE = (1, 2, 3, 4, 5, 7, 9, 13,
                 17, 25, 33, 49,
                 65, 97, 129, 193,
                 257, 385, 513, 769,
                 1025, 1537, 2049, 3073,
                 4097, 6145, 8193, 12289,
                 16385, 24577)

DISTANCE_EXTRA = (0, 0, 0, 0, 1, 1, 2, 2,
                  3, 3, 4, 4,
                  5, 5, 6, 6,
                  7, 7, 8, 8,
                  9, 9, 10, 10,
                  11, 11, 12, 12,
                  13, 13)

CODE_LENGTH_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)

__all__ = ["decompress"]


def _canonical(lengths):
    """Build the decode table of a canonical Huffman code from its code lengths.

    :param lengths: Code length of every symbol, 0 where the symbol is absent.
    :type lengths: tuple[int, ...] or list[int]
    :return: One dictionary of code to symbol for each code length.
    :rtype: list[dict[int, int]]
    """
    counts = [0] * 16
    for length in lengths:
        if length:
            counts[length] += 1
    next_codes = [0] * 16
    code = 0
    for bits in range(1, 16):
        code = (code + counts[bits - 1]) << 1
        next_codes[bits] = code
    tables = [dict() for _ in range(16)]
    for symbol, length in enumerate(lengths):
        if length:
            tables[length][next_codes[length]] = symbol
            next_codes[length] += 1
    return tables


FIXED_LITERALS = _canonical(tuple([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8))

FIXED_DISTANCES = _canonical(tuple([5] * 32))


cdef class BitReader:
    """A bit stream packed least significant bit first, the way deflate writes."""

    cdef bytes source
    cdef const unsigned char[:] data
    cdef Py_ssize_t index
    cdef Py_ssize_t end
    cdef unsigned char current
    cdef int bit

    def __cinit__(self, payload, Py_ssize_t start, Py_ssize_t end):
        """Prepare a reader over one region of a stream.

        :param payload: The whole stream.
        :type payload: bytes
        :param start: Where the blocks begin.
        :type start: int
        :param end: Where the trailing checksum begins.
        :type end: int
        """
        self.source = bytes(payload)
        self.data = self.source
        self.index = start
        self.end = end
        self.current = 0
        self.bit = 8

    cdef int read_bit(self) except -1:
        """Read one bit, the next one in the byte.

        :return: The bit.
        :rtype: int
        :raises ValueError: if the stream ends before the bit arrives.
        """
        cdef int value
        if self.bit == 8:
            if self.index >= self.end:
                raise ValueError("packed pitch track ends inside its deflate stream")
            self.current = self.data[self.index]
            self.index += 1
            self.bit = 0
        value = (self.current >> self.bit) & 1
        self.bit += 1
        return value

    cdef unsigned int read_value(self, int count) except *:
        """Read ``count`` bits into one number, low bit first.

        :param count: How many bits to read.
        :type count: int
        :return: The number they form.
        :rtype: int
        :raises ValueError: if the stream ends before they arrive.
        """
        cdef unsigned int value = 0
        cdef int position
        for position in range(count):
            value |= <unsigned int> self.read_bit() << position
        return value

    cdef void align(self):
        """Leave the rest of the current byte unread."""
        self.bit = 8

    cdef object read_bytes(self, Py_ssize_t count):
        """Read whole bytes after the stream has been aligned.

        :param count: How many bytes to read.
        :type count: int
        :return: The bytes.
        :rtype: bytes
        :raises ValueError: if the stream ends before they arrive.
        """
        cdef object chunk
        if count < 0 or self.index + count > self.end:
            raise ValueError("packed pitch track ends inside its deflate stream")
        chunk = bytes(self.data[self.index:self.index + count])
        self.index += count
        return chunk


cdef int decode_symbol(BitReader reader, object tables) except -2:
    """Read one symbol against a canonical Huffman table, a bit at a time.

    :param reader: The stream to read from.
    :type reader: BitReader
    :param tables: One dictionary of code to symbol for each code length.
    :type tables: list[dict[int, int]]
    :return: The symbol.
    :rtype: int
    :raises ValueError: if the bits do not name a symbol of this table.
    """
    cdef unsigned int code = 0
    cdef Py_ssize_t length, count = len(tables)
    cdef object table
    cdef int value
    for length in range(1, count):
        value = reader.read_bit()
        code = (code << 1) | <unsigned int> value
        table = tables[length]
        if table and code in table:
            return <int> table[code]
    raise ValueError("packed pitch track holds an unusable deflate code")


def _read_tables(BitReader reader):
    """Read the three tables a dynamic block carries in front of its symbols.

    :param reader: The stream to read from.
    :type reader: BitReader
    :return: The literal and length table, then the distance table.
    :rtype: tuple[list[dict[int, int]], list[dict[int, int]]]
    """
    cdef int hlit = reader.read_value(5) + 257
    cdef int hdist = reader.read_value(5) + 1
    cdef int hclen = reader.read_value(4) + 4
    cdef int symbol

    code_lengths = [0] * 19
    for index in range(hclen):
        code_lengths[CODE_LENGTH_ORDER[index]] = reader.read_value(3)
    code_tables = _canonical(code_lengths)

    lengths = []
    while len(lengths) < hlit + hdist:
        symbol = decode_symbol(reader, code_tables)
        if symbol < 16:
            lengths.append(symbol)
        elif symbol == 16:
            if not lengths:
                raise ValueError("packed pitch track repeats a code length that is not there")
            lengths.extend([lengths[-1]] * (3 + reader.read_value(2)))
        elif symbol == 17:
            lengths.extend([0] * (3 + reader.read_value(3)))
        else:
            lengths.extend([0] * (11 + reader.read_value(7)))
    lengths = lengths[:hlit + hdist]
    return _canonical(lengths[:hlit]), _canonical(lengths[hlit:])


def _adler32(data):
    """The checksum the stream carries behind its blocks.

    :param data: What the stream inflated to.
    :type data: bytes
    :return: The checksum as a number.
    :rtype: int
    """
    a = 1
    b = 0
    for index, byte in enumerate(data):
        a += byte
        b += a
        if (index & 4095) == 4095:
            a %= 65521
            b %= 65521
    return ((b % 65521) << 16) | (a % 65521)


def decompress(payload):
    """Inflate the deflate stream a stored pitch track carries.

    :param payload: The whole zlib stream, header and checksum included.
    :type payload: bytes
    :return: The bytes the column holds: the variable-length differences.
    :rtype: bytes
    :raises ValueError: if the wrapper does not check out, a block is unusable, or
        the inflated bytes fail their adler32.
    """
    cdef bytes data = bytes(payload)
    cdef BitReader reader
    cdef int cmf, flg, final, block_type, length_index, distance_symbol
    cdef int symbol
    cdef Py_ssize_t length, distance, step

    if len(data) < 6:
        raise ValueError("packed pitch track holds a truncated deflate stream")
    cmf, flg = data[0], data[1]
    if (cmf & 0x0F) != 8:
        raise ValueError(f"packed pitch track is not a deflate stream: method {cmf & 0x0F}")
    if ((cmf << 8) | flg) % 31:
        raise ValueError("packed pitch track holds a deflate wrapper that does not check out")
    if flg & 0x20:
        raise ValueError("packed pitch track deflate stream carries a preset dictionary")

    reader = BitReader(data, 2, len(data) - 4)
    out = bytearray()
    while True:
        final = reader.read_bit()
        block_type = reader.read_value(2)
        if block_type == 0:
            reader.align()
            length = reader.read_value(16)
            if (length ^ 0xFFFF) != reader.read_value(16):
                raise ValueError("packed pitch track holds a stored block whose length "
                                 "does not check out")
            out.extend(reader.read_bytes(length))
        elif block_type in (1, 2):
            if block_type == 1:
                literal_tables, distance_tables = FIXED_LITERALS, FIXED_DISTANCES
            else:
                literal_tables, distance_tables = _read_tables(reader)
            while True:
                symbol = decode_symbol(reader, literal_tables)
                if symbol < 256:
                    out.append(symbol)
                elif symbol == 256:
                    break
                length_index = symbol - 257
                if length_index > 28:
                    raise ValueError("packed pitch track holds an unusable length code")
                length = (LENGTH_BASE[length_index]
                          + reader.read_value(LENGTH_EXTRA[length_index]))
                distance_symbol = decode_symbol(reader, distance_tables)
                if distance_symbol > 29:
                    raise ValueError("packed pitch track holds an unusable distance code")
                distance = (DISTANCE_BASE[distance_symbol]
                            + reader.read_value(DISTANCE_EXTRA[distance_symbol]))
                if distance > len(out):
                    raise ValueError("packed pitch track reaches further back than the "
                                     "bytes it holds allow")
                for step in range(length):
                    out.append(out[len(out) - distance])
        else:
            raise ValueError("packed pitch track holds an invalid deflate block type")

        if final:
            break

    inflated = bytes(out)
    if _adler32(inflated) != int.from_bytes(data[-4:], "big"):
        raise ValueError("packed pitch track deflate stream fails its adler32")
    return inflated
