class RepetitionTable:

    def __init__(self):

        self.table = {}

    def lookup(self, zobrist_hash):
        """Look up an entry in the table by zobrist_hash and return how many times position has occurred."""
        
        return self.table.get(zobrist_hash)
    
    def store(self, zobrist_hash):
        """Increment count of the position."""

        if zobrist_hash in self.table:
            self.table[zobrist_hash] += 1
        else:
            self.table[zobrist_hash] = 1

    def remove(self, zobrist_hash):
        "Decrement count of the position"

        if self.table[zobrist_hash] > 1:
            self.table[zobrist_hash] -= 1
        else:
            del self.table[zobrist_hash]

    def clear(self):
        """Clear the entire table."""
        
        self.table.clear()